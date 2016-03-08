extern "C" {
#include "../thnets.h"
};

#ifdef HAVEHALF
#include "cuda_fp16.h"
#endif

#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

// Use 1024 threads per block, which requires cuda sm_2x or above
static int CUDA_NUM_THREADS = 1024;

// CUDA: number of blocks for threads.
static int GET_BLOCKS(const int N)
{
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

__global__ void MaxUnpoolForward(const int nthreads, const float *bottom_data, const float *bottom_mask, 
	const int num, const int channels, const int iheight, const int iwidth, const int oheight, const int owidth, float *top_data)
{
	CUDA_KERNEL_LOOP(index, nthreads)
	{ //index here indices the input pixels
		int c = (index / iwidth / iheight) % channels;
		int n = index / iwidth / iheight / channels;
		top_data += (n*channels + c)*oheight*owidth;
		int maxind = bottom_mask[index]-1;

		top_data[maxind] = bottom_data[index];
	}
}

__global__ void MaxUnpoolForwardH(const int nthreads, const __half *bottom_data, const float *bottom_mask, 
	const int num, const int channels, const int iheight, const int iwidth, const int oheight, const int owidth, __half *top_data)
{
	CUDA_KERNEL_LOOP(index, nthreads)
	{ //index here indices the input pixels
		int c = (index / iwidth / iheight) % channels;
		int n = index / iwidth / iheight / channels;
		top_data += (n*channels + c)*oheight*owidth;
		int maxind = bottom_mask[index]-1;

		top_data[maxind] = bottom_data[index];
	}
}

THFloatTensor *cunn_SpatialMaxUnpooling_updateOutput(struct module *module, THFloatTensor *input)
{

	THFloatTensor *output = module->output;
	THFloatTensor *indices = 0;
	int i;
	long owidth, oheight;

	if( !(input->nDimension == 3 || input->nDimension == 4) )
		THError("3D or 4D (batch) tensor expected");

	struct network *net = module->net;
	for(i = 0; i < net->nelem; i++)
		if(net->modules[i].type == MT_SpatialMaxPooling &&
			net->modules[i].nnmodule == module->SpatialMaxUnpooling.pooling)
		{
			owidth = net->modules[i].SpatialMaxPooling.iwidth;
			oheight = net->modules[i].SpatialMaxPooling.iheight;
			indices = net->modules[i].SpatialMaxPooling.indices;
			break;
		}
	if (!indices || !THFloatTensor_isSameSizeAs(input, indices))
		THError("Invalid input size w.r.t current indices size");

	long nInputCols, nInputRows, nInputPlane, batchSize;

	if (input->nDimension == 3)
	{
		nInputCols = input->size[2];
		nInputRows = input->size[1];
		nInputPlane = input->size[0];
		batchSize = 1;
	} else {
		nInputCols = input->size[3];
		nInputRows = input->size[2];
		nInputPlane = input->size[1];
		batchSize = input->size[0];
	}

	THCudaTensor_resize4d(output, batchSize, nInputPlane, oheight, owidth);
	cudaMemset(THFloatTensor_data(output), 0, (floattype == CUDNN_DATA_HALF ? 2 : 4) * THFloatTensor_nElement(output));

	int count = THFloatTensor_nElement(input);

#ifdef HAVEHALF
	if(floattype == CUDNN_DATA_HALF)
		MaxUnpoolForwardH <<< GET_BLOCKS(count), CUDA_NUM_THREADS >>>
			(count, (__half *)THFloatTensor_data(input), THFloatTensor_data(indices),
			batchSize, nInputPlane, nInputRows, nInputCols, oheight, owidth, (__half *)THFloatTensor_data(output));
	else
#endif
		MaxUnpoolForward <<< GET_BLOCKS(count), CUDA_NUM_THREADS >>>
			(count, THFloatTensor_data(input), THFloatTensor_data(indices),
			batchSize, nInputPlane, nInputRows, nInputCols, oheight, owidth, THFloatTensor_data(output));

	if(input->nDimension == 3)
		THCudaTensor_resize3d(output, nInputPlane, oheight, owidth);

	// check for errors
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		THError("Error in SpatialMaxUnpooling.updateOutput: %s", cudaGetErrorString(err));
	return output;
}
