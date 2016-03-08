extern "C" {
#include "../thnets.h"
};

#include <stdio.h>
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

__global__ void MaxPoolForward(const int nthreads, const float *bottom_data,
	const int num, const int channels, const int height,
	const int width, const int pooled_height, const int pooled_width,
	const int kernel_h, const int kernel_w, const int stride_h,
	const int stride_w, const int pad_h, const int pad_w, float *top_data,
	float *top_mask)
{
	CUDA_KERNEL_LOOP(index, nthreads)
	{
		int pw = index % pooled_width;
		int ph = (index / pooled_width) % pooled_height;
		int c = (index / pooled_width / pooled_height) % channels;
		int n = index / pooled_width / pooled_height / channels;
		int hstart = ph * stride_h - pad_h;
		int wstart = pw * stride_w - pad_w;
		int hend = min(hstart + kernel_h, height);
		int wend = min(wstart + kernel_w, width);
		hstart = max(hstart, 0);
		wstart = max(wstart, 0);
		float maxval = -FLT_MAX;
		int maxidx = -1;
		bottom_data += (n * channels + c) * height * width;
		for (int h = hstart; h < hend; ++h)
		{
			for (int w = wstart; w < wend; ++w)
			{
				if (bottom_data[h * width + w] > maxval)
				{
					maxidx = h * width + w;
					maxval = bottom_data[maxidx];
				}
			}
		}
		top_data[index] = maxval;
		top_mask[index] = maxidx + 1;
	}
}

#ifdef HAVEHALF
__global__ void MaxPoolForwardH(const int nthreads, const __half *bottom_data,
	const int num, const int channels, const int height,
	const int width, const int pooled_height, const int pooled_width,
	const int kernel_h, const int kernel_w, const int stride_h,
	const int stride_w, const int pad_h, const int pad_w, __half *top_data,
	float *top_mask)
{
	CUDA_KERNEL_LOOP(index, nthreads)
	{
		int pw = index % pooled_width;
		int ph = (index / pooled_width) % pooled_height;
		int c = (index / pooled_width / pooled_height) % channels;
		int n = index / pooled_width / pooled_height / channels;
		int hstart = ph * stride_h - pad_h;
		int wstart = pw * stride_w - pad_w;
		int hend = min(hstart + kernel_h, height);
		int wend = min(wstart + kernel_w, width);
		hstart = max(hstart, 0);
		wstart = max(wstart, 0);
		__half maxval = __float2half(-6e4);
		int maxidx = -1;
		bottom_data += (n * channels + c) * height * width;
		for (int h = hstart; h < hend; ++h)
		{
			for (int w = wstart; w < wend; ++w)
			{
				if (__hgt(bottom_data[h * width + w],maxval) )
				{
					maxidx = h * width + w;
					maxval = bottom_data[maxidx];
				}
			}
		}
		top_data[index] = maxval;
		top_mask[index] = maxidx + 1;
	}
}
#endif

THFloatTensor *cunn_SpatialMaxPooling_updateOutput(struct module *module, THFloatTensor *input)
{
	int kW = module->SpatialMaxPooling.kW;
	int kH = module->SpatialMaxPooling.kH;
	int dW = module->SpatialMaxPooling.dW;
	int dH = module->SpatialMaxPooling.dH;
	int padW = module->SpatialMaxPooling.padW;
	int padH = module->SpatialMaxPooling.padH;
	int ceil_mode = module->SpatialMaxPooling.ceil_mode;
	THFloatTensor *output = module->output;
	THFloatTensor *indices = module->SpatialMaxPooling.indices;

	long nInputCols, nInputRows, nInputPlane, batchSize;
	long nOutputCols, nOutputRows;

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
	module->SpatialMaxPooling.iwidth = nInputCols;
	module->SpatialMaxPooling.iheight = nInputRows;

	if( !(nInputCols >= kW - padW && nInputRows >= kH - padH) )
		THError("input image smaller than kernel size");
	if( !(kW/2 >= padW && kH/2 >= padH) )
		THError("pad should be smaller than half of kernel size");

	if(ceil_mode) {
		nOutputCols = ceil((float)(nInputCols - kW + 2*padW) / dW) + 1;
		nOutputRows = ceil((float)(nInputRows - kH + 2*padH) / dH) + 1;
	} else {
		nOutputCols = floor((float)(nInputCols - kW + 2*padW) / dW) + 1;
		nOutputRows = floor((float)(nInputRows - kH + 2*padH) / dH) + 1;
	}

	if (padW || padH)
	{
		// ensure that the last pooling starts inside the image
		if ((nOutputRows - 1)*dH >= nInputRows + padH)
			--nOutputRows;
		if ((nOutputCols  - 1)*dW >= nInputCols  + padW)
			--nOutputCols;
	}

	float *input_data = THFloatTensor_data(input);

	THCudaTensor_resize4d(output, batchSize, nInputPlane, nOutputRows, nOutputCols);

	int count = THFloatTensor_nElement(output);

#ifdef HAVEHALF
	if(floattype == CUDNN_DATA_HALF)
	{
		floattype = CUDNN_DATA_FLOAT;
		THCudaTensor_resizeAs(indices, output);
		floattype = CUDNN_DATA_HALF;
		__half *h_input_data = (__half *)input_data;
		float *indices_data = THFloatTensor_data(indices);
		__half *h_output_data = (__half *)THFloatTensor_data(output);

		MaxPoolForwardH <<< GET_BLOCKS(count), CUDA_NUM_THREADS >>>
			(count, h_input_data,
			batchSize, nInputPlane, nInputRows, nInputCols, nOutputRows, nOutputCols,
			kH, kW, dH, dW, padH, padW, h_output_data, indices_data);
	} else
#endif
	{
		THCudaTensor_resizeAs(indices, output);
		float *indices_data = THFloatTensor_data(indices);
		float *output_data = THFloatTensor_data(output);

		MaxPoolForward <<< GET_BLOCKS(count), CUDA_NUM_THREADS >>>
			(count, input_data,
			batchSize, nInputPlane, nInputRows, nInputCols, nOutputRows, nOutputCols,
			kH, kW, dH, dW, padH, padW, output_data, indices_data);
	}

	if(input->nDimension == 3)
		THCudaTensor_resize3d(output, nInputPlane, nOutputRows, nOutputCols);

	// check for errors
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		THError("error in SpatialMaxPooling.updateOutput: %s", cudaGetErrorString(err));
	return output;
}	
