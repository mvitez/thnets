extern "C" {
#include "../thnets.h"
};

#include "col2im.h"

THFloatTensor *cunn_SpatialFullConvolution_updateOutput(struct module *module, THFloatTensor *input)
{
	int dW = module->SpatialFullConvolution.dW;
	int dH = module->SpatialFullConvolution.dH;
	int kW = module->SpatialFullConvolution.kW;
	int kH = module->SpatialFullConvolution.kH;
	int padW = module->SpatialFullConvolution.padW;
	int padH = module->SpatialFullConvolution.padH;
	int adjW = module->SpatialFullConvolution.adjW;
	int adjH = module->SpatialFullConvolution.adjH;

	THFloatTensor *weight = module->SpatialFullConvolution.weight;
	THFloatTensor *bias = module->SpatialFullConvolution.bias;
	THFloatTensor *output = module->output;
	THFloatTensor *columns = module->SpatialFullConvolution.columns;
	THFloatTensor *ones = module->SpatialFullConvolution.ones;

	int nInputPlane = weight->size[0];
	int nOutputPlane = weight->size[1];

	if(input->nDimension != 3 && input->nDimension != 4)
		THError("3D or 4D (batch mode) tensor is expected");

	int batch = 1;
	if (input->nDimension == 3)
	{
		if(input->size[0] != nInputPlane)
			THError("input channels and nInputPlane dont match");
		// Force batch
		batch = 0;
		THCudaTensor_resize4d(input, 1, input->size[0], input->size[1], input->size[2]);
	} else if(input->size[1] != nInputPlane)
		THError("input channels and nInputPlane dont match");

	long inputWidth   = input->size[3];
	long inputHeight  = input->size[2];
	long outputWidth  = (inputWidth - 1) * dW - 2*padW + kW + adjW;
	long outputHeight = (inputHeight - 1) * dH - 2*padH + kH + adjH;

	// Batch size + input planes
	long batchSize = input->size[0];

	// Resize output
	THCudaTensor_resize4d(output, batchSize, nOutputPlane, outputHeight, outputWidth);

	// Resize temporary columns
	THCudaTensor_resize2d(columns, nOutputPlane*kW*kH, inputHeight*inputWidth);

	// Define a buffer of ones, for bias accumulation
	// Note: this buffer can be shared with other modules, it only ever gets increased,
	// and always contains ones.
	if (ones->nDimension != 2 || ones->size[0]*ones->size[1] < outputHeight*outputWidth)
	{
		// Resize plane and fill with ones...
		THCudaTensor_resize2d(ones, outputHeight, outputWidth);
		THCudaTensor_Ones(ones);
	}

	// For each elt in batch, do:
	for (int elt = 0; elt < batchSize; elt ++)
	{
		// Matrix multiply per output:
		THFloatTensor *input_n = THFloatTensor_newSelect(input, 0, elt);
		THFloatTensor *output_n = THFloatTensor_newSelect(output, 0, elt);
		float *indata, *outdata;
#ifdef HAVEHALF
		if(floattype == CUDNN_DATA_HALF)
			indata = (float *)((unsigned short *)input_n->storage->data + input_n->storageOffset);
		else
#endif
		indata = THFloatTensor_data(input_n);

#ifdef HAVEHALF
		if(floattype == CUDNN_DATA_HALF)
			outdata = (float *)((unsigned short *)output_n->storage->data + output_n->storageOffset);
		else
#endif
		outdata = THFloatTensor_data(output_n);

		// M,N,K are dims of matrix A and B
		// (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
		long m = weight->size[1] * weight->size[2] * weight->size[3];
		long n = columns->size[1];
		long k = weight->size[0];

		// Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
		THCudaBlas_gemm(
			'n', 't',
			n, m, k,
			1,
			indata, n,
			THFloatTensor_data(weight), m,
			0,
			THFloatTensor_data(columns), n
		);

		// Unpack columns back into input:
#ifdef HAVEHALF
		if(floattype == CUDNN_DATA_HALF)
			col2imH((__half *)THFloatTensor_data(columns),
			nOutputPlane, outputHeight, outputWidth, kH, kW, padH, padW, dH, dW,
			(__half *)outdata);
		else
#endif
			col2im(THFloatTensor_data(columns),
			nOutputPlane, outputHeight, outputWidth, kH, kW, padH, padW, dH, dW,
			THFloatTensor_data(output_n));

		// Do Bias after:
		// M,N,K are dims of matrix A and B
		// (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
		long m_ = nOutputPlane;
		long n_ = outputHeight * outputWidth;
		long k_ = 1;

		// Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
		THCudaBlas_gemm(
			't', 'n',
			n_, m_, k_,
			1,
			THFloatTensor_data(ones), k_,
			THFloatTensor_data(bias), k_,
			1,
			outdata, n_
		);
		THFloatTensor_free(input_n);
		THFloatTensor_free(output_n);
	}

	// Resize output
	if (batch == 0)
	{
		THCudaTensor_resize3d(output, nOutputPlane, outputHeight, outputWidth);
		THCudaTensor_resize3d(input, nInputPlane, inputHeight, inputWidth);
	}

	// return output
	return output;
}
