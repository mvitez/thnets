#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../thnets.h"

THFloatTensor *cudnn_SpatialConvolution_updateOutput(struct module *module, THFloatTensor *input)
{
	int kW = module->SpatialConvolution.kW;
	int kH = module->SpatialConvolution.kH;
	int dW = module->SpatialConvolution.dW;
	int dH = module->SpatialConvolution.dH;
	int padW = module->SpatialConvolution.padW;
	int padH = module->SpatialConvolution.padH;
	int nInputPlane  = module->SpatialConvolution.nInputPlane;
	int nOutputPlane = module->SpatialConvolution.nOutputPlane;

	THFloatTensor *weight = module->SpatialConvolution.weight;
	THFloatTensor *bias = module->SpatialConvolution.bias;
	THFloatTensor *output = module->output;

	int sizes[4];
	int pad[2], filterStride[2], upscale[2];
	cudnnTensorDescriptor_t dinput, dbias, doutput;
	cudnnConvolutionDescriptor_t dconv;
	cudnnFilterDescriptor_t dweight;
	float one = 1, zero = 0;
	size_t reqwssize;
	static void *ws;
	static size_t wssize;
	static const int alg = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;

	pad[0] = padH;
	pad[1] = padW;
	filterStride[0] = dH;
	filterStride[1] = dW;
	upscale[0] = 1;
	upscale[1] = 1;

	if(input->nDimension <= 2)
	{
		// Here we use the SpatialConvolution module to perform a linear transformation
		errcheck(cudnnCreateTensorDescriptor(&dinput));
		if(input->nDimension == 1)
			errcheck(cudnnSetTensor4dDescriptor(dinput, CUDNN_TENSOR_NCHW, floattype, 1, input->size[0], 1, 1));
		else errcheck(cudnnSetTensor4dDescriptor(dinput, CUDNN_TENSOR_NCHW, floattype, input->size[0], input->size[1], 1, 1));
	} else errcheck(THcudnn_TensorDescriptor(&dinput, input));
	errcheck(cudnnCreateFilterDescriptor(&dweight));
	errcheck(cudnnSetFilter4dDescriptor(dweight, floattype, nOutputPlane, nInputPlane, kH, kW));
	errcheck(cudnnCreateTensorDescriptor(&dbias));
	errcheck(cudnnSetTensor4dDescriptor(dbias, CUDNN_TENSOR_NCHW, floattype, 1, bias->size[0], 1, 1));
	errcheck(cudnnCreateConvolutionDescriptor(&dconv));
	errcheck(cudnnSetConvolutionNdDescriptor(dconv, 2, pad, filterStride, upscale, CUDNN_CROSS_CORRELATION, floattype));
	errcheck(cudnnGetConvolutionNdForwardOutputDim(dconv, dinput, dweight, 4, sizes));
	THCudaTensor_resize4d(output, sizes[0], sizes[1], sizes[2], sizes[3]);
	errcheck(THcudnn_TensorDescriptor(&doutput, output));
	if(alg == CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM || alg == CUDNN_CONVOLUTION_FWD_ALGO_GEMM ||
		alg == CUDNN_CONVOLUTION_FWD_ALGO_FFT || alg == CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING)
	{
		errcheck(cudnnGetConvolutionForwardWorkspaceSize(THcudnn_getHandle(), dinput, dweight, dconv, doutput, alg, &reqwssize));
		if(reqwssize > wssize)
		{
			wssize = reqwssize;
			errcheck(cudaMalloc(&ws, reqwssize));
		}			
	}
	errcheck(cudnnConvolutionForward(THcudnn_getHandle(), &one, dinput, THFloatTensor_data(input),
		dweight, THFloatTensor_data(weight), dconv, alg, ws, wssize, &zero,
		doutput, THFloatTensor_data(output)));
	errcheck(cudnnAddTensor_v3(THcudnn_getHandle(), &one, dbias, THFloatTensor_data(bias),
		&one, doutput, THFloatTensor_data(output)));
	cudnnDestroyTensorDescriptor(dinput);
	cudnnDestroyFilterDescriptor(dweight);
	cudnnDestroyTensorDescriptor(dbias);
	cudnnDestroyTensorDescriptor(doutput);
	cudnnDestroyConvolutionDescriptor(dconv);
	return output;
}
