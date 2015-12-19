#include "../thnets.h"

THFloatTensor *cudnn_SpatialMaxPooling_updateOutput(struct module *module, THFloatTensor *input)
{
	int kW = module->SpatialMaxPooling.kW;
	int kH = module->SpatialMaxPooling.kH;
	int dW = module->SpatialMaxPooling.dW;
	int dH = module->SpatialMaxPooling.dH;
	int padW = module->SpatialMaxPooling.padW;
	int padH = module->SpatialMaxPooling.padH;

	THFloatTensor *output = module->output;
	cudnnTensorDescriptor_t dinput, doutput;
	cudnnPoolingDescriptor_t dpool;
	float one = 1, zero = 0;
	int sizes[4];

	errcheck(THcudnn_TensorDescriptor(&dinput, input));
	errcheck(cudnnCreatePoolingDescriptor(&dpool));
	errcheck(cudnnSetPooling2dDescriptor(dpool, CUDNN_POOLING_MAX, kH, kW, padH, padW, dH, dW));
	errcheck(cudnnGetPoolingNdForwardOutputDim(dpool, dinput, 4, sizes));
	THCudaTensor_resize4d(output, sizes[0], sizes[1], sizes[2], sizes[3]);
	errcheck(THcudnn_TensorDescriptor(&doutput, output));

	errcheck(cudnnPoolingForward(THcudnn_getHandle(), dpool, &one, dinput, THFloatTensor_data(input), &zero,
		doutput, THFloatTensor_data(output)));

	cudnnDestroyTensorDescriptor(dinput);
	cudnnDestroyTensorDescriptor(doutput);
	cudnnDestroyPoolingDescriptor(dpool);
	return output;
}
