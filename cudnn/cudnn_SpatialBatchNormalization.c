#include "../thnets.h"

THFloatTensor *cudnn_SpatialBatchNormalization_updateOutput(struct module *module, THFloatTensor *input)
{
	THFloatTensor *output = module->output;
	THFloatTensor *running_mean = module->SpatialBatchNormalization.running_mean;
	THFloatTensor *running_var = module->SpatialBatchNormalization.running_var;
	THFloatTensor *weight = module->SpatialBatchNormalization.weight;
	THFloatTensor *bias = module->SpatialBatchNormalization.bias;

	double eps = module->SpatialBatchNormalization.eps;

	cudnnTensorDescriptor_t dinput, doutput, dscalebiasmeanvar;
	float one = 1, zero = 0;

	THCudaTensor_resizeAs(output, input);
	errcheck(THcudnn_TensorDescriptor(&dinput, input));
	errcheck(THcudnn_TensorDescriptor(&doutput, output));
	errcheck(cudnnCreateTensorDescriptor(&dscalebiasmeanvar));
	errcheck(cudnnSetTensor4dDescriptor(dscalebiasmeanvar, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1,
		THFloatTensor_nElement(running_mean), 1, 1));

	errcheck(cudnnBatchNormalizationForwardInference(THcudnn_getHandle(), CUDNN_BATCHNORM_SPATIAL,
		&one, &zero, dinput, THFloatTensor_data(input),
		doutput, THFloatTensor_data(output), dscalebiasmeanvar, THFloatTensor_data(weight),
		THFloatTensor_data(bias), THFloatTensor_data(running_mean), THFloatTensor_data(running_var), eps));

	cudnnDestroyTensorDescriptor(dinput);
	cudnnDestroyTensorDescriptor(doutput);
	cudnnDestroyTensorDescriptor(dscalebiasmeanvar);
	return output;
}
