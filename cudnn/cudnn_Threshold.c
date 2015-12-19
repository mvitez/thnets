#include "../thnets.h"

THFloatTensor *cudnn_Threshold_updateOutput(struct module *module, THFloatTensor *input)
{
	THFloatTensor *output = module->output;
	cudnnTensorDescriptor_t dinput, doutput;
	int inplace = module->Threshold.inplace;
	float one = 1, zero = 0;

	errcheck(THcudnn_TensorDescriptor(&dinput, input));
	if(inplace)
		THFloatTensor_set(output, input);
	else THCudaTensor_resize4d(output, input->size[0], input->size[1], input->size[2], input->size[3]);
	errcheck(THcudnn_TensorDescriptor(&doutput, output));

	errcheck(cudnnActivationForward(THcudnn_getHandle(), CUDNN_ACTIVATION_RELU, &one, dinput, THFloatTensor_data(input), &zero,
		doutput, THFloatTensor_data(output)));

	cudnnDestroyTensorDescriptor(dinput);
	cudnnDestroyTensorDescriptor(doutput);
	return output;
}
