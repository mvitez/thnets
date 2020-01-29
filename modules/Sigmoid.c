#include "../thnets.h"
#include <math.h>

#ifdef ONNX
void onnxload_Sigmoid(const void *graph, struct module *m, int nodeidx)
{
	m->updateOutput = nn_Sigmoid_updateOutput;
	m->type = MT_Sigmoid;
}
#endif

THNTensor *nn_Sigmoid_updateOutput(struct module *module, THNTensor *input)
{
	THNTensor *output = module->output;
	float *input_data, *output_data;
	long i, n = THNTensor_nElement(input);

	THNTensor_resizeAs(output, input);

	input_data = THNTensor_data(input);
	output_data = THNTensor_data(output);

#pragma omp parallel for
	for(i = 0; i < n; i++)
		output_data[i] = 1.0f / (1.0f + expf(-input_data[i]));

	return output;
}
