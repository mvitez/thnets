#include "../thnets.h"
#include <math.h>

#ifdef ONNX
void onnxload_Sigmoid(const void *graph, struct module *m, int nodeidx)
{
	m->updateOutput = nn_Sigmoid_updateOutput;
	m->type = MT_Sigmoid;
}
#endif

THFloatTensor *nn_Sigmoid_updateOutput(struct module *module, THFloatTensor *input)
{
	THFloatTensor *output = module->output;
	float *input_data, *output_data;
	long i, n = THFloatTensor_nElement(input);

	THFloatTensor_resizeAs(output, input);

	input_data = THFloatTensor_data(input);
	output_data = THFloatTensor_data(output);

#pragma omp parallel for
	for(i = 0; i < n; i++)
		output_data[i] = 1.0f / (1.0f + expf(-input_data[i]));

	return output;
}
