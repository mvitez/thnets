#include "../thnets.h"

int nnload_SoftMax(struct module *mod, struct nnmodule *n)
{
	mod->type = MT_SoftMax;
	mod->updateOutput = nn_SoftMax_updateOutput;
	return 0;
}

#ifdef ONNX
void onnxload_SoftMax(const void *graph, struct module *m, int nodeidx)
{
	m->updateOutput = nn_SoftMax_updateOutput;
	m->type = MT_SoftMax;
}
#endif

THNTensor *nn_SoftMax_updateOutput(struct module *module, THNTensor *input)
{
	THNTensor *output = module->output;
	float *input_data, *output_data;
	long nframe = 0, dim = 0, stride = 0;
	long t;

	if(input->nDimension == 1)
	{
		nframe = 1;
		dim = input->size[0];
		stride = 1;
	}
	else if(input->nDimension == 2)
	{
		nframe = input->size[0];
		dim = input->size[1];
		stride = 1;
	}
	else if(input->nDimension == 3)
	{
		nframe = 1;
		dim = input->size[0];
		stride = input->size[1]*input->size[2];
	}
	else if(input->nDimension == 4)
	{
		nframe = input->size[0];
		dim = input->size[1];
		stride = input->size[2]*input->size[3];
	}
	else
		THError("1D, 2D, 3D or 4D tensor expected");

	THNTensor_resizeAs(output, input);

	input_data = THNTensor_data(input);
	output_data = THNTensor_data(output);

#pragma omp parallel for private(t)
	for(t = 0; t < stride*nframe; t++)
	{
		float *input_ptr = input_data + (t/stride)*dim*stride + t % stride;
		float *output_ptr = output_data + (t/stride)*dim*stride + t % stride;

		float inputMax = -THInf;
		float sum;

		long d;
		for(d = 0; d < dim; d++) {
			if (input_ptr[d*stride] >= inputMax) inputMax = input_ptr[d*stride];
		}

		sum = 0;
		for(d = 0; d < dim; d++) {
			float z = THExpMinusApprox(inputMax - input_ptr[d*stride]);
			output_ptr[d*stride] = z;
			sum += z;
		}

		for(d = 0; d < dim; d++) {
			output_ptr[d*stride] *= 1/sum;
		}
	}

	return output;
}
