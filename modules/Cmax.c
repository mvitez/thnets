#include <string.h>
#include "../thnets.h"

#define MAX(a,b) ((a) > (b) ? (a) : (b))

#ifdef ONNX
void onnxload_Cmax(const void *graph, struct module *m, int nodeidx)
{
	m->updateOutput = nn_Cmax_updateOutput;
	m->type = MT_Cmax;
}
#endif


THNTensor *nn_Cmax_updateOutput(struct module *module, THNTensor *input)
{
	THNTensor *output = module->output;
	struct module *concattable_module = (struct module *)input;
	int nelem = concattable_module->ConcatTable.net->nelem;
	long size[4];
	int i, j, batch;
	struct module *modules = concattable_module->ConcatTable.net->modules;
	// Check correctness
	for(i = 1; i < nelem; i++)
	{
		if(modules[i].output->nDimension != modules[0].output->nDimension)
			THError("Cmax of tensors of different dimensionality");
		for(j = 0; j < modules[0].output->nDimension; j++)
			if(modules[0].output->size[j] < modules[i].output->size[j])
				THError("Cmax of tensors of different sizes");
	}
	memcpy(size, modules[0].output->size, sizeof(size));
	THNTensor_resize(output, size, modules[0].output->nDimension);
	float *out = THNTensor_data(output);
	int batchsize = modules[0].output->nDimension % 2 == 0 ? size[0] : 1;
	for(batch = 0; batch < batchsize; batch++)
	{
		long n[nelem];
		float *outs[nelem];
		for(i = 0; i < nelem; i++)
		{
			outs[i] = THNTensor_data(modules[i].output) + batch * modules[i].output->stride[0];
			n[i] = THNTensor_nElement(modules[i].output) / batchsize;
		}
		if(nelem == 2)
		{
			// Optimized case
			for(j = 0; j < n[1]; j++)
				out[j] = MAX(outs[0][j], outs[1][j]);
			for(; j < n[0]; j++)
				out[j] = outs[0][j];
		} else {
			for(j = 0; j < n[0]; j++)
				out[j] = outs[0][j];
			for(i = 0; i < nelem; i++)
				for(j = 0; j < n[i]; j++)
					out[j] = MAX(out[j], outs[i][j]);
		}
		out += output->stride[0];
	}
	return output;
}
