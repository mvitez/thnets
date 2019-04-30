#include <math.h>
#include <string.h>
#include <stdio.h>
#include "../thnets.h"

int nnload_CAddTable(struct module *mod, struct nnmodule *n)
{
	mod->type = MT_CAddTable;
	mod->updateOutput = nn_CAddTable_updateOutput;
	return 0;
}

void pyload_Add(struct pyfunction *f)
{
	f->module.updateOutput = nn_CAddTable_updateOutput;
	f->module.type = MT_CAddTable;
}

#ifdef ONNX
void onnxload_Add(const void *graph, struct module *m, int nodeidx)
{
	m->updateOutput = nn_CAddTable_updateOutput;
	m->type = MT_CAddTable;
}
void onnxload_Sub(const void *graph, struct module *m, int nodeidx)
{
	m->updateOutput = nn_CAddTable_updateOutput;
	m->type = MT_CSubTable;
}
#endif

THFloatTensor *nn_CAddTable_updateOutput(struct module *module, THFloatTensor *input)
{
	THFloatTensor *output = module->output;
	struct module *concattable_module = (struct module *)input;
	int nelem = concattable_module->ConcatTable.net->nelem;
	long size[4];
	int i, j;
	struct module *modules = concattable_module->ConcatTable.net->modules;
	// Check correctness
	for(i = 1; i < nelem; i++)
	{
		if(modules[i].output->nDimension != modules[0].output->nDimension)
			THError("Sum of tensors of different dimensionality");
		for(j = 0; j < modules[0].output->nDimension; j++)
			if(modules[0].output->size[j] < modules[i].output->size[j])
				THError("Sum of tensors of different sizes");
	}
	memcpy(size, modules[0].output->size, sizeof(size));
	THFloatTensor_resize(output, size, modules[0].output->nDimension);
	float *out = THFloatTensor_data(output);
	long n[nelem];
	float *outs[nelem];
	for(i = 0; i < nelem; i++)
	{
		outs[i] = THFloatTensor_data(modules[i].output);
		n[i] = THFloatTensor_nElement(modules[i].output);
	}
	if(nelem == 2)
	{
		// Optimized case
        if(module->type == MT_CSubTable)
		    for(j = 0; j < n[1]; j++)
			    out[j] = outs[0][j] + outs[1][j];
        else
		    for(j = 0; j < n[1]; j++)
			    out[j] = outs[0][j] - outs[1][j];
		for(; j < n[0]; j++)
			out[j] = outs[0][j];
		
	} else {
		for(j = 0; j < n[0]; j++)
			out[j] = outs[0][j];
		if(module->type == MT_CSubTable)
		{
			for(i = 0; i < nelem; i++)
				for(j = 0; j < n[i]; j++)
					out[j] -= outs[i][j];
		} else {
			for(i = 0; i < nelem; i++)
				for(j = 0; j < n[i]; j++)
					out[j] += outs[i][j];
		}
	}
	return output;
}
