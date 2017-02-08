#include <math.h>
#include <string.h>
#include "../thnets.h"

static void nnfree_Concat(struct module *mod)
{
	freenetwork(mod->Concat.net);
}

int nnload_Concat(struct module *mod, struct nnmodule *n)
{
	struct table *t = n->table;
	mod->type = MT_Concat;
	mod->Concat.dimension = TableGetNumber(t, "dimension") - 1;
	mod->Concat.net = Module2Network(n);
	mod->nnfree = nnfree_Concat;
	mod->updateOutput = nn_Concat_updateOutput;
	return 0;
}

THFloatTensor *nn_Concat_updateOutput(struct module *module, THFloatTensor *input)
{
	THFloatTensor *output = module->output;
	int nelem = module->Concat.net->nelem;
	long size[4];
	int dimension = module->Concat.dimension;
	int i, j, sizen = 0;
	struct module *modules = module->Concat.net->modules;
	if(dimension == 1 && (input->nDimension == 1 || input->nDimension == 3))
		dimension--;
	for(i = 0; i < nelem; i++)
	{
		modules[i].updateOutput(&modules[i], input);
		sizen += modules[i].output->size[dimension];
	}
	// Check correctness
	for(i = 1; i < nelem; i++)
	{
		if(modules[i].output->nDimension != modules[0].output->nDimension)
			THError("Concatenation of tensors of different dimensionality");
		for(j = 0; j < modules[0].output->nDimension; j++)
			if(j != dimension && modules[0].output->size[j] != modules[i].output->size[j])
				THError("Concatenation of tensors of different sizes");
	}
	memcpy(size, modules[0].output->size, sizeof(size));
	size[dimension] = sizen;
	THFloatTensor_resize(output, size, modules[0].output->nDimension);
	long offset = 0;
	if(dimension == 0)
	{
		for(i = 0; i < nelem; i++)
		{
			memcpy(THFloatTensor_data(output) + output->stride[0] * offset, THFloatTensor_data(modules[i].output),
				THFloatTensor_nElement(modules[i].output) * sizeof(*output->storage->data));
			offset += modules[i].output->size[0];
		}
	} else if(dimension == 1)
	{
		long transfersize = sizeof(*output->storage->data);
		for(j = dimension + 1; j < output->nDimension; j++)
			transfersize *= output->size[j];
		for(j = 0; j < size[0]; j++)
		{
			offset = 0;
			for(i = 0; i < nelem; i++)
			{
				memcpy(THFloatTensor_data(output) + output->stride[0] * j + output->stride[1] * offset,
					THFloatTensor_data(modules[i].output) + modules[i].output->stride[0] * j,
					transfersize * modules[i].output->size[1]);
				offset += modules[i].output->size[1];
			}
		}
	}
	return output;

}
