#include <math.h>
#include <string.h>
#include "../thnets.h"

int nnload_JoinTable(struct module *mod, struct nnmodule *n)
{
	struct table *t = n->table;
	mod->type = MT_JoinTable;
	mod->JoinTable.dimension = TableGetNumber(t, "dimension") - 1;
	mod->updateOutput = nn_JoinTable_updateOutput;
	return 0;
}

void pyload_Concat(struct pyfunction *f)
{
	f->module.updateOutput = nn_JoinTable_updateOutput;
	f->module.type = MT_JoinTable;
	struct Concat *p = &f->module.Concat;
	struct pyelement *el;
	if( (el = findelement(f->params, "dim", 0)) && el->type == ELTYPE_INT)
		p->dimension = el->ivalue;
}

THFloatTensor *nn_JoinTable_updateOutput(struct module *module, THFloatTensor *input)
{
	THFloatTensor *output = module->output;
	struct module *concattable_module = (struct module *)input;
	int nelem = concattable_module->ConcatTable.net->nelem;
	long size[4];
	int dimension = module->JoinTable.dimension;
	int i, j, sizen = 0;
	struct module *modules = concattable_module->ConcatTable.net->modules;
	if(dimension == 1 && (modules[0].output->nDimension == 1 || modules[0].output->nDimension == 3))
		dimension--;
	for(i = 0; i < nelem; i++)
		sizen += modules[i].output->size[dimension];
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
