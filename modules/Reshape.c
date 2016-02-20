#include <string.h>
#include "../thnets.h"

int nnload_Reshape(struct module *mod, struct nnmodule *n)
{
	struct table *t = n->table;
	mod->type = MT_Reshape;
	mod->updateOutput = nn_Reshape_updateOutput;
	struct Reshape *m = &mod->Reshape;
	m->numElements = TableGetNumber(t, "nelement");
	m->batchMode = TableGetBoolean(t, "batchMode");
	void *data = TableGetStorage(t, "size", &m->nsize);
	if(data && m->nsize <= 4)
		memcpy(m->size, data, sizeof(*m->size) * m->nsize);
	data = TableGetStorage(t, "batchsize", &m->nbatchsize);
	if(data && m->nbatchsize <= 4)
		memcpy(m->batchsize, data, sizeof(*m->batchsize) * m->nbatchsize);
	return 0;
}

THFloatTensor *nn_Reshape_updateOutput(struct module *module, THFloatTensor *input)
{
	long numElements = module->Reshape.numElements;
	long nElements = THFloatTensor_nElement(input);
	THFloatTensor_set(module->output, input);
	if(module->Reshape.batchMode == 0 ||
		(module->Reshape.batchMode == -1 && nElements == numElements && input->size[0] != 1))
		THFloatTensor_resize(module->output, module->Reshape.size, module->Reshape.nsize);
	else {
		module->Reshape.batchsize[0] = input->size[0];
		THFloatTensor_resize(module->output, module->Reshape.batchsize, module->Reshape.nbatchsize);
	}
	return module->output;
}
