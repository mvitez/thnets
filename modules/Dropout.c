#include "../thnets.h"

int nnload_Dropout(struct module *mod, struct nnmodule *n)
{
	struct table *t = n->table;
	mod->type = MT_Dropout;
	mod->updateOutput = nn_Dropout_updateOutput;
	struct Dropout *m = &mod->Dropout;
	m->inplace = TableGetBoolean(t, "inplace");
	m->v2 = TableGetBoolean(t, "v2");
	m->p = TableGetNumber(t, "p");
	return 0;
}

#ifdef ONNX
void onnxload_Dropout(const void *graph, struct module *m, int nodeidx)
{
	m->updateOutput = nn_Dropout_updateOutput;
	m->type = MT_Dropout;
	struct Dropout *p = &m->Dropout;
	p->inplace = 1;
	p->v2 = 1;
	p->p = 0;
}
#endif

THNTensor *nn_Dropout_updateOutput(struct module *module, THNTensor *input)
{
	float p = module->Dropout.p;
	if(module->Dropout.inplace == 1)
		THNTensor_set(module->output, input);
	else {
		THNTensor_resizeAs(module->output, input);
		THNTensor_copy(module->output, input);
	}
	if(module->Dropout.v2 != 1)
	{
		long i, n = THNTensor_nElement(input);
		for(i = 0; i < n; i++)
			module->output->storage->data[i] = module->output->storage->data[i] * (1 - p);
	}
	return module->output;
}
