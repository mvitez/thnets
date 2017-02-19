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

void pyload_Dropout(struct pyfunction *f)
{
	f->module.updateOutput = nn_Dropout_updateOutput;
	f->module.type = MT_Dropout;
	struct Dropout *p = &f->module.Dropout;
	p->inplace = 1;
	p->v2 = 1;
	p->p = 0;
}

THFloatTensor *nn_Dropout_updateOutput(struct module *module, THFloatTensor *input)
{
	float p = module->Dropout.p;
	if(module->Dropout.inplace == 1)
		THFloatTensor_set(module->output, input);
	else {
		THFloatTensor_resizeAs(module->output, input);
		THFloatTensor_copy(module->output, input);
	}
	if(module->Dropout.v2 != 1)
	{
		long i, n = THFloatTensor_nElement(input);
		for(i = 0; i < n; i++)
			module->output->storage->data[i] = module->output->storage->data[i] * (1 - p);
	}
	return module->output;
}
