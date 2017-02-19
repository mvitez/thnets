#include "../thnets.h"

int nnload_Threshold(struct module *mod, struct nnmodule *n)
{
	struct table *t = n->table;
	mod->type = MT_Threshold;
	mod->updateOutput = nn_Threshold_updateOutput;
	struct Threshold *m = &mod->Threshold;
	m->threshold = TableGetNumber(t, "threshold");
	m->val = TableGetNumber(t, "val");
	m->inplace = TableGetBoolean(t, "inplace");
	return 0;
}

void pyload_Threshold(struct pyfunction *f)
{
	f->module.updateOutput = nn_Threshold_updateOutput;
	f->module.type = MT_Threshold;
	struct Threshold *p = &f->module.Threshold;
	struct pyelement *el;
	if( (el = findelement(f->params, "threshold", 0)) && el->type == ELTYPE_FLOAT)
		p->threshold = el->fvalue;
	if( (el = findelement(f->params, "value", 0)) && el->type == ELTYPE_FLOAT)
		p->val = el->fvalue;
	if( (el = findelement(f->params, "inplace", 0)) && el->type == ELTYPE_INT)
		p->inplace = el->ivalue;
}

THFloatTensor *nn_Threshold_updateOutput(struct module *module, THFloatTensor *input)
{
	float val = module->Threshold.val;
	float threshold = module->Threshold.threshold;
	THFloatTensor *output = module->output;
	int inPlace = module->Threshold.inplace == 1;

	long i, n = THFloatTensor_nElement(input);
	if (inPlace)
	{
		for(i = 0; i < n; i++)
			if (input->storage->data[i] <= threshold)
				input->storage->data[i] = val;
		THFloatTensor_set(output, input);
	} else {
		THFloatTensor_resizeAs(output, input);
		for(i = 0; i < n; i++)
			output->storage->data[i] = (input->storage->data[i] > threshold) ? input->storage->data[i] : val;
	}
	return output;
}
