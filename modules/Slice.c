#include "../thnets.h"

void pyload_Slice(struct pyfunction *f)
{
	f->module.updateOutput = nn_Slice_updateOutput;
	f->module.type = MT_Slice;
	struct pyelement *el;
	if( (el = findelement(f->params, "index", 0)) && el->type == ELTYPE_INTVECT)
	{
		struct Slice *p = &f->module.Slice;
		p->from = el->ivect[0];
		p->to = el->ivect[1];
	}
}

THFloatTensor *nn_Slice_updateOutput(struct module *module, THFloatTensor *input)
{
	struct Slice *p = &module->Slice;
	THFloatTensor_slice(module->output, input, 0, p->from, p->to);
	return module->output;
}
