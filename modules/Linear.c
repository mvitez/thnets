#include "../thnets.h"

static void nnfree_Linear(struct module *mod)
{
	THFloatTensor_free(mod->Linear.bias);
	THFloatTensor_free(mod->Linear.weight);
	THFloatTensor_free(mod->Linear.addBuffer);
}

int nnload_Linear(struct module *mod, struct nnmodule *n)
{
	struct table *t = n->table;
	mod->type = MT_Linear;
	mod->updateOutput = nn_Linear_updateOutput;
	mod->nnfree = nnfree_Linear;
	struct Linear *m = &mod->Linear;
	m->weight = TableGetTensor(t, "weight");
	m->addBuffer = TableGetTensor(t, "addBuffer");
	m->bias = TableGetTensor(t, "bias");
	return 0;
}

void pyload_Linear(struct pyfunction *f)
{
	f->module.updateOutput = nn_Linear_updateOutput;
	f->module.type = MT_Linear;
	f->module.nnfree = nnfree_Linear;
	struct Linear *p = &f->module.Linear;
	p->weight = pygettensor(f->params, "", 0);
	p->bias = pygettensor(f->params, "", 1);
	p->addBuffer = THFloatTensor_new();
}

#ifdef ONNX
void onnxload_Linear(const void *graph, struct module *m, int nodeidx)
{
	m->updateOutput = nn_Linear_updateOutput;
	m->type = MT_Linear;
	m->nnfree = nnfree_Linear;
	struct Linear *p = &m->Linear;
	p->weight = onnx_gettensor(graph, nodeidx, 1);
	p->bias = onnx_gettensor(graph, nodeidx, 2);
	p->addBuffer = THFloatTensor_new();
}
#endif

THFloatTensor *nn_Linear_updateOutput(struct module *module, THFloatTensor *input)
{
	THFloatTensor *weight = module->Linear.weight;
	THFloatTensor *bias = module->Linear.bias;
	THFloatTensor *output = module->output;
	THFloatTensor *addBuffer = module->Linear.addBuffer;

	THFloatTensor *in = THFloatTensor_squeeze(input);
	if (in->nDimension == 1) {
		THFloatTensor_resize1d(output, bias->size[0]);
		THFloatTensor_copy(output, bias);
		THFloatTensor_addmv(output, 1, output, 1, weight, in);

	} else if (in->nDimension == 2) {
		long nframe = in->size[0];
		long nElement = THFloatTensor_nElement(in);
		THFloatTensor_resize2d(output, nframe, bias->size[0]);
		if (THFloatTensor_nElement(output) != nElement)
			THFloatTensor_zero(output);

		if (THFloatTensor_nElement(addBuffer) != nframe) {
			THFloatTensor_resize1d(addBuffer, nframe);
			THFloatTensor_fill(addBuffer, 1.0);
		}
		THFloatTensor *t2 = THFloatTensor_newTranspose(weight, 0, 1);
		THFloatTensor_addmm(output, 0, output, 1, in, t2);
		THFloatTensor_free(t2);
		THFloatTensor_addr(output, 1, output, 1, addBuffer, bias);

	} else
		THError("input must be vector or matrix");
	THFloatTensor_free(in);
	return output;
}
