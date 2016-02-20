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

THFloatTensor *nn_Linear_updateOutput(struct module *module, THFloatTensor *input)
{
	THFloatTensor *weight = module->Linear.weight;
	THFloatTensor *bias = module->Linear.bias;
	THFloatTensor *output = module->output;
	THFloatTensor *addBuffer = module->Linear.addBuffer;

	if (input->nDimension == 1) {
		THFloatTensor_resize1d(output, bias->size[0]);
		THFloatTensor_copy(output, bias);
		THFloatTensor_addmv(output, 1, output, 1, weight, input);

	} else if (input->nDimension == 2) {
		long nframe = input->size[0];
		long nElement = THFloatTensor_nElement(input);
		THFloatTensor_resize2d(output, nframe, bias->size[0]);
		if (THFloatTensor_nElement(output) != nElement)
			THFloatTensor_zero(output);

		if (THFloatTensor_nElement(addBuffer) != nframe) {
			THFloatTensor_resize1d(addBuffer, nframe);
			THFloatTensor_fill(addBuffer, 1.0);
		}
		THFloatTensor *t2 = THFloatTensor_newTranspose(weight, 0, 1);
		THFloatTensor_addmm(output, 0, output, 1, input, t2);
		THFloatTensor_free(t2);
		THFloatTensor_addr(output, 1, output, 1, addBuffer, bias);

	} else
		THError("input must be vector or matrix");

	return output;
}
