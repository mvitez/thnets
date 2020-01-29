#include "../thnets.h"

static void nnfree_Linear(struct module *mod)
{
	THNTensor_free(mod->Linear.bias);
	THNTensor_free(mod->Linear.weight);
	THNTensor_free(mod->Linear.addBuffer);
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

#ifdef ONNX
void onnxload_Linear(const void *graph, struct module *m, int nodeidx)
{
	m->updateOutput = nn_Linear_updateOutput;
	m->type = MT_Linear;
	m->nnfree = nnfree_Linear;
	struct Linear *p = &m->Linear;
	int widx = 1;
	if(!onnx_isinitializer(graph, nodeidx, 1))
	{
		widx = 0;
		p->commute = 1;
		if(!onnx_isinitializer(graph, nodeidx, 0))
			THError("MatMul between two inputs is not supported\n");
	}
	THNTensor *weight = onnx_gettensor(graph, nodeidx, widx);
	if(weight->nDimension > 2)
	{
		p->weight = THNTensor_squeeze(weight);	// To deal with some networks that have a reshape after weight
		THNTensor_free(weight);
	} else p->weight = weight;
	p->bias = onnx_gettensor(graph, nodeidx, 2);
	if (onnx_getfloat(graph, nodeidx, "beta", -1) == 0.0 && onnx_getfloat(graph, nodeidx, "beta", -3) == 1)
	{
		THNTensor_free(p->bias);
		p->bias = THNTensor_new();
	}
	if(!onnx_getint(graph, nodeidx, "transB", -1) && !p->commute)
	{
		THNTensor *weight = THNTensor_newTranspose(p->weight, 0, 1);
		THNTensor *cweight = THNTensor_new();
		THNTensor_resize2d(cweight, weight->size[0], weight->size[1]);
		THNTensor_safecopy(cweight, weight);
		THNTensor_free(weight);
		THNTensor_free(p->weight);
		p->weight = cweight;
	}
	p->addBuffer = THNTensor_new();
}
#endif

THNTensor *nn_Linear_updateOutput(struct module *module, THNTensor *input)
{
	THNTensor *weight = module->Linear.weight;
	THNTensor *bias = module->Linear.bias;
	THNTensor *output = module->output;
	THNTensor *addBuffer = module->Linear.addBuffer;
	int commute = module->Linear.commute;

	if(commute)
	{
		THNTensor *tmp = input;
		input = weight;
		weight = tmp;
	}
	THNTensor *in = THNTensor_squeeze(input);
	if (in->nDimension == 1) {
		THNTensor_resize1d(output, bias->size[0]);
		THNTensor_copy(output, bias);
		THNTensor_addmv(output, 1, output, 1, weight, in);

	} else if (in->nDimension == 2) {
		long nframe = in->size[0];
		THNTensor *t2 = commute ? THNTensor_newWithTensor(weight) : THNTensor_newTranspose(weight, 0, 1);
		long nElement = THNTensor_nElement(in);
		THNTensor_resize2d(output, nframe, t2->size[1]);
		if (THNTensor_nElement(output) != nElement)
			THNTensor_zero(output);

		if (bias->storage && THNTensor_nElement(addBuffer) != nframe) {
			THNTensor_resize1d(addBuffer, nframe);
			THNTensor_fill(addBuffer, 1.0);
		}
		THNTensor_addmm(output, 0, output, 1, in, t2);
		THNTensor_free(t2);
		if(bias->storage)
			THNTensor_addr(output, 1, output, 1, addBuffer, bias);

	} else
		THError("input must be vector or matrix");
	THNTensor_free(in);
	return output;
}
