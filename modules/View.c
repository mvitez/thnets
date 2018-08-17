#include "../thnets.h"

int nnload_View(struct module *mod, struct nnmodule *n)
{
	struct table *t = n->table;
	mod->type = MT_View;
	mod->updateOutput = nn_View_updateOutput;
	struct View *m = &mod->View;
	m->numElements = TableGetNumber(t, "numElements");
	return 0;
}

void pyload_View(struct pyfunction *f)
{
	f->module.updateOutput = nn_View_updateOutput;
	f->module.type = MT_View;
	struct View *p = &f->module.View;
	struct pyelement *el;
	if( (el = findelement(f->params, "sizes", 0)) && el->type == ELTYPE_INTVECT)
		p->numElements = el->ivect[1];
}

#ifdef ONNX
void onnxload_View(const void *graph, struct module *m, int nodeidx)
{
	m->updateOutput = nn_View_updateOutput;
	m->type = MT_View;
	struct View *p = &m->View;
	p->numElements = onnx_getint(graph, nodeidx, "shape", 1);
	if(!p->numElements)
	{
		THFloatTensor *t = onnx_getshapetensor(graph, nodeidx, 1);
		if(t)
		{
			p->numElements = t->size[1];
			if(!p->numElements)
				p->numElements = -1;
			THFloatTensor_free(t);
		} else p->numElements = -1;
	}
}
#endif

THFloatTensor *nn_View_updateOutput(struct module *module, THFloatTensor *input)
{
	long nElements = THFloatTensor_nElement(input);
	long numElements = module->View.numElements;
	long batchSize = input->nDimension == 4 ? input->size[0] : 1;
	if(numElements == -1)
		numElements = nElements / batchSize;
	else batchSize = nElements / numElements;
	THFloatTensor *view;

	if (batchSize > 1)
		view = THFloatTensor_newWithStorage2d(input->storage, input->storageOffset, batchSize, numElements, numElements, 1);
	else
		view = THFloatTensor_newWithStorage1d(input->storage, input->storageOffset, numElements, 1);

	THFloatTensor_free(module->output);
	module->output = view;

	return module->output;
}
