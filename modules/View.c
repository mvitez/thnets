#include "../thnets.h"
#include <string.h>

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
	int i;

	p->nDimension = 0;
	p->numElements = -1;
	THFloatTensor *t = onnx_getshapetensor(graph, nodeidx, 1);
	if(t)
	{
		p->nDimension = t->nDimension;
		memcpy(p->size, t->size, sizeof(p->size));
		THFloatTensor_free(t);
	} else {
		p->nDimension = onnx_getint(graph, nodeidx, "shape", -2);
		for(i = 0; i < p->nDimension; i++)
			p->size[i] = onnx_getint(graph, nodeidx, "shape", i);
	}
	if(p->nDimension)
	{
		// Remove the first "batch" dimension, as we don't use it
		p->nDimension--;
		memmove(p->size, p->size+1, p->nDimension * sizeof(p->size[0]));
	}
}
#endif

THFloatTensor *nn_View_updateOutput(struct module *module, THFloatTensor *input)
{
	long nElements = THFloatTensor_nElement(input);
	struct View *p = &module->View;
	int i, j;

	if(p->nDimension)
	{
		long nelements = 1, size[4];
		memcpy(size, p->size, sizeof(size));
		for(i = 0; i < p->nDimension; i++)
			if(size[i] == 0)
			{
				if(i >= input->nDimension)
					THError("Reshape has size 0 for non-existing dimension %d\n", i);
				nelements *= input->size[i];
			} else if(size[i] > 0)
				nelements *= size[i];
			else {
				size[i] = 1;
				for(j = i; j < input->nDimension; j++)
					size[i] *= input->size[j];
				nelements *= size[i];
			}
		if(nelements != THFloatTensor_nElement(input))
			THError("Wrong number of elements in Reshape: %ld (input) vs %ld (reshaped)\n", THFloatTensor_nElement(input), nelements);
		THFloatTensor_resize(module->output, size, p->nDimension);
	} else {
		long numElements = p->numElements;
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
	}
	return module->output;
}
