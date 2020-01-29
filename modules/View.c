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

#ifdef ONNX
void onnxload_View(const void *graph, struct module *m, int nodeidx)
{
	m->updateOutput = nn_View_updateOutput;
	m->type = MT_View;
	struct View *p = &m->View;
	int i;

	p->nDimension = 0;
	p->numElements = -1;
	THNTensor *t = onnx_getshapetensor(graph, nodeidx, 1);
	if(t)
	{
		p->nDimension = t->nDimension;
		memcpy(p->size, t->size, sizeof(p->size));
		THNTensor_free(t);
	} else {
		p->nDimension = onnx_getint(graph, nodeidx, "shape", -2);
		for(i = 0; i < p->nDimension; i++)
			p->size[i] = onnx_getint(graph, nodeidx, "shape", i);
	}
}
#endif

THNTensor *nn_View_updateOutput(struct module *module, THNTensor *input)
{
	long nElements = THNTensor_nElement(input);
	struct View *p = &module->View;
	int i, j;

	if(p->nDimension)
	{
		long nelements = 1, size[4];
		int ndim;
		if(input->nDimension == 4)
		{
			ndim = p->nDimension;
			memcpy(size, p->size, sizeof(size));
		} else {
			// p->size refers to a 4D tensor, we are working with 3D tensors
			ndim = p->nDimension-1;
			memcpy(size, p->size+1, sizeof(size[0]) * 3);
		}
		for(i = 0; i < ndim; i++)
			if(size[i] == 0)
			{
				if(i >= ndim)
					THError("Reshape has size 0 for non-existing dimension %d\n", i);
				nelements *= input->size[i];
			} else if(size[i] > 0)
				nelements *= size[i];
			else {
				size[i] = 1;
				for(j = i; j < ndim; j++)
					size[i] *= input->size[j];
				nelements *= size[i];
			}
		if(nelements != THNTensor_nElement(input))
			THError("Wrong number of elements in Reshape: %ld (input) vs %ld (reshaped)\n", THNTensor_nElement(input), nelements);
		THNTensor_set(module->output, input);
		THNTensor_resize(module->output, size, ndim);
	} else {
		long numElements = p->numElements;
		long batchSize = input->nDimension == 4 ? input->size[0] : 1;
		if(numElements == -1)
			numElements = nElements / batchSize;
		else batchSize = nElements / numElements;
		THNTensor *view;
		if (batchSize > 1)
			view = THNTensor_newWithStorage2d(input->storage, input->storageOffset, batchSize, numElements, numElements, 1);
		else
			view = THNTensor_newWithStorage1d(input->storage, input->storageOffset, numElements, 1);
		THNTensor_free(module->output);
		module->output = view;
	}
	return module->output;
}
