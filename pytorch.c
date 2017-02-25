#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "thnets.h"

static int read8(FILE *fp)
{
	unsigned char v;

	if(fread(&v, 1, 1, fp) != 1)
		return -1;
	return v;
}

static int readint(FILE *fp, int *v)
{
	if(fread(v, 4, 1, fp) != 1)
		return -1;
	return 0;
}

static int readfloat(FILE *fp, float *v)
{
	if(fread(v, 4, 1, fp) != 1)
		return -1;
	return 0;
}

static int readstring(FILE *fp, char *s, int size)
{
	do {
		if(fread(s, 1, 1, fp) != 1)
			return -1;
		s++;
		size--;
		if(*s && !size)
			return -1;
	} while(s[-1]);
	return 0;
}

static int readintvect(FILE *fp, int *ndim, int *v)
{
	int i, n;

	if(readint(fp, &n))
		return -1;
	if(n <= 0 || n > 4)
		return -1;
	memset(v, 0, 4*sizeof(*v));
	for(i = 0; i < n; i++)
		if(readint(fp, v+i))
			return -1;
	if(ndim)
		*ndim = n;
	return 0;
}

THFloatTensor *readtensor(FILE *fp)
{
	int ndim, size[4], i;
	long long storagesize, stride=1;

	if(readintvect(fp, &ndim, size))
		return 0;
	if(fread(&storagesize, 8, 1, fp) != 1)
		return 0;
	
	THFloatTensor *th = THFloatTensor_new();
	th->nDimension = ndim;
	for(i = 0; i < ndim; i++)
		th->size[i] = size[i];
	for(i = ndim - 1; i >= 0; i--)
	{
		th->stride[i] = (long)stride;
		stride *= th->size[i];
	}
	th->storage = THFloatStorage_new((long)storagesize);
	if(fread(th->storage->data, sizeof(*th->storage->data), (size_t)storagesize, fp) != storagesize)
	{
		THFloatTensor_free(th);
		return 0;
	}
	return th;
}

struct pyelement *findelement(struct elemlist *list, const char *name, int skip)
{
	while(list)
	{
		if(!strcmp(list->elem->name, name))
		{
			if(!skip)
				return list->elem;
			skip--;
		}
		list = list->next;
	}
	return 0;
}

THFloatTensor *pygettensor(struct elemlist *list, const char *name, int skip)
{
	THFloatTensor *t = THFloatTensor_new();
	struct pyelement *el = findelement(list, name, skip);
	if(el && el->type == ELTYPE_TENSOR)
		THFloatTensor_set(t, el->tensor);
	return t;
}

struct {
	const char *name;
	void (*pyload)(struct pyfunction *f);
} name2loadf[] =
{
	{"ConvNd", pyload_SpatialConvolution},
	{"Linear", pyload_Linear},
	{"BatchNorm", pyload_SpatialBatchNormalization},
	{"MaxPool2d", pyload_SpatialMaxPooling},
	{"AvgPool2d", pyload_SpatialAveragePooling},
	{"Threshold", pyload_Threshold},
	{"Dropout", pyload_Dropout},
	{"Softmax", pyload_SoftMax},
	{"View", pyload_View},
	{"Add", pyload_Add},
	{"Concat", pyload_Concat}
};

static void buildmodule(const char *fname, struct pyfunction *f)
{
	int i;
	char *name = strrchr(fname, '.');
	if(!name)
		THError("Unsupported layer type %s\n", fname);
	name++;
	for(i = 0; i < sizeof(name2loadf)/sizeof(*name2loadf); i++)
		if(!strcmp(name, name2loadf[i].name))
		{
			f->module.output = THFloatTensor_new();
			name2loadf[i].pyload(f);
			return;
		}
	THError("Unsupported function %s\n", fname);
}

static struct pyelement *load_elem(FILE *fp, struct pyelement **nodes)
{
	char name[100];
	struct pyelement *node, *elem;

	node = calloc(1, sizeof(*node));
	node->type = read8(fp);
	switch(node->type)
	{
	case ELTYPE_END:
	case ELTYPE_INPUT:
		break;
	case ELTYPE_FUNCTION:
		if(readint(fp, &node->function.id))
			return 0;
		if((unsigned)node->function.id >= MAXPYNODES)
			return 0;
		nodes[node->function.id] = node;
		if(readstring(fp, name, sizeof(name)))
			return 0;
		node->name = strdup(name);
		for(;;)
		{
			elem = load_elem(fp, nodes);
			if(!elem)
				return 0;
			if(elem->type == ELTYPE_INT || elem->type == ELTYPE_FLOAT ||
				elem->type == ELTYPE_INTVECT || elem->type == ELTYPE_TENSOR)
			{
				// Append parameter to the linked list
				struct elemlist *entry = calloc(1, sizeof(*entry));
				entry->elem = elem;
				if(node->function.params)
				{
					struct elemlist *cur = node->function.params;
					while(cur->next)
						cur = cur->next;
					cur->next = entry;
				} else node->function.params = entry;
			} else if(elem->type == ELTYPE_FUNCTION || elem->type == ELTYPE_INPUT || elem->type == ELTYPE_OUTPUTID)
			{
				// Append input to the linked list
				struct elemlist *entry = calloc(1, sizeof(*entry));
				entry->elem = elem;
				if(node->function.inputs)
				{
					struct elemlist *cur = node->function.inputs;
					while(cur->next)
						cur = cur->next;
					cur->next = entry;
					node->function.ninputs++;
				} else {
					node->function.inputs = entry;
					node->function.ninputs = 1;
				}
			} else {
				free(elem);
				break;
			}
		}
		buildmodule(name, &node->function);
		break;
	case ELTYPE_INT:
		if(readstring(fp, name, sizeof(name)))
		{
			free(node);
			return 0;
		}
		node->name = strdup(name);
		if(readint(fp, &node->ivalue))
		{
			free(node);
			return 0;
		}
		break;
	case ELTYPE_FLOAT:
		if(readstring(fp, name, sizeof(name)))
		{
			free(node);
			return 0;
		}
		node->name = strdup(name);
		if(readfloat(fp, &node->fvalue))
		{
			free(node);
			return 0;
		}
		break;
	case ELTYPE_INTVECT:
		if(readstring(fp, name, sizeof(name)))
		{
			free(node);
			return 0;
		}
		node->name = strdup(name);
		if(readintvect(fp, 0, node->ivect))
		{
			free(node);
			return 0;
		}
		break;
	case ELTYPE_TENSOR:
		if(readstring(fp, name, sizeof(name)))
		{
			free(node);
			return 0;
		}
		node->name = strdup(name);
		node->tensor = readtensor(fp);
		if(!node->tensor)
		{
			free(node);
			return 0;
		}
		break;
	case ELTYPE_OUTPUTID:
		if(readint(fp, &node->ivalue))
		{
			free(node);
			return 0;
		}	
		break;
	default:
		free(node);
		return 0;
	}
	return node;
}

struct pyelement *loadpytorch(const char *path, struct pyelement **allpynodes)
{
	char header[24];
	FILE *fp = fopen(path, "rb");
	if(!fp)
		return 0;
	if(fread(header, 1, 24, fp) != 24)
	{
		fclose(fp);
		return 0;
	}
	if(strcmp(header, "PyTorch Graph Dump 1.00"))
	{
		fclose(fp);
		return 0;
	}
	struct pyelement *node = load_elem(fp, allpynodes);
	fclose(fp);
	return node;
}

THFloatTensor *forward_pytorch(struct pyelement *node, THFloatTensor *in, struct pyelement **allpynodes)
{
	if(node->type == ELTYPE_FUNCTION)
	{
		if(node->function.module.type == MT_CAddTable || node->function.module.type == MT_JoinTable)
		{
			// Instead of writing a new function, we just create the right parameters for it
			// It expects a module, but only takes ConcatTable.net.nelem and ConcatTable.net.modules
			int i = 0;
			struct network net;
			struct module m;
			struct module modules[node->function.ninputs];
			m.ConcatTable.net = &net;
			net.nelem = node->function.ninputs;
			net.modules = modules;
			struct elemlist *inputs = node->function.inputs;
			do {
				modules[i++].output = forward_pytorch(inputs->elem, in, allpynodes);
				inputs = inputs->next;
			} while(inputs);
			return node->function.module.updateOutput(&node->function.module, (THFloatTensor *)&m);
		}
		return node->function.module.updateOutput(&node->function.module, forward_pytorch(node->function.inputs->elem, in, allpynodes));
	} else if(node->type == ELTYPE_OUTPUTID)
		return allpynodes[node->ivalue]->function.module.output;
	else return in;
}

void freepynet(struct pyelement *node)
{
	if(node->name)
		free(node->name);
	if(node->type == ELTYPE_FUNCTION)
	{
		struct elemlist *e2, *el = node->function.params;
		while(el)
		{
			freepynet(el->elem);
			e2 = el;
			el = el->next;
			free(e2);
		}
		el = node->function.inputs;
		while(el)
		{
			freepynet(el->elem);
			e2 = el;
			el = el->next;
			free(e2);
		}
		THFloatTensor_free(node->function.module.output);
		if(node->function.module.nnfree)
			node->function.module.nnfree(&node->function.module);
	} else if(node->type == ELTYPE_TENSOR)
		THFloatTensor_free(node->tensor);
	free(node);
}
