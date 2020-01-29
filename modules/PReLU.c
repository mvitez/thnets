#include "../thnets.h"

static void nnfree_PReLU(struct module *mod)
{
	THNTensor_free(mod->PReLU.weight);
}

int nnload_PReLU(struct module *mod, struct nnmodule *n)
{
	struct table *t = n->table;
	mod->type = MT_PReLU;
	mod->updateOutput = nn_PReLU_updateOutput;
	mod->nnfree = nnfree_PReLU;
	struct PReLU *m = &mod->PReLU;
	m->nOutputPlane = TableGetNumber(t, "nOutputPlane");
	m->weight = TableGetTensor(t, "weight");
	return 0;
}

#ifdef ONNX
void onnxload_PReLU(const void *graph, struct module *m, int nodeidx)
{
	m->type = MT_PReLU;
	m->updateOutput = nn_PReLU_updateOutput;
	m->nnfree = nnfree_PReLU;
	struct PReLU *p = &m->PReLU;
	p->weight = onnx_gettensor(graph, nodeidx, 1);
	p->nOutputPlane = (int)p->weight->size[0];
}
#endif

THNTensor *nn_PReLU_updateOutput(struct module *module, THNTensor *input)
{
	THNTensor *output = module->output;
	THNTensor *weight = module->PReLU.weight;

	THNTensor_resizeAs(output, input);
	float *in = THNTensor_data(input);
	float *out = THNTensor_data(output);
	float *w = THNTensor_data(weight);
	long bs, ks, nOutputPlane = module->PReLU.nOutputPlane;
	if(nOutputPlane == 0)
	{
		long i, n = THNTensor_nElement(input);
		for(i = 0; i < n; i++)
			out[i] = in[i] > 0 ? in[i] : *w*in[i];
		return output;
	}
	int input_ndim = input->nDimension;
	switch (input_ndim)
	{
	case 1:
		bs = 1;
		ks = 1;
		break;
	case 2:
		bs = input->size[0];
		ks = 1;
		break;
	case 3:
		bs = 1;
		ks = input->size[1] * input->size[2];
		break;
	case 4:
		bs = input->size[0];
		ks = input->size[2] * input->size[3];
		break;
	default:
		ks = 0;
		bs = 0;
		break;
	}

	if (input->size[(input_ndim + 1) % 2] != nOutputPlane)
		THError("wrong number of input planes");

	int i, j, k;
#pragma omp parallel for private(j,k)
	for(i = 0; i < bs; ++i)
	{
		float *n_input_data = in + i*nOutputPlane*ks;
		float *n_output_data = out + i*nOutputPlane*ks;
		for (j = 0; j < nOutputPlane; ++j)
		{
			for (k = 0; k < ks; ++k)
				n_output_data[k] = (n_input_data[k] > 0) ? n_input_data[k] : w[j] * n_input_data[k];
			n_input_data += ks;
			n_output_data += ks;
		}
	}
	return output;
}
