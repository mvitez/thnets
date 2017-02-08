#include "../thnets.h"

static void nnfree_PReLU(struct module *mod)
{
	THFloatTensor_free(mod->PReLU.weight);
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

THFloatTensor *nn_PReLU_updateOutput(struct module *module, THFloatTensor *input)
{
	THFloatTensor *output = module->output;
	THFloatTensor *weight = module->PReLU.weight;

	THFloatTensor_resizeAs(output, input);
	float *in = THFloatTensor_data(input);
	float *out = THFloatTensor_data(output);
	float *w = THFloatTensor_data(weight);
	int bs, ks, nOutputPlane = module->PReLU.nOutputPlane;
	if(nOutputPlane == 0)
	{
		int i, n = THFloatTensor_nElement(input);
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
