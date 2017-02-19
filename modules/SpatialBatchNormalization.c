#include <math.h>
#include "../thnets.h"

static void nnfree_SpatialBatchNormalization(struct module *mod)
{
	THFloatTensor_free(mod->SpatialBatchNormalization.running_mean);
	THFloatTensor_free(mod->SpatialBatchNormalization.running_var);
	THFloatTensor_free(mod->SpatialBatchNormalization.weight);
	THFloatTensor_free(mod->SpatialBatchNormalization.bias);
}

int nnload_SpatialBatchNormalization(struct module *mod, struct nnmodule *n)
{
	struct table *t = n->table;
	mod->type = MT_SpatialBatchNormalization;
	mod->updateOutput = nn_SpatialBatchNormalization_updateOutput;
	mod->nnfree = nnfree_SpatialBatchNormalization;
	struct SpatialBatchNormalization *m = &mod->SpatialBatchNormalization;
	m->running_mean = TableGetTensor(t, "running_mean");
	m->running_var = TableGetTensor(t, "running_var");
	m->weight = TableGetTensor(t, "weight");
	m->bias = TableGetTensor(t, "bias");
	m->eps = TableGetNumber(t, "eps");
	return 0;
}

void pyload_SpatialBatchNormalization(struct pyfunction *f)
{
	f->module.updateOutput = nn_SpatialBatchNormalization_updateOutput;
	f->module.type = MT_SpatialBatchNormalization;
	f->module.nnfree = nnfree_SpatialBatchNormalization;
	struct SpatialBatchNormalization *p = &f->module.SpatialBatchNormalization;
	p->weight = pygettensor(f->params, "", 0);
	p->bias = pygettensor(f->params, "", 1);
	p->running_mean = pygettensor(f->params, "running_mean", 0);
	p->running_var = pygettensor(f->params, "running_var", 0);
	struct pyelement *el;
	if( (el = findelement(f->params, "eps", 0)) && el->type == ELTYPE_FLOAT)
		p->eps = el->fvalue;
}

THFloatTensor *nn_SpatialBatchNormalization_updateOutput(struct module *module, THFloatTensor *input)
{
	THFloatTensor *output = module->output;
	THFloatTensor *running_mean = module->SpatialBatchNormalization.running_mean;
	THFloatTensor *running_var = module->SpatialBatchNormalization.running_var;
	THFloatTensor *weight = module->SpatialBatchNormalization.weight;
	THFloatTensor *bias = module->SpatialBatchNormalization.bias;
	long nFeature = input->nDimension == 4 ? input->size[1] : input->size[0];

	double eps = module->SpatialBatchNormalization.eps;
	THFloatTensor_resizeAs(output, input);
	long f;

#ifndef MEMORYDEBUG
#pragma omp parallel for
#endif
	for (f = 0; f < nFeature; ++f)
	{
		THFloatTensor *in = THFloatTensor_newSelect(input, input->nDimension == 4 ? 1 : 0, f);
		THFloatTensor *out = THFloatTensor_newSelect(output, input->nDimension == 4 ? 1 : 0, f);

		float mean, invstd;

		mean = running_mean->storage->data[running_mean->storageOffset + running_mean->stride[0] * f];
		invstd = 1 / sqrt(running_var->storage->data[running_var->storageOffset + running_var->stride[0] * f] + eps);

		// compute output
		float w = weight ? weight->storage->data[weight->storageOffset + weight->stride[0] * f] : 1;
		float b = bias ? bias->storage->data[bias->storageOffset + bias->stride[0] * f] : 0;

		float *ind = in->storage->data + in->storageOffset;
		float *outd = out->storage->data + out->storageOffset;
		
		if(in->nDimension == 1)
		{
			long i;
			for(i = 0; i < in->size[0]; i++)
				outd[out->stride[0] * i] = ((ind[in->stride[0] * i] - mean) * invstd) * w + b;
		} else if(in->nDimension == 2)
		{
			long i, j;
			for(i = 0; i < in->size[0]; i++)
				for(j = 0; j < in->size[1]; j++)
					outd[out->stride[0] * i + out->stride[1] * j] =
						((ind[in->stride[0] * i + in->stride[1] * j] - mean) * invstd) * w + b;
		} else if(in->nDimension == 3)
		{
			long i, j, k;
			for(i = 0; i < in->size[0]; i++)
				for(j = 0; j < in->size[1]; j++)
					for(k = 0; k < in->size[2]; k++)
						outd[out->stride[0] * i + out->stride[1] * j + out->stride[2] * k] =
							((ind[in->stride[0] * i + in->stride[1] * j + in->stride[2] * k] - mean) * invstd) * w + b;
		} else THError("SpatialBatchNormalization not supported for input dimensions higher of 4");
			
		THFloatTensor_free(out);
		THFloatTensor_free(in);
	}
	return output;
}
