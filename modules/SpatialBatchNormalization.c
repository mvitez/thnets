#include <math.h>
#include "../thnets.h"

static void nnfree_SpatialBatchNormalization(struct module *mod)
{
	THNTensor_free(mod->SpatialBatchNormalization.running_mean);
	THNTensor_free(mod->SpatialBatchNormalization.running_var);
	THNTensor_free(mod->SpatialBatchNormalization.weight);
	THNTensor_free(mod->SpatialBatchNormalization.bias);
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

#ifdef ONNX
void onnxload_SpatialBatchNormalization(const void *graph, struct module *m, int nodeidx)
{
	if(m->ninputs == 2)
		m->type = MT_CMulTable;
    else
		m->type = MT_SpatialBatchNormalization;
	m->updateOutput = nn_SpatialBatchNormalization_updateOutput;
	m->nnfree = nnfree_SpatialBatchNormalization;
	struct SpatialBatchNormalization *p = &m->SpatialBatchNormalization;
	p->weight = onnx_gettensor(graph, nodeidx, 1);
	p->bias = onnx_gettensor(graph, nodeidx, 2);
	p->running_mean = onnx_gettensor(graph, nodeidx, 3);
	p->running_var = onnx_gettensor(graph, nodeidx, 4);
	p->eps = onnx_getfloat(graph, nodeidx, "epsilon", -1);
}
#endif

THNTensor *nn_SpatialBatchNormalization_updateOutput(struct module *module, THNTensor *input)
{
	THNTensor *output = module->output;
	THNTensor *running_mean = module->SpatialBatchNormalization.running_mean;
	THNTensor *running_var = module->SpatialBatchNormalization.running_var;
	THNTensor *weight = module->SpatialBatchNormalization.weight;
	THNTensor *bias = module->SpatialBatchNormalization.bias;
	long nFeature = input->nDimension == 4 ? input->size[1] : input->size[0];

	double eps = module->SpatialBatchNormalization.eps;
	THNTensor_resizeAs(output, input);
	long f;

#ifndef MEMORYDEBUG
#pragma omp parallel for
#endif
	for (f = 0; f < nFeature; ++f)
	{
		THNTensor *in = THNTensor_newSelect(input, input->nDimension == 4 ? 1 : 0, f);
		THNTensor *out = THNTensor_newSelect(output, input->nDimension == 4 ? 1 : 0, f);

		float mean, invstd;

		mean = running_mean->storage->data[running_mean->storageOffset + running_mean->stride[0] * f];
		invstd = 1 / sqrt(running_var->storage->data[running_var->storageOffset + running_var->stride[0] * f] + eps);

		// compute output
		float w = weight && weight->storage ? weight->storage->data[weight->storageOffset + weight->stride[0] * f] : 1;
		float b = bias && bias->storage ? bias->storage->data[bias->storageOffset + bias->stride[0] * f] : 0;

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
		} else THError("SpatialBatchNormalization not supported for input dimensions higher of 4 (%d)\n", in->nDimension);
		THNTensor_free(out);
		THNTensor_free(in);
	}
	return output;
}
