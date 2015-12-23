#include <stdlib.h>
#include <string.h>
#include "../thnets.h"

static cudnnHandle_t handle;
int floattype = CUDNN_DATA_FLOAT;

void checkerr(int status)
{
	if(status)
		THError("cudnn call failed");
}

cudnnHandle_t THcudnn_getHandle()
{
	if(!handle)
		cudnnCreate(&handle);
	return handle;
}

int THcudnn_TensorDescriptor(cudnnTensorDescriptor_t *d, THFloatTensor *t)
{
	int i, rc, base;
	int size[4], stride[4];

	rc = cudnnCreateTensorDescriptor(d);
	if(rc)
		return rc;
	if(t->nDimension == 3)
	{
		size[0] = 1;
		stride[0] = THFloatTensor_nElement(t);
		base = 1;
	} else base = 0;
	for(i = 0; i < t->nDimension; i++)
	{
		size[i+base] = t->size[i];
		stride[i+base] = t->stride[i];
	}
	return cudnnSetTensorNdDescriptor(*d, floattype, 4, size, stride);
}

THFloatStorage *THCudaStorage_new(long size)
{
	THFloatStorage *s = malloc(sizeof(*s));
	errcheck(cudaMalloc((void **)&s->data, size * sizeof(*s->data)));
	s->nref = 1;
	s->mustfree = 2;
	return s;
}

THFloatTensor *THCudaTensor_newFromFloatTensor(THFloatTensor *t)
{
	THFloatTensor *n = malloc(sizeof(*n));
	memcpy(n, t, sizeof(*n));
	if(t->storage)
	{
		n->storage = malloc(sizeof(*n->storage));
		n->storage->nref = 1;
		n->storage->mustfree = 2;
		errcheck(cudaMalloc((void **)&n->storage->data, THFloatTensor_nElement(t) * sizeof(*n->storage->data)));
		errcheck(cudaMemcpy(n->storage->data, THFloatTensor_data(t), THFloatTensor_nElement(t) * sizeof(*n->storage->data), cudaMemcpyHostToDevice));
	}
	return n;
}

THFloatTensor *THFloatTensor_newFromCudaTensor(THFloatTensor *t)
{
	THFloatTensor *n = malloc(sizeof(*n));
	memcpy(n, t, sizeof(*n));
	if(t->storage)
	{
		n->storage = malloc(sizeof(*n->storage));
		n->storage->nref = 1;
		n->storage->mustfree = 1;
		n->storage->data = malloc(THFloatTensor_nElement(t) * sizeof(*n->storage->data));
		errcheck(cudaMemcpy(n->storage->data, THFloatTensor_data(t), THFloatTensor_nElement(t) * sizeof(*n->storage->data), cudaMemcpyDeviceToHost));
	}
	return n;
}

void THCudaTensor_resize4d(THFloatTensor *t, long size0, long size1, long size2, long size3)
{
	t->nDimension = 4;
	t->size[0] = size0;
	t->size[1] = size1;
	t->size[2] = size2;
	t->size[3] = size3;
	t->stride[3] = 1;
	t->stride[2] = size3;
	t->stride[1] = size2 * size3;
	t->stride[0] = size1 * size2 * size3;
	if(!t->storage)
	{
		t->storage = malloc(sizeof(*t->storage));
		t->storage->nref = 1;
		t->storage->mustfree = 2;
		errcheck(cudaMalloc((void **)&t->storage->data, THFloatTensor_nElement(t) * sizeof(*t->storage->data)));
	}
}

struct network *THcudnn_ToCUDNN(struct network *net)
{
	int i;
	struct network *nn = malloc(sizeof(*nn));

	nn->nelem = net->nelem;
	nn->modules = malloc(sizeof(net->modules[0]) * net->nelem);
	nn->cuda = 1;
	memcpy(nn->modules, net->modules, sizeof(net->modules[0]) * net->nelem);
	for(i = 0; i < net->nelem; i++)
	{
		nn->modules[i].output = THCudaTensor_newFromFloatTensor(net->modules[i].output);
		switch(net->modules[i].type)
		{
		case MT_SpatialConvolutionMM:
		case MT_SpatialConvolution:
			nn->modules[i].updateOutput = cudnn_SpatialConvolution_updateOutput;
#ifdef HAVEFP16
			if(floattype == CUDNN_DATA_HALF)
			{
				nn->modules[i].SpatialConvolution.weight = THHalfCudaTensor_newFromFloatTensor(net->modules[i].SpatialConvolution.weight);
				nn->modules[i].SpatialConvolution.bias = THHalfCudaTensor_newFromFloatTensor(net->modules[i].SpatialConvolution.bias);
			} else
#endif
			{
				nn->modules[i].SpatialConvolution.weight = THCudaTensor_newFromFloatTensor(net->modules[i].SpatialConvolution.weight);
				nn->modules[i].SpatialConvolution.bias = THCudaTensor_newFromFloatTensor(net->modules[i].SpatialConvolution.bias);
			}
			nn->modules[i].SpatialConvolution.finput = 0;
			break;
		case MT_SpatialMaxPooling:
			nn->modules[i].updateOutput = cudnn_SpatialMaxPooling_updateOutput;
			break;
		case MT_Threshold:
			if(nn->modules[i].Threshold.threshold || nn->modules[i].Threshold.val)
				THError("Threshold not supported in CUDNN, only ReLU is supported");
			nn->modules[i].updateOutput = cudnn_Threshold_updateOutput;
			break;
		case MT_SoftMax:
			nn->modules[i].updateOutput = cudnn_SoftMax_updateOutput;
			break;
		case MT_Dropout:
			if(!nn->modules[i].Dropout.v2)
				THError("Non v2 dropout not supported in CUDNN");
			break;
		case MT_SpatialZeroPadding:
			THError("SpatialZeroPadding not supported in CUDNN");
			break;
		case MT_Linear:
			nn->modules[i].type = MT_SpatialConvolutionMM;
			nn->modules[i].updateOutput = cudnn_SpatialConvolution_updateOutput;
			struct SpatialConvolution *c = &nn->modules[i].SpatialConvolution;
			c->finput = 0;
			c->padW = c->padH = 0;
			c->dW = c->dH = 1;
			c->kW = c->kH = 1;
			c->nOutputPlane = c->weight->size[0];
			c->nInputPlane = c->weight->size[1];
#ifdef HAVEFP16
			if(floattype == CUDNN_DATA_HALF)
			{
				nn->modules[i].SpatialConvolution.weight = THHalfCudaTensor_newFromFloatTensor(net->modules[i].SpatialConvolution.weight);
				nn->modules[i].SpatialConvolution.bias = THHalfCudaTensor_newFromFloatTensor(net->modules[i].SpatialConvolution.bias);
			} else
#endif
			{
				nn->modules[i].SpatialConvolution.weight = THCudaTensor_newFromFloatTensor(net->modules[i].SpatialConvolution.weight);
				nn->modules[i].SpatialConvolution.bias = THCudaTensor_newFromFloatTensor(net->modules[i].SpatialConvolution.bias);
			}
			break;
		}
	}
	return nn;
}


#ifdef HAVEFP16

void tofp16(__fp16 *dst, const float *src, size_t len)
{
	size_t i;

	for(i = 0; i < len; i++)
		dst[i] = src[i];
}

void fromfp16(float *dst, const __fp16 *src, size_t len)
{
	size_t i;

	for(i = 0; i < len; i++)
		dst[i] = src[i];
}

THFloatTensor *THHalfCudaTensor_newFromFloatTensor(THFloatTensor *t)
{
	THFloatTensor *n = malloc(sizeof(*n));
	memcpy(n, t, sizeof(*n));
	if(t->storage)
	{
		n->storage = malloc(sizeof(*n->storage));
		n->storage->nref = 1;
		n->storage->mustfree = 2;
		void *tmp = malloc(THFloatTensor_nElement(t) * 2);
		tofp16(tmp, THFloatTensor_data(t), THFloatTensor_nElement(t));
		errcheck(cudaMalloc((void **)&n->storage->data, THFloatTensor_nElement(t) * 2));
		errcheck(cudaMemcpy(n->storage->data, tmp, THFloatTensor_nElement(t) * 2, cudaMemcpyHostToDevice));
		free(tmp);
	}
	return n;
}

THFloatTensor *THFloatTensor_newFromHalfCudaTensor(THFloatTensor *t)
{
	THFloatTensor *n = malloc(sizeof(*n));
	memcpy(n, t, sizeof(*n));
	if(t->storage)
	{
		n->storage = malloc(sizeof(*n->storage));
		n->storage->nref = 1;
		n->storage->mustfree = 2;
		void *tmp = malloc(THFloatTensor_nElement(t) * 2);
		errcheck(cudaMemcpy(tmp, t->storage->data, THFloatTensor_nElement(t) * 2, cudaMemcpyDeviceToHost));
		n->storage->data = malloc(THFloatTensor_nElement(t) * sizeof(*n->storage->data));
		fromfp16(n->storage->data, tmp, THFloatTensor_nElement(t));
		free(tmp);
	}
	return n;
}

#endif
