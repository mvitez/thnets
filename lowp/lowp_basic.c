#include <string.h>
#include <math.h>
#include "../thnets.h"

THFloatTensor *THLowpTensor_newFromFloatTensor(THFloatTensor *t)
{
	THFloatTensor *n = malloc(sizeof(*n));
	memcpy(n, t, sizeof(*n));
	if(t->storage)
	{
		n->storage = malloc(sizeof(*n->storage));
		n->storage->nref = 1;
		n->storage->mustfree = 1;
		n->storageOffset = 0;
		int len = THFloatTensor_nElement(t) ;
		n->storage->data = malloc(len);
		float *buf = THFloatTensor_data(t);
		float min = 0, max = 0, mult;
		int i;
		for(i = 0; i < len; i++)
		{
			if(buf[i] < min)
				min = buf[i];
			if(buf[i] > max)
				max = buf[i];
		}
		if(max - min > 0)
			mult = 255.0 / (max - min);
		else mult = 0;
		unsigned char *dst = (unsigned char *)n->storage->data;
		for(i = 0; i < len; i++)
			//dst[i] = roundf((buf[i] - min) * mult);
			dst[i] = roundf(buf[i] * mult) - roundf(min*mult);
		n->sub = min;
		n->mult = mult;
	}
	return n;
}

THFloatTensor *THFloatTensor_newFromLowpTensor(THFloatTensor *t)
{
	THFloatTensor *n = malloc(sizeof(*n));
	memcpy(n, t, sizeof(*n));
	if(t->storage)
	{
		n->storage = malloc(sizeof(*n->storage));
		n->storage->nref = 1;
		n->storage->mustfree = 1;
		n->storageOffset = 0;
		int i, len = THFloatTensor_nElement(t) ;
		n->storage->data = (float *)malloc(len * sizeof(*n->storage->data));
		unsigned char *buf = (unsigned char *)THFloatTensor_data(t);
		float invmult = t->mult ? 1.0 / t->mult : 0;
		for(i = 0; i < len; i++)
			n->storage->data[i] = buf[i] * invmult + t->sub;
	}
	return n;
}

static void rgb2tensord(unsigned char *dst, const unsigned char *src, int width, int height, int srcstride, const int *sub, const float *mult)
{
	int c, i, j;

#pragma omp parallel for private(c, i, j)
	for(c = 0; c < 3; c++)
		for(i = 0; i < height; i++)
			for(j = 0; j < width; j++)
				dst[j + (i + c * height) * width] = roundf((src[c + 3*j + srcstride*i] - sub[c]) * mult[c]);
}

static void bgr2tensord(unsigned char *dst, const unsigned char *src, int width, int height, int srcstride, const int *sub, const float *mult)
{
	int c, i, j;

#pragma omp parallel for private(c, i, j)
	for(c = 0; c < 3; c++)
		for(i = 0; i < height; i++)
			for(j = 0; j < width; j++)
				dst[j + (i + c * height) * width] = roundf((src[2-c + 3*j + srcstride*i] - sub[c]) * mult[c]);
}

THFloatTensor *Lowp_LoadImages(unsigned char **src, int nimages, int width, int height, int srcstride, const float *mean, const float *std, int bgr)
{
	int i, sub[3];
	float mult[3], min, max;

	THFloatTensor *n = malloc(sizeof(*n));
	n->nDimension = 4;
	n->size[0] = nimages;
	n->size[1] = 3;
	n->size[2] = height;
	n->size[3] = width;
	n->stride[3] = 1;
	n->stride[2] = width;
	n->stride[1] = width * height;
	n->stride[0] = width * height * 3;
	n->storageOffset = 0;
	n->storage = (THFloatStorage *)malloc(sizeof(*n->storage));
	n->storage->nref = 1;
	n->storage->mustfree = 1;
	n->storage->data = (float *)malloc(nimages * n->stride[0]);
	min = 1e30;
	max = -1e30;
	for(i = 0; i < 3; i++)
	{
		if(-mean[i] / std[i] < min)
			min = -mean[i] / std[i];
		if((1-mean[i]) / std[i] > max)
			max = (1-mean[i]) / std[i];
	}
	n->sub = min;
	if(max - min)
		n->mult = 255 / (max - min);
	else n->mult = 0;
	for(i = 0; i < 3; i++)
	{
		sub[i] = roundf(255 * (mean[i] + std[i] * n->sub));
		mult[i] = n->mult / (255 * std[i]);
	}
	if(bgr)
	{
#pragma omp parallel for if(nimages>1) private(i)
		for(i = 0; i < nimages; i++)
			bgr2tensord((unsigned char *)n->storage->data + i * width * height * 3, src[i], width, height, srcstride, sub, mult);
	} else {
#pragma omp parallel for if(nimages>1) private(i)
		for(i = 0; i < nimages; i++)
			rgb2tensord((unsigned char *)n->storage->data + i * width * height * 3, src[i], width, height, srcstride, sub, mult);
	}
	return n;
}

struct network *THLowp_ToLowp(struct network *net, float range)
{
	int i;
	struct network *nn = malloc(sizeof(*nn));
	float sub = -range / 2;
	float mult = 255 / range;

	nn->nelem = net->nelem;
	nn->modules = malloc(sizeof(net->modules[0]) * net->nelem);
	nn->engine = ENGINE_LOWP;
	memcpy(nn->modules, net->modules, sizeof(net->modules[0]) * net->nelem);
	for(i = 0; i < net->nelem; i++)
	{
		nn->modules[i].output = THLowpTensor_newFromFloatTensor(net->modules[i].output);
		nn->modules[i].output->mult = mult;
		nn->modules[i].output->sub = sub;
		nn->modules[i].net = nn;
		switch(net->modules[i].type)
		{
		case MT_SpatialConvolutionMM:
		case MT_SpatialConvolution:
		case MT_SpatialConvolutionVirtMM:
			nn->modules[i].updateOutput = Lowp_SpatialConvolution_updateOutput;
			nn->modules[i].SpatialConvolution.weight = THLowpTensor_newFromFloatTensor(net->modules[i].SpatialConvolution.weight);
			nn->modules[i].SpatialConvolution.bias = THLowpTensor_newFromFloatTensor(net->modules[i].SpatialConvolution.bias);
			nn->modules[i].SpatialConvolution.finput = THFloatTensor_new();
			break;
		case MT_SpatialMaxPooling:
			nn->modules[i].SpatialMaxPooling.indices = 0;
			nn->modules[i].updateOutput = Lowp_SpatialMaxPooling_updateOutput;
			break;
		case MT_SpatialMaxUnpooling:
			THError("MT_SpatialMaxUnpooling not supported in Lowp");
			break;
		case MT_Threshold:
			nn->modules[i].updateOutput = Lowp_Threshold_updateOutput;
			break;
		case MT_SoftMax:
			nn->modules[i].updateOutput = Lowp_SoftMax_updateOutput;
			break;
		case MT_Dropout:
			if(!nn->modules[i].Dropout.v2)
				THError("Non v2 dropout not supported in Lowp");
			break;
		case MT_SpatialZeroPadding:
			THError("SpatialZeroPadding not supported in Lowp");
			break;
		case MT_Linear:
			nn->modules[i].type = MT_SpatialConvolutionMM;
			nn->modules[i].updateOutput = Lowp_SpatialConvolution_updateOutput;
			struct SpatialConvolution *c = &nn->modules[i].SpatialConvolution;
			c->finput = 0;
			c->padW = c->padH = 0;
			c->dW = c->dH = 1;
			c->kW = c->kH = 1;
			c->nOutputPlane = c->weight->size[0];
			c->nInputPlane = c->weight->size[1];
			nn->modules[i].SpatialConvolution.weight = THLowpTensor_newFromFloatTensor(net->modules[i].SpatialConvolution.weight);
			nn->modules[i].SpatialConvolution.bias = THLowpTensor_newFromFloatTensor(net->modules[i].SpatialConvolution.bias);
			nn->modules[i].SpatialConvolution.finput = THFloatTensor_new();
			break;
		case MT_SpatialBatchNormalization:
			THError("MT_SpatialBatchNormalization not supported in Lowp");
			break;
		case MT_SpatialFullConvolution:
			THError("MT_SpatialFullConvolution not supported in Lowp");
			break;
		case MT_SpatialAveragePooling:
			THError("MT_SpatialAveragePooling not supported in lowp");
			break;
		case MT_Sequential:
			THError("MT_Sequential not supported in lowp");
			break;
		case MT_Concat:
			THError("MT_Concat not supported in lowp");
			break;
		}
	}
	return nn;
}

unsigned char THLowp_ScaleFloat(THFloatTensor *t, float value)
{
	float scaled = (value - t->sub) * t->mult;
	if(scaled < 0)
		return 0;
	if(scaled > 255)
		return 255;
	return (unsigned char)scaled;
}

THFloatStorage *THLowpStorage_new(long size)
{
	THFloatStorage *s = malloc(sizeof(*s));
	s->data = malloc(size);
	if(!s->data)
		THError("Out of memory");
	s->nref = 1;
	s->mustfree = 1;
	return s;
}

void THLowpTensor_resizeAs(THFloatTensor *tdst, THFloatTensor *tsrc)
{
	if(tsrc == tdst)
		return;
	long nelemsrc = THFloatTensor_nElement(tsrc);
	long nelemdst = THFloatTensor_nElement(tdst);
	tdst->nDimension = tsrc->nDimension;
	memcpy(tdst->size, tsrc->size, sizeof(tsrc->size));
	memcpy(tdst->stride, tsrc->stride, sizeof(tsrc->stride));
	if(nelemsrc != nelemdst)
	{
		if(tdst->storage)
			tdst->storage->data = realloc(tdst->storage->data, nelemsrc);
		else tdst->storage = THLowpStorage_new(nelemsrc);
	}
}

void THLowpTensor_resize4d(THFloatTensor *t, long size0, long size1, long size2, long size3)
{
	long nElement = THFloatTensor_nElement(t);
	t->nDimension = 4;
	t->size[0] = size0;
	t->size[1] = size1;
	t->size[2] = size2;
	t->size[3] = size3;
	t->stride[3] = 1;
	t->stride[2] = size3;
	t->stride[1] = size2 * size3;
	t->stride[0] = size1 * size2 * size3;
	if(nElement != size0 * size1 * size2 * size3)
	{
		if(t->storage)
			t->storage->data = realloc(t->storage->data, size0 * size1 * size2 * size3);
		else t->storage = THLowpStorage_new(size0 * size1 * size2 * size3);
	}
}

void THLowpTensor_resize3d(THFloatTensor *t, long size0, long size1, long size2)
{
	long nElement = THFloatTensor_nElement(t);
	t->nDimension = 3;
	t->size[0] = size0;
	t->size[1] = size1;
	t->size[2] = size2;
	t->stride[2] = 1;
	t->stride[1] = size2;
	t->stride[0] = size1 * size2;
	if(nElement != size0 * size1 * size2)
	{
		if(t->storage)
			t->storage->data = realloc(t->storage->data, size0 * size1 * size2);
		else t->storage = THLowpStorage_new(size0 * size1 * size2);
	}
}

void lowpgemm(const int is_a_transposed,
	const int is_b_transposed,
	const int is_c_transposed,
	const int m, const int n, const int k,
	const unsigned char *a, const unsigned char *b, unsigned char *c,
	const int lda, const int ldb, const int ldc,
	const int a_offset, const int b_offset, const int c_offset,
	const int c_mult, const int c_shift);

void THLowpTensor_mm(THFloatTensor *r_, THFloatTensor *m1, THFloatTensor *m2)
{
	const int transpose_m1 = 1;
	const int transpose_m2 = 1;
	const int transpose_r_ = 1;

	const int m = m1->size[0];
	const int n = m2->size[1];
	const int k = m1->size[1];
	
	float scaling = m1->mult * m2->mult / r_->mult;
	int shift = roundf((log(scaling) / log(2))) + 10;	// 10 is to keep mult a 10 bit number to keep 8 bit precision
	int mult = roundf((1<<shift) / scaling);
	int offset = roundf(-r_->sub * m1->mult * m2->mult);
	
	lowpgemm(transpose_m1, transpose_m2, transpose_r_,
		m, n, k,
		(unsigned char *)THFloatTensor_data(m1),
		(unsigned char *)THFloatTensor_data(m2),
		(unsigned char *)THFloatTensor_data(r_),
		k, n, n,
		roundf(m1->sub * m1->mult), roundf(m2->sub * m2->mult), offset, mult, shift);
}
