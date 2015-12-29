#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "thnets.h"

static int lasterror, longsize = 8;

#ifdef CUDNN
int cuda_maphostmem;
#endif

THFloatTensor *forward(struct network *net, THFloatTensor *in)
{
	int i;
	
	for(i = 0; i < net->nelem; i++)
	{
		in = net->modules[i].updateOutput(&net->modules[i], in);
		//printf("%d) %d %d %ld %ld %ld %ld\n", i+1, net->modules[i].type, in->nDimension, in->size[0], in->size[1], in->size[2], in->size[3]);
	}
	return in;
}

THNETWORK *THLoadNetwork(const char *path)
{
	char tmppath[255];
	int i;
	
	THNETWORK *net = calloc(1, sizeof(*net));
	sprintf(tmppath, "%s/model.net", path);
	net->netobj = malloc(sizeof(*net->netobj));
	lasterror = loadtorch(tmppath, net->netobj, longsize);
	if(lasterror)
	{
		free(net->netobj);
		free(net);
		return 0;
	}
	//printobject(net->netobj, 0);
	net->net = Object2Network(net->netobj);
	if(!net->net)
	{
		lasterror = ERR_WRONGOBJECT;
		freeobject(net->netobj);
		free(net->netobj);
		free(net);
		return 0;
	}
	sprintf(tmppath, "%s/stat.t7", path);
	net->statobj = malloc(sizeof(*net->statobj));
	lasterror = loadtorch(tmppath, net->statobj, longsize);
	if(lasterror)
	{
		free(net->statobj);
		freenetwork(net->net);
		freeobject(net->netobj);
		free(net->netobj);
		free(net);
		return 0;
	}
	if(net->statobj->type != TYPE_TABLE || net->statobj->table->nelem != 2)
	{
		lasterror = ERR_WRONGOBJECT;
		freenetwork(net->net);
		freeobject(net->netobj);
		free(net->netobj);
		freeobject(net->statobj);
		free(net->statobj);
		free(net);
	}
	net->std[0] = net->std[1] = net->std[2] = 1;
	net->mean[0] = net->mean[1] = net->mean[2] = 0;
	for(i = 0; i < net->statobj->table->nelem; i++)
		if(net->statobj->table->records[i].name.type == TYPE_STRING)
		{
			if(!strcmp(net->statobj->table->records[i].name.string.data, "mean"))
				memcpy(net->mean, net->statobj->table->records[i].value.tensor->storage->data, sizeof(net->mean));
			else if(!strcmp(net->statobj->table->records[i].name.string.data, "std"))
				memcpy(net->std, net->statobj->table->records[i].value.tensor->storage->data, sizeof(net->std));
		}
	return net;
}

void THInit()
{
#if defined CUDNN && defined USECUDAHOSTALLOC
	static int init;

	if(init)
		return;
	init = 1;
	// cuda_maphostmem = 1 requires that memory was allocated with cudaHostAlloc
	// cuda_maphostmem = 2 will work with malloc, but Tegra TX1 does not support cudaHostRegister with cudaHostRegisterMapped
	struct cudaDeviceProp prop;

	cudaGetDeviceProperties(&prop, 0);
	if(prop.canMapHostMemory)
	{
		errcheck(cudaSetDeviceFlags(cudaDeviceMapHost));
		cuda_maphostmem = 1;
	}
#endif
}

int THProcessFloat(THNETWORK *network, float *data, int batchsize, int width, int height, float **result, int *outwidth, int *outheight)
{
	int b, c, i;
	THFloatTensor *t = THFloatTensor_new();
	THFloatTensor *out;
	t->nDimension = 4;
	t->size[0] = batchsize;
	t->size[1] = 3;
	t->size[2] = height;
	t->size[3] = width;
	t->stride[0] = 3 * width * height;
	t->stride[1] = width * height;
	t->stride[2] = width;
	t->stride[3] = 1;
	t->storage = THFloatStorage_newwithbuffer((float *)data);
#pragma omp parallel for private(b)
	for(b = 0; b < batchsize; b++)
		for(c = 0; c < 3; c++)
			for(i = 0; i < width*height; i++)
				data[b * t->stride[0] + c * t->stride[1] + i] =
					(data[b * t->stride[0] + c * t->stride[1] + i] - network->mean[c]) / network->std[c];
#ifdef CUDNN
	if(network->net->cuda)
	{
		THFloatTensor *t2 = THCudaTensor_newFromFloatTensor(t);
		out = forward(network->net, t2);
		THFloatTensor_free(t2);
		if(network->out)
			THFloatTensor_free(network->out);
		network->out = THFloatTensor_newFromCudaTensor(out);
		out = network->out;
	} else
#endif
	out = forward(network->net, t);
	THFloatTensor_free(t);
	*result = out->storage->data;
	if(out->nDimension >= 3)
	{
		*outwidth = out->size[out->nDimension - 1];
		*outheight = out->size[out->nDimension - 2];
	} else *outwidth = *outheight = 1;
	return THFloatTensor_nElement(out);
}

#define BYTE2FLOAT 0.003921568f // 1/255

void rgb2float(float *dst, const unsigned char *src, int width, int height, int srcstride, const float *mean, const float *std)
{
	int c, i, j;

	for(c = 0; c < 3; c++)
		for(i = 0; i < height; i++)
			for(j = 0; j < width; j++)
				dst[j + (i + c * height) * width] = (src[c + 3*j + srcstride*i] * BYTE2FLOAT - mean[c]) / std[c];
}

int THProcessImages(THNETWORK *network, unsigned char **images, int batchsize, int width, int height, int stride, float **results, int *outwidth, int *outheight)
{
	int i;
	THFloatTensor *out;
	THFloatStorage *st;
	
#ifdef CUDNN
	if(network->net->cuda)
	{
#ifdef HAVEFP16
		if(floattype == CUDNN_DATA_HALF)
		{
			st = THCudaStorage_new(batchsize * ((width * height * 3 + 1) / 2));
			for(i = 0; i < batchsize; i++)
				cuda_rgb2half(st->data + i * ((width * height * 3 + 1) / 2), images[i], width, height, stride, network->mean, network->std);
		} else
#endif
		{
			st = THCudaStorage_new(batchsize * width * height * 3);
			for(i = 0; i < batchsize; i++)
				cuda_rgb2float(st->data + i * width * height * 3, images[i], width, height, stride, network->mean, network->std);
		}
	} else
#endif
	{
		st = THFloatStorage_new(batchsize * width * height * 3);
#pragma omp parallel for private(i)
		for(i = 0; i < batchsize; i++)
			rgb2float(st->data + i * width * height * 3, images[i], width, height, stride, network->mean, network->std);
	}
	THFloatTensor *t = THFloatTensor_new();
	t->storage = st;
	if(batchsize == 1)
	{
		t->nDimension = 3;
		t->size[0] = 3;
		t->size[1] = height;
		t->size[2] = width;
		t->stride[0] = width * height;
		t->stride[1] = width;
		t->stride[2] = 1;
	} else {
		t->nDimension = 4;
		t->size[0] = batchsize;
		t->size[1] = 3;
		t->size[2] = height;
		t->size[3] = width;
		t->stride[0] = 3 * width * height;
		t->stride[1] = width * height;
		t->stride[2] = width;
		t->stride[3] = 1;
	}
#ifdef CUDNN
	if(network->net->cuda)
	{
		out = forward(network->net, t);
		if(network->out)
			THFloatTensor_free(network->out);
#ifdef HAVEFP16
		if(floattype == CUDNN_DATA_HALF)
			network->out = THFloatTensor_newFromHalfCudaTensor(out);
		else
#endif
			network->out = THFloatTensor_newFromCudaTensor(out);
		out = network->out;
	} else
#endif
		out = forward(network->net, t);
	THFloatTensor_free(t);
	*results = out->storage->data;
	if(out->nDimension >= 3)
	{
		*outwidth = out->size[out->nDimension - 1];
		*outheight = out->size[out->nDimension - 2];
	} else *outwidth = *outheight = 1;
	return THFloatTensor_nElement(out);
}

void THFreeNetwork(THNETWORK *network)
{
	freenetwork(network->net);
	if(network->netobj)
	{
		freeobject(network->netobj);
		free(network->netobj);
	}
	if(network->statobj)
	{
		freeobject(network->statobj);
		free(network->statobj);
	}
	if(network->out)
		THFloatTensor_free(network->out);
	free(network);
}

int THLastError()
{
	return lasterror;
}

void THMakeSpatial(THNETWORK *network)
{
	int i, size = 231, nInputPlane = 3;
	
	for(i = 0; i < network->net->nelem; i++)
	{
		if(network->net->modules[i].type == MT_View || network->net->modules[i].type == MT_Reshape)
		{
			THFloatTensor_free(network->net->modules[i].output);
			memmove(network->net->modules+i, network->net->modules+i+1, sizeof(*network->net->modules) * (network->net->nelem - i - 1));
			network->net->nelem--;
			i--;
		} else if(network->net->modules[i].type == MT_Linear)
		{
			THFloatTensor_free(network->net->modules[i].Linear.addBuffer);
			network->net->modules[i].type = MT_SpatialConvolutionMM;
			network->net->modules[i].updateOutput = nn_SpatialConvolutionMM_updateOutput;
			struct SpatialConvolution *c = &network->net->modules[i].SpatialConvolution;
			c->finput = THFloatTensor_new();
			c->padW = c->padH = 0;
			c->dW = c->dH = 1;
			c->kW = c->kH = size;
			c->nInputPlane = nInputPlane;
			nInputPlane = c->nOutputPlane = c->weight->size[0];
			size = (size + 2*c->padW - c->kW) / c->dW + 1;
		} else if(network->net->modules[i].type == MT_SpatialConvolutionMM)
		{
			struct SpatialConvolution *c = &network->net->modules[i].SpatialConvolution;
			size = (size + 2*c->padW - c->kW) / c->dW + 1;
			nInputPlane = network->net->modules[i].SpatialConvolution.nOutputPlane;
		} else if(network->net->modules[i].type == MT_SpatialMaxPooling)
		{
			struct SpatialMaxPooling *c = &network->net->modules[i].SpatialMaxPooling;
			if(c->ceil_mode)
				size = (long)(ceil((float)(size - c->kH + 2*c->padH) / c->dH)) + 1;
			else size = (long)(floor((float)(size - c->kH + 2*c->padH) / c->dH)) + 1;
		} else if(network->net->modules[i].type == MT_SpatialZeroPadding)
		{
			struct SpatialZeroPadding *c = &network->net->modules[i].SpatialZeroPadding;
			size += c->pad_l + c->pad_r;
		}
	}
}

int THUseSpatialConvolutionMM(THNETWORK *network, int mm_on)
{
	int i;
	int rc = 0;

	for(i = 0; i < network->net->nelem; i++)
	{
		if(mm_on && network->net->modules[i].type == MT_SpatialConvolution)
		{
			struct SpatialConvolution *c = &network->net->modules[i].SpatialConvolution;
			network->net->modules[i].type = MT_SpatialConvolutionMM;
			network->net->modules[i].updateOutput = nn_SpatialConvolutionMM_updateOutput;
			if(c->weight->nDimension == 4)
				THFloatTensor_resize2d(c->weight, c->weight->size[0], c->weight->size[1] * c->weight->size[2] * c->weight->size[3]);
		} else if(!mm_on && network->net->modules[i].type == MT_SpatialConvolutionMM)
		{
			struct SpatialConvolution *c = &network->net->modules[i].SpatialConvolution;
			if(c->padW || c->padH)
			{
				rc = ERR_NOTIMPLEMENTED;
				continue;
			}
			network->net->modules[i].type = MT_SpatialConvolution;
			network->net->modules[i].updateOutput = nn_SpatialConvolution_updateOutput;
			if(c->weight->nDimension == 2)
				THFloatTensor_resize4d(c->weight, c->nOutputPlane, c->nInputPlane, c->kH, c->kW);
		}
	}
	return rc;
}

THNETWORK *THCreateCudaNetwork(THNETWORK *net)
{
#ifdef CUDNN
	THNETWORK *nn = malloc(sizeof(*nn));
	memcpy(nn, net, sizeof(*nn));
	nn->netobj = 0;
	nn->statobj = 0;
	nn->net = THcudnn_ToCUDNN(net->net);
	return nn;
#else
	return 0;
#endif
}

int THCudaHalfFloat(int enable)
{
#if defined CUDNN && defined HAVEFP16
	if(enable)
	{
		floattype = CUDNN_DATA_HALF;
	} else floattype = CUDNN_DATA_FLOAT;
	return 0;
#else
	return ERR_NOTIMPLEMENTED;
#endif
}
