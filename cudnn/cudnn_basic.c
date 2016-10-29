#include <stdlib.h>
#include <string.h>
#include "../thnets.h"
#include "cublas_v2.h"

static cudnnHandle_t handle;
static cublasHandle_t blashandle;
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

cublasHandle_t THcublas_getHandle()
{
	if(!blashandle)
		cublasCreate(&blashandle);
	return blashandle;
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
	int datasize = floattype == CUDNN_DATA_HALF ? 2 : 4;
	errcheck(cudaMalloc((void **)&s->data, size * datasize));
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
		n->storageOffset = 0;
		int datasize = floattype == CUDNN_DATA_HALF ? 2 : 4;
		errcheck(cudaMalloc((void **)&n->storage->data, THFloatTensor_nElement(t) * datasize));
		errcheck(cudaMemcpy(n->storage->data, THFloatTensor_data(t), THFloatTensor_nElement(t) * datasize, cudaMemcpyHostToDevice));
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
		n->storageOffset = 0;
		n->storage->nref = 1;
		n->storage->mustfree = 1;
		n->storage->data = malloc(THFloatTensor_nElement(t) * sizeof(*n->storage->data));
		errcheck(cudaMemcpy(n->storage->data, THFloatTensor_data(t), THFloatTensor_nElement(t) * sizeof(*n->storage->data), cudaMemcpyDeviceToHost));
	}
	return n;
}

void THCudaTensor_resize4d(THFloatTensor *t, long size0, long size1, long size2, long size3)
{
	long nelemorig = THFloatTensor_nElement(t);
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
		int datasize = floattype == CUDNN_DATA_HALF ? 2 : 4;
		errcheck(cudaMalloc((void **)&t->storage->data, THFloatTensor_nElement(t) * datasize));
	} else if(nelemorig != THFloatTensor_nElement(t))
		THError("Resizing of CUDA tensors not supported");
}

void THCudaTensor_resize3d(THFloatTensor *t, long size0, long size1, long size2)
{
	long nelemorig = THFloatTensor_nElement(t);
	t->nDimension = 3;
	t->size[0] = size0;
	t->size[1] = size1;
	t->size[2] = size2;
	t->stride[2] = 1;
	t->stride[1] = size2;
	t->stride[0] = size1 * size2;
	if(!t->storage)
	{
		t->storage = malloc(sizeof(*t->storage));
		t->storage->nref = 1;
		t->storage->mustfree = 2;
		int datasize = floattype == CUDNN_DATA_HALF ? 2 : 4;
		errcheck(cudaMalloc((void **)&t->storage->data, THFloatTensor_nElement(t) * datasize));
	} else if(nelemorig != THFloatTensor_nElement(t))
		THError("Resizing of CUDA tensors not supported");
}

void THCudaTensor_resize2d(THFloatTensor *t, long size0, long size1)
{
	long nelemorig = THFloatTensor_nElement(t);
	t->nDimension = 2;
	t->size[0] = size0;
	t->size[1] = size1;
	t->stride[1] = 1;
	t->stride[0] = size1;
	if(!t->storage)
	{
		t->storage = malloc(sizeof(*t->storage));
		t->storage->nref = 1;
		t->storage->mustfree = 2;
		int datasize = floattype == CUDNN_DATA_HALF ? 2 : 4;
		errcheck(cudaMalloc((void **)&t->storage->data, THFloatTensor_nElement(t) * datasize));
	} else if(nelemorig != THFloatTensor_nElement(t))
		THError("Resizing of CUDA tensors not supported");
}

void THCudaTensor_resize1d(THFloatTensor *t, long size0)
{
	long nelemorig = THFloatTensor_nElement(t);
	t->nDimension = 1;
	t->size[0] = size0;
	t->stride[0] = 1;
	if(!t->storage)
	{
		t->storage = malloc(sizeof(*t->storage));
		t->storage->nref = 1;
		t->storage->mustfree = 2;
		int datasize = floattype == CUDNN_DATA_HALF ? 2 : 4;
		errcheck(cudaMalloc((void **)&t->storage->data, THFloatTensor_nElement(t) * datasize));
	} else if(nelemorig != THFloatTensor_nElement(t))
		THError("Resizing of CUDA tensors not supported");
}

void THCudaTensor_resizeAs(THFloatTensor *tdst, THFloatTensor *tsrc)
{
	if(tsrc == tdst)
		return;
	long nelemsrc = THFloatTensor_nElement(tsrc);
	tdst->nDimension = tsrc->nDimension;
	memcpy(tdst->size, tsrc->size, sizeof(tsrc->size));
	memcpy(tdst->stride, tsrc->stride, sizeof(tsrc->stride));
	if(!tdst->storage)
		tdst->storage = THCudaStorage_new(nelemsrc);
	else if(nelemsrc != THFloatTensor_nElement(tdst))
		THError("Resizing of CUDA tensors not supported");
}

struct network *THcudnn_ToCUDNN(struct network *net)
{
	int i;
	struct network *nn = malloc(sizeof(*nn));

	nn->nelem = net->nelem;
	nn->modules = malloc(sizeof(net->modules[0]) * net->nelem);
	nn->engine = ENGINE_CUDA;
	memcpy(nn->modules, net->modules, sizeof(net->modules[0]) * net->nelem);
	for(i = 0; i < net->nelem; i++)
	{
		nn->modules[i].output = THCudaTensor_newFromFloatTensor(net->modules[i].output);
		nn->modules[i].net = nn;
		switch(net->modules[i].type)
		{
		case MT_SpatialConvolutionMM:
		case MT_SpatialConvolution:
		case MT_SpatialConvolutionVirtMM:
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
			nn->modules[i].SpatialMaxPooling.indices = THCudaTensor_newFromFloatTensor(net->modules[i].SpatialMaxPooling.indices);
			nn->modules[i].updateOutput = cunn_SpatialMaxPooling_updateOutput;
			break;
		case MT_SpatialMaxUnpooling:
			nn->modules[i].updateOutput = cunn_SpatialMaxUnpooling_updateOutput;
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
			nn->modules[i].Dropout.inplace = 1;
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
		case MT_SpatialBatchNormalization:
			nn->modules[i].updateOutput = cudnn_SpatialBatchNormalization_updateOutput;
			nn->modules[i].SpatialBatchNormalization.weight = THCudaTensor_newFromFloatTensor(net->modules[i].SpatialBatchNormalization.weight);
			nn->modules[i].SpatialBatchNormalization.bias = THCudaTensor_newFromFloatTensor(net->modules[i].SpatialBatchNormalization.bias);
			nn->modules[i].SpatialBatchNormalization.running_mean = THCudaTensor_newFromFloatTensor(net->modules[i].SpatialBatchNormalization.running_mean);
			nn->modules[i].SpatialBatchNormalization.running_var = THCudaTensor_newFromFloatTensor(net->modules[i].SpatialBatchNormalization.running_var);
			break;
		case MT_SpatialFullConvolution:
			nn->modules[i].updateOutput = cunn_SpatialFullConvolution_updateOutput;
#ifdef HAVEFP16
			if(floattype == CUDNN_DATA_HALF)
			{
				nn->modules[i].SpatialFullConvolution.weight = THHalfCudaTensor_newFromFloatTensor(net->modules[i].SpatialFullConvolution.weight);
				nn->modules[i].SpatialFullConvolution.bias = THHalfCudaTensor_newFromFloatTensor(net->modules[i].SpatialFullConvolution.bias);
				nn->modules[i].SpatialFullConvolution.columns = THHalfCudaTensor_newFromFloatTensor(net->modules[i].SpatialFullConvolution.columns);
				nn->modules[i].SpatialFullConvolution.ones = THHalfCudaTensor_newFromFloatTensor(net->modules[i].SpatialFullConvolution.ones);
			} else
#endif
			{
				nn->modules[i].SpatialFullConvolution.weight = THCudaTensor_newFromFloatTensor(net->modules[i].SpatialFullConvolution.weight);
				nn->modules[i].SpatialFullConvolution.bias = THCudaTensor_newFromFloatTensor(net->modules[i].SpatialFullConvolution.bias);
				nn->modules[i].SpatialFullConvolution.columns = THCudaTensor_newFromFloatTensor(net->modules[i].SpatialFullConvolution.columns);
				nn->modules[i].SpatialFullConvolution.ones = THCudaTensor_newFromFloatTensor(net->modules[i].SpatialFullConvolution.ones);
			}
			break;
		case MT_SpatialAveragePooling:
			THError("MT_SpatialAveragePooling not supported in CUDNN");
			break;
		case MT_Sequential:
			THError("MT_Sequential not supported in CUDNN");
			break;
		case MT_Concat:
			THError("MT_Concat not supported in CUDNN");
			break;
		}
	}
	return nn;
}

void adjustLd(char transa, char transb, long m, long n, long k, long *lda, long *ldb, long *ldc)
{
	int transa_ = ((transa == 't') || (transa == 'T'));
	int transb_ = ((transb == 't') || (transb == 'T'));

	if(n == 1)
		*ldc = m;

	if(transa_)
	{
		if(m == 1)
			*lda = k;
	}
	else
	{
		if(k == 1)
			*lda = m;
	}

	if(transb_)
	{
		if(k == 1)
			*ldb = n;
	}
	else
	{
		if(n == 1)
			*ldb = k;
	}
}

void THCudaBlas_gemm(char transa, char transb, long m, long n, long k, float alpha, float *a, long lda, float *b, long ldb, float beta, float *c, long ldc)
{
	adjustLd(transa, transb, m, n, k, &lda, &ldb, &ldc);

	if( (m <= INT_MAX) && (n <= INT_MAX) && (k <= INT_MAX) && (lda <= INT_MAX)  && (ldb <= INT_MAX) && (ldc <= INT_MAX) )
	{
		int i_m = (int)m;
		int i_n = (int)n;
		int i_k = (int)k;
		int i_lda = (int)lda;
		int i_ldb = (int)ldb;
		int i_ldc = (int)ldc;

#ifdef HAVEFP16
		if(floattype == CUDNN_DATA_HALF)
		{
			__half h_alpha;
			__half h_beta;
			tofp16((__fp16 *)&h_alpha, &alpha, 1);
			tofp16((__fp16 *)&h_beta, &beta, 1);
			errcheck(cublasHgemm(THcublas_getHandle(), transa == 't' ? CUBLAS_OP_T : CUBLAS_OP_N,
				transb == 't' ? CUBLAS_OP_T : CUBLAS_OP_N,
				i_m, i_n, i_k, &h_alpha, (__half *)a, i_lda, (__half *)b, i_ldb, &h_beta, (__half *)c, i_ldc));
		} else
#endif
		errcheck(cublasSgemm(THcublas_getHandle(), transa == 't' ? CUBLAS_OP_T : CUBLAS_OP_N,
			transb == 't' ? CUBLAS_OP_T : CUBLAS_OP_N,
			i_m, i_n, i_k, &alpha, a, i_lda, b, i_ldb, &beta, c, i_ldc));
		return;
	}
	THError("Cublas_gemm only supports m, n, k, lda, ldb, ldc with the bound [val] <= %d", INT_MAX);
}

void THCudaTensor_Ones(THFloatTensor *t)
{
	int n1 = 0, n2 = 0, stride = 0;
	
	if(t->nDimension == 2)
	{
		n1 = t->size[0];
		n2 = t->size[1];
		stride = t->stride[0];
	} else if(t->nDimension == 1)
	{
		n1 = 1;
		n2 = t->size[0];
	} else THError("Unsupported nDimension for THCudaTensor_Ones");

#ifdef HAVEFP16
	if(floattype == CUDNN_DATA_HALF)
		cuda_fillwithoneH(n1, n2, THFloatTensor_data(t), stride);
	else
#endif
		cuda_fillwithone(n1, n2, THFloatTensor_data(t), stride);
}

#ifdef HAVEFP16

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
		n->storage->mustfree = 1;
		void *tmp = malloc(THFloatTensor_nElement(t) * 2);
		errcheck(cudaMemcpy(tmp, t->storage->data, THFloatTensor_nElement(t) * 2, cudaMemcpyDeviceToHost));
		n->storage->data = malloc(THFloatTensor_nElement(t) * sizeof(*n->storage->data));
		fromfp16(n->storage->data, tmp, THFloatTensor_nElement(t));
		free(tmp);
	}
	return n;
}

#endif
