#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <limits.h>
#include "thnets.h"
#ifndef USEBLAS
#include "sgemm.h"
#endif
#ifdef ACCELERATE
#include <Accelerate/Accelerate.h>
#endif

#define THAtomicIncrement(a) __sync_fetch_and_add(a, 1);
#define THAtomicDecrement(a) __sync_fetch_and_add(a, -1);

THNStorage *THNStorage_new(long size)
{
	THNStorage *s = malloc(sizeof(*s));
	s->data = malloc(sizeof(*s->data) * size);
	if(!s->data)
		THError("Out of memory tryting to allocate %u bytes", sizeof(*s->data) * size);
	s->nref = 1;
	s->mustfree = 1;
	return s;
}

THNStorage *THNStorage_newwithbuffer(void *buffer)
{
	THNStorage *s = malloc(sizeof(*s));
	s->data = buffer;
	s->nref = 1;
	s->mustfree = 0;
	return s;
}

void THNStorage_free(THNStorage *s)
{
	THAtomicDecrement(&s->nref);
	if(s->nref == 0)
	{
#ifdef CUDNN
		if(s->mustfree == 2)
			cudaFree(s->data);
		else
#endif
#ifdef OPENCL
		if(s->mustfree == 3)
			clReleaseMemObject((cl_mem)s->data);
		else
#endif
		if(s->mustfree)
			free(s->data);
		free(s);
	}
}

void THNTensor_resize(THNTensor *t, long *size, int nDimension)
{
	int i;
	long stride = 1;
	char nostorage = 0;

	long nelem = THNTensor_nElement(t);
	t->nDimension = nDimension;
	memcpy(t->size, size, nDimension * sizeof(*t->size));
	for(i = nDimension - 1; i >= 0; i--)
	{
		t->stride[i] = stride;
		stride *= t->size[i];
		if(t->size[i] == -1)
			nostorage = 1;
	}
	if(nelem != THNTensor_nElement(t) || !t->storage)
	{
		if(nostorage)
		{
			if(t->storage)
			{
				THNStorage_free(t->storage);
				t->storage = 0;
			}
		} else if(t->storage)
			t->storage->data = realloc(t->storage->data, sizeof(*t->storage->data) * stride);
		else t->storage = THNStorage_new(stride);
	}
}

void THNTensor_resize4d(THNTensor *t, long size0, long size1, long size2, long size3)
{
	long nElement = THNTensor_nElement(t);
	t->nDimension = 4;
	t->size[0] = size0;//batch
	t->size[1] = size1;//plane
	t->size[2] = size2;//row
	t->size[3] = size3;//col

	#ifdef USEQSML
		t->stride[3] = size1;//col
		t->stride[2] = size1 * size3;//row
		t->stride[1] = 1;//plane
		t->stride[0] = size1 * size2 * size3;//batch
	#else
		t->stride[3] = 1;//col
		t->stride[2] = size3;//row
		t->stride[1] = size2 * size3;//plane
		t->stride[0] = size1 * size2 * size3;//batch
	#endif
	if(nElement != size0 * size1 * size2 * size3 || !t->storage)
	{
		if(t->storage)
			t->storage->data = realloc(t->storage->data, sizeof(*t->storage->data) * size0 * size1 * size2 * size3);
		else t->storage = THNStorage_new(size0 * size1 * size2 * size3);
	}
}

void THNTensor_resize3d(THNTensor *t, long size0, long size1, long size2)
{
	long nElement = THNTensor_nElement(t);
	t->nDimension = 3;
	t->size[0] = size0;//col
	t->size[1] = size1;//row
	t->size[2] = size2;//plane

	#ifdef USEQSML
		t->stride[2] = size2;//col
		t->stride[1] = size1 * size2;//row
		t->stride[0] = 1;//plane
	#else
		t->stride[2] = 1;//col
		t->stride[1] = size2;//row
		t->stride[0] = size1 * size2;//plane
	#endif
	if(nElement != size0 * size1 * size2 || !t->storage)
	{
		if(t->storage)
			t->storage->data = realloc(t->storage->data, sizeof(*t->storage->data) * size0 * size1 * size2);
		else t->storage = THNStorage_new(size0 * size1 * size2);
	}
}

void THNTensor_resize2d(THNTensor *t, long size0, long size1)
{
	long nElement = THNTensor_nElement(t);
	t->nDimension = 2;
	t->size[0] = size0;
	t->size[1] = size1;
	t->stride[1] = 1;
	t->stride[0] = size1;
	if(nElement != size0 * size1 || !t->storage)
	{
		if(t->storage)
			t->storage->data = realloc(t->storage->data, sizeof(*t->storage->data) * size0 * size1);
		else t->storage = THNStorage_new(size0 * size1);
	}
}

void THNTensor_resize1d(THNTensor *t, long size0)
{
	long nElement = THNTensor_nElement(t);
	t->nDimension = 1;
	t->size[0] = size0;
	t->stride[0] = 1;
	if(nElement != size0 || !t->storage)
	{
		if(t->storage)
			t->storage->data = realloc(t->storage->data, sizeof(*t->storage->data) * size0);
		else t->storage = THNStorage_new(size0);
	}
}

void THError(const char *fmt, ...)
{
	va_list ap;

	va_start(ap, fmt);
	vfprintf(stderr, fmt, ap);
	va_end(ap);
	fprintf(stderr, "\n");
	exit(-1);
}

void THNTensor_free(THNTensor *t)
{
	if(!t)
		return;
	if(t->storage)
		THNStorage_free(t->storage);
	free(t);
}

void THNTensor_slice(THNTensor *dst, THNTensor *src, int dimension, long from, long to)
{
	int i;

	if(dst->storage)
		THNStorage_free(dst->storage);
	dst->nDimension = src->nDimension;
	dst->storageOffset = from * src->stride[dimension];
	dst->size[dimension] = to - from;
	for(i = 0; i < src->nDimension; i++)
	{
		if(i != dimension)
			dst->size[i] = src->size[i];
		dst->stride[i] = src->stride[i];
	}
	dst->storage = src->storage;
	THAtomicIncrement(&dst->storage->nref);
}

THNTensor *THNTensor_newSelect(THNTensor *tensor, int dimension, long sliceIndex)
{
	int i;

	THNTensor *t = malloc(sizeof(*t));
#ifdef LOWP
	t->mult = tensor->mult;
	t->sub = tensor->sub;
#endif
	t->nDimension = tensor->nDimension - 1;
	t->storageOffset = tensor->storageOffset + sliceIndex * tensor->stride[dimension];
	for(i = 0; i < dimension; i++)
	{
		t->size[i] = tensor->size[i];
		t->stride[i] = tensor->stride[i];
	}
	for(i = dimension; i < t->nDimension; i++)
	{
		t->size[i] = tensor->size[i+1];
		t->stride[i] = tensor->stride[i+1];
	}
	t->storage = tensor->storage;
	THAtomicIncrement(&t->storage->nref);
	return t;
}

long THNTensor_nElement(THNTensor *t)
{
	long nElement = 1;
	int i;
	for(i = 0; i < t->nDimension; i++)
		nElement *= t->size[i];
	return nElement;
}

int THNTensor_isSameSizeAs(const THNTensor *self, const THNTensor* src)
{
	int d;
	if (self->nDimension != src->nDimension)
		return 0;
	for(d = 0; d < self->nDimension; ++d)
	{
		if(self->size[d] != src->size[d])
			return 0;
	}
	return 1;
}

void THNTensor_resizeAs(THNTensor *tdst, THNTensor *tsrc)
{
	if(tsrc == tdst)
		return;
	long nelemdst = THNTensor_nElement(tdst);
	long nelemsrc = THNTensor_nElement(tsrc);
	tdst->nDimension = tsrc->nDimension;
	memcpy(tdst->size, tsrc->size, sizeof(tsrc->size));
	memcpy(tdst->stride, tsrc->stride, sizeof(tsrc->stride));
	if(nelemsrc != nelemdst)
	{
		if(tdst->storage)
			tdst->storage->data = realloc(tdst->storage->data, sizeof(*tdst->storage->data) * nelemsrc);
		else tdst->storage = THNStorage_new(nelemsrc);
	}
}

void THNTensor_set(THNTensor *tdst, THNTensor *tsrc)
{
	if(tsrc == tdst)
		return;
	if(tdst->storage)
		THNStorage_free(tdst->storage);
	*tdst = *tsrc;
	if(tdst->storage)
		THAtomicIncrement(&tsrc->storage->nref);
}

float *THNTensor_data(THNTensor *tensor)
{
	if(tensor && tensor->storage && tensor->storage->data)
		return tensor->storage->data + tensor->storageOffset;
	return 0;
}

THNTensor *THNTensor_new()
{
	return calloc(1, sizeof(THNTensor));
}

THNTensor *THNTensor_newWithStorage3d(THNStorage *storage, long storageOffset, long size0, long stride0, long size1, long stride1, long size2, long stride2)
{
	THNTensor *t = THNTensor_new();
	t->nDimension = 3;
	t->size[0] = size0;
	t->size[1] = size1;
	t->size[2] = size2;
	t->stride[0] = stride0 == -1 ? size1 * size2 : stride0;
	t->stride[1] = stride1 == -1 ? size2 : stride1;
	t->stride[2] = stride2 == -1 ? 1 : stride2;
	t->storage = storage;
	t->storageOffset = storageOffset;
	THAtomicIncrement(&t->storage->nref);
	return t;
}

THNTensor *THNTensor_newWithStorage2d(THNStorage *storage, long storageOffset, long size0, long stride0, long size1, long stride1)
{
	THNTensor *t = THNTensor_new();
	t->nDimension = 2;
	t->size[0] = size0;
	t->size[1] = size1;
	t->stride[0] = stride0 == -1 ? size1 : stride0;
	t->stride[1] = stride1 == -1 ? 1 : stride1;
	t->storage = storage;
	t->storageOffset = storageOffset;
	THAtomicIncrement(&t->storage->nref);
	return t;
}

THNTensor *THNTensor_newWithStorage1d(THNStorage *storage, long storageOffset, long size0, long stride0)
{
	THNTensor *t = THNTensor_new();
	t->nDimension = 1;
	t->size[0] = size0;
	t->stride[0] = stride0 == -1 ? 1 : stride0;
	t->storage = storage;
	t->storageOffset = storageOffset;
	THAtomicIncrement(&t->storage->nref);
	return t;
}

THNTensor *THNTensor_newWithTensor(THNTensor *tensor)
{
	THNTensor *self = THNTensor_new();
	THNTensor_set(self, tensor);
	return self;
}

void THNTensor_zero(THNTensor *t)
{
	memset(t->storage->data, 0, THNTensor_nElement(t) * sizeof(*t->storage->data));
}

void THNTensor_fill(THNTensor *t, float value)
{
	THFloatVector_fill(t->storage->data, value, THNTensor_nElement(t));
}

void THNTensor_copy(THNTensor *tdst, THNTensor *tsrc)
{
	float *src, *dst;

	src = THNTensor_data(tsrc);
	dst = THNTensor_data(tdst);
	memcpy(dst, src, sizeof(*dst) * THNTensor_nElement(tsrc));
}

void THNTensor_safecopy(THNTensor *tdst, THNTensor *tsrc)
{
	float *src, *dst;
	long i0, i1, i2, i3;

	src = THNTensor_data(tsrc);
	dst = THNTensor_data(tdst);
	if(tsrc->nDimension == 1)
	{
		for(i0 = 0; i0 < tsrc->size[0]; i0++)
			dst[tdst->stride[0] * i0] = src[tsrc->stride[0] * i0];
		return;
	}
	if(tsrc->nDimension == 2)
	{
		tdst->stride[0] = tdst->size[1];
		tdst->stride[1] = 1;
		for(i0 = 0; i0 < tsrc->size[0]; i0++)
			for(i1 = 0; i1 < tsrc->size[1]; i1++)
				dst[tdst->stride[0] * i0 + tdst->stride[1] * i1] = src[tsrc->stride[0] * i0 + tsrc->stride[1] * i1];
		return;
	}
	if(tsrc->nDimension == 3)
	{
		tdst->stride[0] = tdst->size[1] * tdst->size[2];
		tdst->stride[1] = tdst->size[2];
		tdst->stride[2] = 1;
		for(i0 = 0; i0 < tsrc->size[0]; i0++)
			for(i1 = 0; i1 < tsrc->size[1]; i1++)
				for(i2 = 0; i2 < tsrc->size[2]; i2++)
					dst[tdst->stride[0] * i0 + tdst->stride[1] * i1 + tdst->stride[2] * i2] = src[tsrc->stride[0] * i0 + tsrc->stride[1] * i1 + tsrc->stride[2] * i2];
		return;
	}
	tdst->stride[0] = tdst->size[1] * tdst->size[2] * tdst->size[3];
	tdst->stride[1] = tdst->size[2] * tdst->size[3];
	tdst->stride[2] = tdst->size[3];
	tdst->stride[3] = 1;
	for(i0 = 0; i0 < tsrc->size[0]; i0++)
		for(i1 = 0; i1 < tsrc->size[1]; i1++)
			for(i2 = 0; i2 < tsrc->size[2]; i2++)
				for(i3 = 0; i3 < tsrc->size[3]; i3++)
					dst[tdst->stride[0] * i0 + tdst->stride[1] * i1 + tdst->stride[2] * i2 + tdst->stride[3] * i3] =
						src[tsrc->stride[0] * i0 + tsrc->stride[1] * i1 + tsrc->stride[2] * i2 + tsrc->stride[3] * i3];
}

void THNTensor_transpose(THNTensor *tdst, THNTensor *tsrc, int dimension1, int dimension2)
{
	long z;

	if(!tsrc)
		tsrc = tdst;

	THNTensor_set(tdst, tsrc);

	if(dimension1 == dimension2)
		return;

	z = tdst->stride[dimension1];
	tdst->stride[dimension1] = tdst->stride[dimension2];
	tdst->stride[dimension2] = z;
	z = tdst->size[dimension1];
	tdst->size[dimension1] = tdst->size[dimension2];
	tdst->size[dimension2] = z;
}

THNTensor *THNTensor_newTranspose(THNTensor *tensor, int dimension1_, int dimension2_)
{
  THNTensor *self = THNTensor_newWithTensor(tensor);
  THNTensor_transpose(self, NULL, dimension1_, dimension2_);
  return self;
}

THNTensor *THNTensor_squeeze(THNTensor *t)
{
	int ndim = 0, i;
	THNTensor *t2 = THNTensor_newWithTensor(t);
	for(i = 0; i < t->nDimension; i++)
		if(t->size[i] != 1)
		{
			if(i != ndim)
			{
				t2->size[ndim] = t->size[i];
				t2->stride[ndim] = t->stride[i];
			}
			ndim++;
		}
	t2->nDimension = ndim;
	return t2;
}

double THExpMinusApprox(double x)
{
#if EXACT_EXPONENTIAL
  return exp(-x);
#else
  /* fast approximation of exp(-x) for x positive */
# define A0   (1.0)
# define A1   (0.125)
# define A2   (0.0078125)
# define A3   (0.00032552083)
# define A4   (1.0172526e-5)
  if (x < 13.0)
  {
/*	assert(x>=0); */
	double y;
	y = A0+x*(A1+x*(A2+x*(A3+x*A4)));
	y *= y;
	y *= y;
	y *= y;
	y = 1/y;
	return y;
  }
  return 0;
# undef A0
# undef A1
# undef A2
# undef A3
# undef A4
#endif
}

void sgemm_(char *transa, char *transb, int *m, int *n, int *k, float *alpha, float *a, int *lda, float *b, int *ldb, float *beta, float *c, int *ldc);
void sger_(int *m, int *n, float *alpha, float *x, int *incx, float *y, int *incy, float *a, int *lda);
void sger(int m, int n, float alpha, float *x, int incx, float *y, int incy, float *a, int lda);
void sgemv(char trans, int m, int n, float alpha, float *a, int lda, float *x, int incx, float beta, float *y, int incy);
void sgemv_(char *trans, int *m, int *n, float *alpha, float *a, int *lda, float *x, int *incx, float *beta, float *y, int *incy);

void THBlas_gemm(char transa, char transb, long m, long n, long k, float alpha, float *a, long lda, float *b, long ldb, float beta, float *c, long ldc)
{
	int transa_ = ((transa == 't') || (transa == 'T'));
	int transb_ = ((transb == 't') || (transb == 'T'));

	if(n == 1)
		ldc = m;

	if(transa_)
	{
		if(m == 1)
			lda = k;
	}
	else
	{
		if(k == 1)
			lda = m;
	}

	if(transb_)
	{
		if(k == 1)
			ldb = n;
	}
	else
	{
		if(n == 1)
			ldb = k;
	}

	if( (m <= INT_MAX) && (n <= INT_MAX) && (k <= INT_MAX) && (lda <= INT_MAX)  && (ldb <= INT_MAX) && (ldc <= INT_MAX) )
	{
#ifdef USEBLAS
		int i_m = (int)m;
		int i_n = (int)n;
		int i_k = (int)k;
		int i_lda = (int)lda;
		int i_ldb = (int)ldb;
		int i_ldc = (int)ldc;
#ifdef ACCELERATE
		cblas_sgemm(CblasColMajor, transa == 't' ? CblasTrans : CblasNoTrans, transb == 't' ? CblasTrans : CblasNoTrans,
			i_m, i_n, i_k, alpha, a, i_lda, b, i_ldb, beta, c, i_ldc);
#else
		sgemm_(&transa, &transb, &i_m, &i_n, &i_k, &alpha, a, &i_lda, b, &i_ldb, &beta, c, &i_ldc);
#endif
#else
		sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
#endif
		return;
	}
	THError("Wrong parameters to gemm");
}

void THBlas_gemv(char trans, long m, long n, float alpha, float *a, long lda, float *x, long incx, float beta, float *y, long incy)
{
	if(n == 1)
		lda = m;

	if( (m <= INT_MAX) && (n <= INT_MAX) &&
		(lda > 0) && (lda <= INT_MAX) &&
		(incx > 0) && (incx <= INT_MAX) &&
		(incy > 0) && (incy <= INT_MAX) )
	{
#ifdef USEBLAS
		int i_m = (int)m;
		int i_n = (int)n;
		int i_lda = (int)lda;
		int i_incx = (int)incx;
		int i_incy = (int)incy;
#ifdef ACCELERATE
		cblas_sgemv(CblasColMajor, trans == 't' ? CblasTrans : CblasNoTrans, i_m, i_n, alpha, a, i_lda, x, i_incx, beta, y, i_incy);
#else
		sgemv_(&trans, &i_m, &i_n, &alpha, a, &i_lda, x, &i_incx, &beta, y, &i_incy);
#endif
#else
		sgemv(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
#endif
	}
}


void THBlas_ger(long m, long n, float alpha, float *x, long incx, float *y, long incy, float *a, long lda)
{
	if(n == 1)
		lda = m;

#ifdef USEBLAS
	int i_m = (int)m;
	int i_n = (int)n;
	int i_lda = (int)lda;
	int i_incx = (int)incx;
	int i_incy = (int)incy;
#ifdef ACCELERATE
	cblas_sger(CblasColMajor, i_m, i_n, alpha, x, i_incx, y, i_incy, a, i_lda);
#else
	sger_(&i_m, &i_n, &alpha, x, &i_incx, y, &i_incy, a, &i_lda);
#endif
#else
	sger(m, n, alpha, x, incx, y, incy, a, lda);
#endif
}

void THNTensor_addmm(THNTensor *r_, float beta, THNTensor *t, float alpha, THNTensor *m1, THNTensor *m2)
{
	char transpose_r, transpose_m1, transpose_m2;
	THNTensor *r__, *m1_, *m2_;

	if( (m1->nDimension != 2) || (m2->nDimension != 2))
		THError("matrices expected, got %dD, %dD tensors", m1->nDimension, m2->nDimension);

	if(m1->size[1] != m2->size[0])
		THError("size mismatch, m1: %ld, m2: %ld", m1->size[1], m2->size[0]);

	if( t->nDimension != 2 )
		THError("matrix expected, got %dD tensor for t", t->nDimension);

	if( (t->size[0] != m1->size[0]) || (t->size[1] != m2->size[1]) )
		THError("size mismatch, t: %ld, m1: %ld, t: %ld, m2: %ld", t->size[0], m1->size[1], t->size[1], m2->size[1]);

	if(t != r_)
		THError("Not implemented: t != r");

	/*  printf("%ldx%ld = %ldx%ld X %ldx%ld\n", r_->size[0], r_->size[1], m1->size[0], m1->size[1], m2->size[0], m2->size[1]); */

	/* r_ */
	if(r_->stride[0] == 1 && r_->stride[1] != 0)
	{
		transpose_r = 'n';
		r__ = r_;
	}
	else if(r_->stride[1] == 1 && r_->stride[0] != 0)
	{
		THNTensor *swap = m2;
		m2 = m1;
		m1 = swap;
		transpose_r = 't';
		r__ = r_;
	}
	else
	{
		THError("Transpose not implemented (1)");
		return;
/*		transpose_r = 'n';

		r__ = THNTensor_newWithSize2d(r_->size[1], r_->size[0]);
		THNTensor_copy(r__, r_);
		THNTensor_transpose(r__, NULL, 0, 1);*/
	}

	/* m1 */
	if(m1->stride[(transpose_r == 'n' ? 0 : 1)] == 1 && m1->stride[(transpose_r == 'n' ? 1 : 0)] != 0)
	{
		transpose_m1 = 'n';
		m1_ = m1;
	}
	else if(m1->stride[(transpose_r == 'n' ? 1 : 0)] == 1 && m1->stride[(transpose_r == 'n' ? 0 : 1)] != 0)
	{
		transpose_m1 = 't';
		m1_ = m1;
	}
	else
	{
		THError("Transpose not implemented (2)");
		return;
		/*transpose_m1 = (transpose_r == 'n' ? 't' : 'n');
		m1_ = THNTensor_newContiguous(m1);*/
	}

	/* m2 */
	if(m2->stride[(transpose_r == 'n' ? 0 : 1)] == 1 && m2->stride[(transpose_r == 'n' ? 1 : 0)] != 0)
	{
		transpose_m2 = 'n';
		m2_ = m2;
	}
	else if(m2->stride[(transpose_r == 'n' ? 1 : 0)] == 1 && m2->stride[(transpose_r == 'n' ? 0 : 1)] != 0)
	{
		transpose_m2 = 't';
		m2_ = m2;
	}
	else
	{
		THError("Transpose not implemented (3)");
		return;
		/*transpose_m2 = (transpose_r == 'n' ? 't' : 'n');
		m2_ = THNTensor_(newContiguous)(m2);*/
	}

	/* do the operation */
	THBlas_gemm(transpose_m1,
		transpose_m2,
		r__->size[(transpose_r == 'n' ? 0 : 1)],
		r__->size[(transpose_r == 'n' ? 1 : 0)],
		m1_->size[(transpose_r == 'n' ? 1 : 0)],
		alpha,
		THNTensor_data(m1_),
		(transpose_m1 == 'n' ? m1_->stride[(transpose_r == 'n' ? 1 : 0)] : m1_->stride[(transpose_r == 'n' ? 0 : 1)]),
		THNTensor_data(m2_),
		(transpose_m2 == 'n' ? m2_->stride[(transpose_r == 'n' ? 1 : 0)] : m2_->stride[(transpose_r == 'n' ? 0 : 1)]),
		beta,
		THNTensor_data(r__),
		r__->stride[(transpose_r == 'n' ? 1 : 0)]);

	/* free intermediate variables */
	if(m1_ != m1)
		THNTensor_free(m1_);

	if(m2_ != m2)
		THNTensor_free(m2_);

	if(r__ != r_)
		THError("freeCopyTo not implemented");
		/*THNTensor_(freeCopyTo)(r__, r_);*/
}

void THNTensor_addmv(THNTensor *r_, float beta, THNTensor *t, float alpha, THNTensor *mat, THNTensor *vec)
{
	if( (mat->nDimension != 2) || (vec->nDimension != 1) )
		THError("matrix and vector expected, got %dD, %dD", mat->nDimension, vec->nDimension);

	if( mat->size[1] != vec->size[0] )
		THError("size mismatch, %ld, %ld", mat->size[1], vec->size[0]);

	if(t->nDimension != 1)
		THError("vector expected, got t: %dD", t->nDimension);

	if(t->size[0] != mat->size[0])
		THError("size mismatch, t: %ld, mat: %ld", t->size[0], mat->size[0]);

	if(r_ != t)
		THError("r_ != t not implemented");

	if(mat->stride[0] == 1)
	{
		THBlas_gemv('n', mat->size[0], mat->size[1], alpha, THNTensor_data(mat), mat->stride[1],
			THNTensor_data(vec), vec->stride[0], beta, THNTensor_data(r_), r_->stride[0]);
	}
	else if(mat->stride[1] == 1)
	{
		THBlas_gemv('t',  mat->size[1], mat->size[0], alpha, THNTensor_data(mat), mat->stride[0],
			THNTensor_data(vec), vec->stride[0], beta, THNTensor_data(r_), r_->stride[0]);
	}
	else THError("addmv for non-contiguous not implemented");
}

#define TH_OMP_OVERHEAD_THRESHOLD 100000

void THNTensor_mul(THNTensor *r_, THNTensor *t, float value)
{
	float *tp = THNTensor_data(t);
	float *rp = THNTensor_data(r_);
	long i;
	long sz = THNTensor_nElement(t);

#pragma omp parallel for if(sz > TH_OMP_OVERHEAD_THRESHOLD) private(i)
	for (i=0; i<sz; i++)
		rp[i] = tp[i] * value;
}

void THNTensor_addr(THNTensor *r_, float beta, THNTensor *t, float alpha, THNTensor *vec1, THNTensor *vec2)
{
	if( (vec1->nDimension != 1) || (vec2->nDimension != 1) )
		THError("vector and vector expected, got %dD, %dD tensors", vec1->nDimension, vec2->nDimension);

	if(t->nDimension != 2)
		THError("expected matrix, got %dD tensor for t", t->nDimension);

	if( (t->size[0] != vec1->size[0]) || (t->size[1] != vec2->size[0]) )
		THError("size mismatch, t: %ld, vec1: %ld, t: %ld, vec2: %ld", t->size[0], vec1->size[0], t->size[1], vec2->size[0]);

	if(r_ != t)
		THError("r_ != t not implemented");

	if(beta != 1)
		THNTensor_mul(r_, r_, beta);

  if(r_->stride[0] == 1)
  {
	THBlas_ger(vec1->size[0], vec2->size[0],
				 alpha, THNTensor_data(vec1), vec1->stride[0],
				 THNTensor_data(vec2), vec2->stride[0],
				 THNTensor_data(r_), r_->stride[1]);
  }
  else if(r_->stride[1] == 1)
  {
	THBlas_ger(vec2->size[0], vec1->size[0],
				 alpha, THNTensor_data(vec2), vec2->stride[0],
				 THNTensor_data(vec1), vec1->stride[0],
				 THNTensor_data(r_), r_->stride[0]);
  }
  else THError("addr for non-contiguous not implemented");
}

void printtensor(THNTensor *t)
{
	if(t->nDimension == 2)
	{
		int i, j;

		for(i = 0; i < t->size[0]; i++)
		{
			printf("%d) ", i);
			for(j = 0; j < t->size[1]; j++)
				printf("%f ", t->storage->data[i * t->stride[0] + j]);
			printf("\n");
		}
	} else printf("printtensor: nDimension not implemented\n");
}

void THNTensor_validXCorr2Dptr(float *r_,
	float alpha,
	float *t_, long ir, long ic,
	float *k_, long kr, long kc,
	long sr, long sc)
{
	long or = (ir - kr) / sr + 1;
	long oc = (ic - kc) / sc + 1;

	long xx, yy, kx, ky;

	if ((sc != 1) || (oc < 4))  {
		/* regular convolution */
		for(yy = 0; yy < or; yy++) {
			for(xx = 0; xx < oc; xx++) {
				/* Dot product in two dimensions... (between input image and the mask) */
				float *pi_ = t_ + yy*sr*ic + xx*sc;
				float *pw_ = k_;
				float sum = 0;
				for(ky = 0; ky < kr; ky++) {
					for(kx = 0; kx < kc; kx++) {
						sum += pi_[kx]*pw_[kx];
					}
					pi_ += ic; /* next input line */
					pw_ += kc; /* next mask line */
				}
				/* Update output */
				*r_++ += alpha*sum;
			}
		}
	} else {
		/* SSE-based convolution */
		for(yy = 0; yy < or; yy++) {
			float *pi_ = t_ + yy*sr*ic;
			float *pw_ = k_;
			for (ky = 0; ky < kr; ky++) {
				float *pis_ = pi_;
				for (kx = 0; kx < kc; kx++) {
					THFloatVector_add(r_, pis_, alpha*pw_[kx], oc);
					pis_++;
				}
				pi_ += ic; /* next input line */
				pw_ += kc; /* next mask line */
			}
			r_ += oc;
		}
	}
}

void THNTensor_conv2Dmv(THNTensor *r_, float beta, float alpha, THNTensor *t_, THNTensor *k_, long srow, long scol, const char *vf, const char *xc)
{
	long nInputPlane, nInputRows, nInputCols;
	long nKernelRows, nKernelCols;
	long nOutputPlane, nOutputRows, nOutputCols;
	long istride0, kstride0, kstride1;
	THNTensor *input;
	THNTensor *kernel;
	float *input_data;
	float *weight_data;
	float *output_data;
	long nelem;
	long k;

	if(t_->nDimension != 3)
		THError("input: 3D Tensor expected");
	if(k_->nDimension != 4)
		THError("kernel: 4D Tensor expected");
	if(srow < 1)
		THError("Stride should be a positive integer");
	if(scol < 1)
		THError("Stride should be a positive integer");
	if(*vf != 'V' || *xc != 'X')
		THError("Type of convolution can be 'V','X' only");

	input = t_;
	kernel = k_;

	nInputPlane = input->size[0];
	istride0	= input->stride[0];
	nInputRows  = input->size[1];
	nInputCols  = input->size[2];

	kstride0	= kernel->stride[0];
	kstride1	= kernel->stride[1];
	nKernelRows = kernel->size[2];
	nKernelCols = kernel->size[3];
	nOutputPlane = kernel->size[0];
	if(kernel->size[1] != nInputPlane)
		THError("invalid number of input planes");
	if(!(nInputRows >= nKernelRows && nInputCols >= nKernelCols))
		THError("conv2Dmv : Input image is smaller than kernel");

	nOutputRows = (nInputRows - nKernelRows) / srow + 1;
	nOutputCols = (nInputCols - nKernelCols) / scol + 1;

	nelem = THNTensor_nElement(r_);
	THNTensor_resize3d(r_, nOutputPlane, nOutputRows, nOutputCols);

	input_data = THNTensor_data(input);
	weight_data = THNTensor_data(kernel);
	output_data = THNTensor_data(r_);

	if (nelem == 0 || beta == 0 || nelem != THNTensor_nElement(r_))
	{
		/*THNTensor_zero)(r_);*/
#pragma omp parallel for private(k)
		for (k = 0; k < r_->size[0]; k++)
		{
			float* ptr_output = output_data + k*nOutputCols*nOutputRows;
			long l;
			for (l = 0; l < nOutputRows*nOutputCols; l++)
			ptr_output[l] = 0.0;
		}
	}
	else if (beta != 1)
	{
		/*THNTensor_mul)(r_, beta);*/
#pragma omp parallel for private(k)
		for (k = 0; k < r_->size[0]; k++)
		{
			float* ptr_output = output_data + k*nOutputCols*nOutputRows;
			long l;
			for (l = 0; l < nOutputRows*nOutputCols; l++)
				ptr_output[l] *= beta;
		}
	}

#pragma omp parallel for private(k)
	for(k = 0; k < nOutputPlane; k++)
	{
		long i;
		/* get output */
		float *ptr_output = output_data + k*nOutputCols*nOutputRows;
		for(i = 0; i < nInputPlane; i++)
		{
			/* get kernel */
			float *ptr_weight = weight_data + k*kstride0 + i*kstride1;
			/* get input */
			float *ptr_input = input_data + i*istride0;

			/* do image, kernel convolution */
			THNTensor_validXCorr2Dptr(ptr_output,
				alpha,
				ptr_input,  nInputRows,  nInputCols,
				ptr_weight, nKernelRows, nKernelCols,
				srow, scol);
		}
	}
}

void THNTensor_conv2Dmm(THNTensor *r_, float beta, float alpha, THNTensor *t_, THNTensor *k_, long srow, long scol, const char *vf, const char *xc)
{
	long nInputPlane, nInputRows, nInputCols;
	long nKernelRows, nKernelCols;
	long nOutputPlane, nOutputRows, nOutputCols;
	long kstride0, kstride1;
	THNTensor *input;
	THNTensor* kernel;
	long nbatch;
	long nelem;
	float *input_data;
	float *weight_data;
	float *output_data;
	long p;

	if(t_->nDimension != 4)
		THError("input: 3D Tensor expected");
	if(k_->nDimension != 4)
		THError("kernel: 4D Tensor expected");
	if(srow < 1)
		THError("Stride should be a positive integer");
	if(scol < 1)
		THError("Stride should be a positive integer");
	if(*vf != 'V' || *xc != 'X')
		THError("Type of convolution can be 'V','X' only");

	input = t_;
	kernel = k_;

	nbatch = input->size[0];
	nInputPlane = input->size[1];
	nInputRows  = input->size[2];
	nInputCols  = input->size[3];

	kstride0	= kernel->stride[0];
	kstride1	= kernel->stride[1];
	nKernelRows = kernel->size[2];
	nKernelCols = kernel->size[3];
	nOutputPlane = kernel->size[0];
	if(kernel->size[1] != nInputPlane)
		THError("invalid number of input planes");

	if(!(nInputRows >= nKernelRows && nInputCols >= nKernelCols))
		THError("conv2Dmv : Input image is smaller than kernel");

	nOutputRows = (nInputRows - nKernelRows) / srow + 1;
	nOutputCols = (nInputCols - nKernelCols) / scol + 1;

	nelem = THNTensor_nElement(r_);
	THNTensor_resize4d(r_, nbatch, nOutputPlane, nOutputRows, nOutputCols);

	input_data = THNTensor_data(input);
	weight_data = THNTensor_data(kernel);
	output_data = THNTensor_data(r_);

	if (nelem == 0 || beta == 0 || nelem != THNTensor_nElement(r_))
	{
		/*THNTensor_(zero)(r_);*/
#pragma omp parallel for private(p)
		for (p=0; p < r_->size[0]; p++)
		{
			long k;
			for (k = 0; k < r_->size[1]; k++)
			{
				float* ptr_output = output_data + p*nOutputPlane*nOutputRows*nOutputCols + k*nOutputCols*nOutputRows;
				long l;
				for (l = 0; l < nOutputRows*nOutputCols; l++)
					ptr_output[l] = 0.0;
			}
		}
	}
	else if (beta != 1)
	{
		/*THNTensor_(mul)(r_, beta);*/
#pragma omp parallel for private(p)
		for(p=0; p < r_->size[0]; p++)
		{
			long k;
			for (k = 0; k < r_->size[1]; k++)
			{
				float* ptr_output = output_data + p*nOutputPlane*nOutputRows*nOutputCols + k*nOutputCols*nOutputRows;
				long l;
				for (l = 0; l < nOutputRows*nOutputCols; l++)
					ptr_output[l] *= beta;
			}
		}
	}

#pragma omp parallel for private(p)
	for(p=0; p < nbatch; p++)
	{
		long k;
		for(k = 0; k < nOutputPlane; k++)
		{
			long i;
			/* get output */
			float *ptr_output = output_data + p*nOutputPlane*nOutputCols*nOutputRows + k*nOutputCols*nOutputRows;
			for(i = 0; i < nInputPlane; i++)
			{
				/* get kernel */
				float *ptr_weight = weight_data + k*kstride0 + i*kstride1;
				/* get input */
				float *ptr_input = input_data + p*nInputPlane*nInputRows*nInputCols + i*nInputRows*nInputCols;

				/* do image, kernel convolution */
				THNTensor_validXCorr2Dptr(ptr_output,
					alpha,
					ptr_input,  nInputRows,  nInputCols,
					ptr_weight, nKernelRows, nKernelCols,
					srow, scol);
			}
		}
	}
}

#ifndef USEBLAS
void THNTensor_convmm(THNTensor *r, float beta, float alpha, THNTensor *filt, THNTensor *m,
	int kH, int kW, int dH, int dW, int padH, int padW)
{
	struct sgemmargs args;

	args.transa = 0;
	args.transb = 0;
	args.m = r->size[1] * r->size[2];
	args.n = r->size[0];
	args.k = filt->size[1];
	args.alpha = alpha;
	args.beta = beta;
	args.lda = m->stride[0];
	args.ldb = filt->stride[0];
	args.ldc = r->stride[0];
	args.a = THNTensor_data(m);
	args.b = THNTensor_data(filt);
	args.c = THNTensor_data(r);
	args.ks0 = kH * kW;
	args.ks1 = kW;
	args.is0 = m->stride[0];
	args.is1 = m->stride[1];
	args.ih = m->size[1];
	args.os0 = r->stride[0];
	args.os1 = r->stride[1];
	args.dW = dW;
	args.dH = dH;
	args.padW = padW;
	args.padH = padH;
	sgemmargs(&args);
}
#endif

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

#endif

#ifdef USEQSML
void init_thnets4qsml_conv(THNETWORK *network)
{
	int m, kW, kH, inP, outP;
	struct module newmod;
	for(m = 0; m < network->net->nelem; m++){
		newmod = network->net->modules[m];
		if(newmod.type==MT_SpatialConvolutionMM ||
		newmod.type==MT_SpatialConvolutionVirtMM ||
		newmod.type==MT_SpatialConvolution){
			kW = newmod.SpatialConvolution.kW;
			kH = newmod.SpatialConvolution.kH;
			inP = newmod.SpatialConvolution.nInputPlane;
			outP = newmod.SpatialConvolution.nOutputPlane;
			transform_mem(newmod,kW,kH,inP,outP);
		}
	}
}

//weight thnets[col,row,plane,outplane] -> weight qsml[outplane,plane,col,row]
void transform_mem(struct module newmod, int col, int row, int plane, int outp)
{
	int i, j, k, m, isx, idx;
	int wsize = col*row*plane*outp;
	float* weightout = THNTensor_data(newmod.SpatialConvolution.weight);
	float* weightin = (float*)malloc(wsize*sizeof(float));
	memcpy(weightin, weightout, wsize*sizeof(float));

	//LOGD("%d,%d,%d,%d, %d\n",col,row,plane,outp,wsize);
	for(m = 0; m < outp; m++) {
		for(k = 0; k < plane; k++) {
			for(j = 0;j < row; j++) {
				for(i = 0; i < col; i++) {
					isx = i + j*col + k*col*row + m*col*row*plane;
					idx = m + k*outp + i*outp*plane + j*outp*col*plane;
					weightout[idx] = weightin[isx];
				}
			}
		}
	}
}

//input thnets[col,row,plane] -> input qsml[plane,col,row]
float* transform_mem_input(float* in1, int col, int row, int plane)
{
	int i, j, k, isx, idx;
	int wsize = col*row*plane;
	float* out = (float*)malloc(wsize*sizeof(float));

	for(k = 0; k < plane; k++) {
		for(j = 0;j < row; j++) {
			for(i = 0; i < col; i++) {
				isx = i + j*col + k*col*row;
				idx = k + i*plane + j*col*plane;
				out[idx] = in1[isx];
			}
		}
	}
	return out;
}
#endif
