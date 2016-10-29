#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <ctype.h>
#include <sys/time.h>
#include <math.h>
#include <CL/opencl.h>
#include "../thnets.h"

static cl_context ctx;
static cl_device_id device = 0;
int cl_datasize = 4;
cl_command_queue cl_queue;
int cl_order = 1;

int thopencl_init()
{
	cl_int err;
	cl_platform_id platform = 0;
	cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };
	size_t param[5];
	cl_ulong ulparam;
	cl_uint uiparam;
	
	char version[300];
	int i, n;

    err = clGetPlatformIDs( 1, &platform, NULL );
	if(err)
		THError("clGetPlatformIDs failed, err=%d\n", err);
    err = clGetDeviceIDs( platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL );
	if(err)
		THError("clGetDeviceIDs failed, err=%d\n", err);

    props[1] = (cl_context_properties)platform;
    ctx = clCreateContext( props, 1, &device, NULL, NULL, &err );
	if(err)
		THError("clCreateContext failed, err=%d\n", err);

   	cl_queue = clCreateCommandQueue( ctx, device, 0, &err);

	if(th_debug >= 1)
	{
		clGetDeviceInfo(device, CL_DEVICE_VERSION, sizeof(version), version, NULL);
		printf("CL_DEVICE_VERSION=%s\n", version);
		clGetDeviceInfo(device, CL_DRIVER_VERSION, sizeof(version), version, NULL);
		printf("CL_DRIVER_VERSION=%s\n", version);
		clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(ulparam), &ulparam, NULL);
		printf("CL_DEVICE_LOCAL_MEM_SIZE=%llu bytes\n", ulparam);
		clGetDeviceInfo(device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(ulparam), &ulparam, NULL);
		printf("CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE=%llu\n", ulparam);
		clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(ulparam), &ulparam, NULL);
		printf("CL_DEVICE_GLOBAL_MEM_SIZE=%llu\n", ulparam);
		clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(ulparam), &ulparam, NULL);
		printf("CL_DEVICE_GLOBAL_MEM_CACHE_SIZE=%llu\n", ulparam);
		clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(uiparam), &uiparam, NULL);
		printf("CL_DEVICE_MAX_COMPUTE_UNITS=%d\n", uiparam);
		clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(param[0]), param, NULL);
		printf("CL_DEVICE_MAX_WORK_GROUP_SIZE=%u\n", param[0]);
		clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(uiparam), &uiparam, NULL);
		printf("CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS=%d\n", uiparam);
		n = uiparam;
		clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(param[0])*n, param, NULL);
		printf("CL_DEVICE_MAX_WORK_ITEM_SIZES=");
		for(i = 0; i < n; i++)
			printf("%u ", (int)param[i]);
		printf("\n");
	}
	return 0;
}

cl_mem OpenCL_Buffer(const void *data, size_t len)
{
	cl_int err;
	cl_mem buf = clCreateBuffer(ctx, CL_MEM_READ_ONLY, len, NULL, &err);
	if(err)
		THError("clCreateBuffer(%d) failed with err = %d\n", len, err);
	err = clEnqueueWriteBuffer(cl_queue, buf, CL_TRUE, 0, len, data, 0, NULL, NULL);
	if(err)
		THError("cEnqueueWriteBuffer(%d) failed with err = %d\n", len, err);
	return buf;
}

THFloatTensor *THOpenCLTensor_newFromFloatTensor(THFloatTensor *t)
{
	THFloatTensor *n = malloc(sizeof(*n));
	memcpy(n, t, sizeof(*n));
	if(t->storage)
	{
		n->storage = malloc(sizeof(*n->storage));
		n->storage->nref = 1;
		n->storage->mustfree = 3;
		n->storageOffset = 0;
		n->storage->data = (float *)OpenCL_Buffer(THFloatTensor_data(t), THFloatTensor_nElement(t) * cl_datasize);
	}
	return n;
}

THFloatTensor *THFloatTensor_newFromOpenCLTensor(THFloatTensor *t)
{
	THFloatTensor *n = malloc(sizeof(*n));
	memcpy(n, t, sizeof(*n));
	if(t->storage)
	{
		n->storage = malloc(sizeof(*n->storage));
		n->storage->nref = 1;
		n->storage->mustfree = 1;
		n->storageOffset = 0;
		int len = THFloatTensor_nElement(t) * cl_datasize;
		n->storage->data = malloc(len);
		if(!n->storage->data)
			THError("Error allocating %d bytes", len);
		cl_int err = clEnqueueReadBuffer(cl_queue, (cl_mem)THFloatTensor_data(t), CL_TRUE, 0, len, n->storage->data, 0, NULL, NULL );
		if(err)
			THError("cEnqueueReadBuffer(%d) failed with err = %d\n", len, err);
	}
	return n;
}

THFloatTensor *THOpenCLTensor_newFromImageTensor(THFloatTensor *t)
{
	if(!cl_order)
		return THOpenCLTensor_newFromFloatTensor(t);
	int x, y, c;
	long *srcstride = t->stride;
	long *srcsize = t->size;
	float *src = THFloatTensor_data(t);
	if(t->nDimension != 3 && t->size[0] != 1)
		THError("Batches are not supported in OpenCL");
	if(t->nDimension == 4)
	{
		srcstride++;
		srcsize++;
	}
	THFloatTensor *tmp = THFloatTensor_new();
	THFloatTensor_resize3d(tmp, srcsize[1], srcsize[2], srcsize[0]);
	float *dst = THFloatTensor_data(tmp);
	for(y = 0; y < tmp->size[0]; y++)
		for(x = 0; x < tmp->size[1]; x++)
			for(c = 0; c < tmp->size[2]; c++)
				dst[y * tmp->stride[0] + x * tmp->stride[1] + c] =
					src[y * srcstride[1] + x * srcstride[2] + c * srcstride[0]];
	THFloatTensor *out = THOpenCLTensor_newFromFloatTensor(tmp);
	THFloatTensor_free(tmp);
	return out;
}

THFloatTensor *THFloatTensor_newFromOpenCLImageTensor(THFloatTensor *t)
{
	if(!cl_order)
		return THFloatTensor_newFromOpenCLTensor(t);
	int x, y, c;
	if(t->nDimension != 3)
		THError("Only 3D tensors are supported for OpenCL images");
	THFloatTensor *tmp = THFloatTensor_newFromOpenCLTensor(t);
	THFloatTensor *out = THFloatTensor_new();
	THFloatTensor_resize3d(out, tmp->size[2], tmp->size[0], tmp->size[1]);
	float *src = THFloatTensor_data(tmp);
	float *dst = THFloatTensor_data(out);
	for(c = 0; c < out->size[0]; c++)
		for(y = 0; y < out->size[1]; y++)
			for(x = 0; x < out->size[2]; x++)
				dst[c * out->stride[0] + y * out->stride[1] + x] =
					src[y * tmp->stride[0] + x * tmp->stride[1] + c];
	THFloatTensor_free(tmp);
	return out;
}

THFloatTensor *THOpenCLTensor_newFromWeightTensor(THFloatTensor *t, int nInputPlanes, int kW, int kH)
{
	if(!cl_order)
		return THOpenCLTensor_newFromFloatTensor(t);
	int x, y, c, o;
	float *src = THFloatTensor_data(t);
	int nOutputPlanes = t->size[0];
	THFloatTensor *tmp = THFloatTensor_new();
	THFloatTensor_resize4d(tmp, kH, kW, nInputPlanes, nOutputPlanes);
	float *dst = THFloatTensor_data(tmp);
	for(y = 0; y < kH; y++)
		for(x = 0; x < kW; x++)
			for(c = 0; c < nInputPlanes; c++)
				for(o = 0; o < nOutputPlanes; o++)
					dst[y * tmp->stride[0] + x * tmp->stride[1] + c * tmp->stride[2] + o] =
						src[y * kW + x + c * kW * kH + o * t->stride[0]];
	THFloatTensor *out = THOpenCLTensor_newFromFloatTensor(tmp);
	THFloatTensor_free(tmp);
	return out;
}

#ifdef HAVEFP16

void THHalfFloatTensor_resize4d(THFloatTensor *t, long size0, long size1, long size2, long size3)
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
	if(!t->storage)
		t->storage = THFloatStorage_new(size0 * size1 * size2 * size3);
	else if(nElement != size0 * size1 * size2 * size3)
		t->storage->data = realloc(t->storage->data, 2 * size0 * size1 * size2 * size3);
}

void THHalfFloatTensor_resize3d(THFloatTensor *t, long size0, long size1, long size2)
{
	long nElement = THFloatTensor_nElement(t);
	t->nDimension = 3;
	t->size[0] = size0;
	t->size[1] = size1;
	t->size[2] = size2;
	t->stride[2] = 1;
	t->stride[1] = size2;
	t->stride[0] = size1 * size2;
	if(!t->storage)
		t->storage = THFloatStorage_new(size0 * size1 * size2);
	else if(nElement != size0 * size1 * size2)
		t->storage->data = realloc(t->storage->data, 2 * size0 * size1 * size2);
}

THFloatTensor *THHalfOpenCLTensor_newFromFloatTensor(THFloatTensor *t)
{
	THFloatTensor *n = malloc(sizeof(*n));
	memcpy(n, t, sizeof(*n));
	if(t->storage)
	{
		n->storage = malloc(sizeof(*n->storage));
		n->storage->nref = 1;
		n->storage->mustfree = 3;
		int nelem = THFloatTensor_nElement(t);
		__fp16 *tmp = malloc(nelem * 2);
		if(!tmp)
			THError("Error allocating %d bytes", nelem * 2);
		tofp16(tmp, THFloatTensor_data(t), nelem);
		n->storage->data = (float *)OpenCL_Buffer(tmp, nelem * 2);
		free(tmp);
	}
	return n;
}

THFloatTensor *THFloatTensor_newFromHalfOpenCLTensor(THFloatTensor *t)
{
	THFloatTensor *n = malloc(sizeof(*n));
	memcpy(n, t, sizeof(*n));
	if(t->storage)
	{
		n->storage = malloc(sizeof(*n->storage));
		n->storage->nref = 1;
		n->storage->mustfree = 1;
		int nelem = THFloatTensor_nElement(t);
		__fp16 *tmp = malloc(2 * nelem);
		if(!tmp)
			THError("Error allocating %d bytes", nelem * 2);
		cl_int err = clEnqueueReadBuffer(cl_queue, (cl_mem)THFloatTensor_data(t), CL_TRUE, 0, 2 * nelem, tmp, 0, NULL, NULL );
		if(err)
			THError("cEnqueueReadBuffer(%d) failed with err = %d\n", 2 * nelem, err);
		n->storage->data = malloc(4 * nelem);
		if(!n->storage->data)
			THError("Error allocating %d bytes", 4 * nelem);
		fromfp16(n->storage->data, tmp, nelem);
		free(tmp);
	}
	return n;
}

THFloatTensor *THHalfOpenCLTensor_newFromImageTensor(THFloatTensor *t)
{
	int x, y, c;
	long *srcstride = t->stride;
	long *srcsize = t->size;
	float *src = THFloatTensor_data(t);
	if(t->nDimension != 3 && t->size[0] != 1)
		THError("Batches are not supported in OpenCL");
	if(t->nDimension == 4)
	{
		srcstride++;
		srcsize++;
	}
	THFloatTensor *tmp = THFloatTensor_new();
	if(!cl_order)
	{
		THHalfFloatTensor_resize3d(tmp, srcsize[0], srcsize[1], srcsize[2]);
		__fp16 *dst = (__fp16 *)THFloatTensor_data(tmp);
		tofp16(dst, src, THFloatTensor_nElement(t));
	} else {
		THHalfFloatTensor_resize3d(tmp, srcsize[1], srcsize[2], srcsize[0]);
		__fp16 *dst = (__fp16 *)THFloatTensor_data(tmp);
		for(y = 0; y < tmp->size[0]; y++)
			for(x = 0; x < tmp->size[1]; x++)
				for(c = 0; c < tmp->size[2]; c++)
					dst[y * tmp->stride[0] + x * tmp->stride[1] + c] =
						src[y * srcstride[1] + x * srcstride[2] + c * srcstride[0]];
	}
	THFloatTensor *out = THOpenCLTensor_newFromFloatTensor(tmp);
	THFloatTensor_free(tmp);
	return out;
}

THFloatTensor *THFloatTensor_newFromHalfOpenCLImageTensor(THFloatTensor *t)
{
	int x, y, c;
	if(t->nDimension != 3)
		THError("Only 3D tensors are supported for OpenCL images");
	THFloatTensor *tmp = THFloatTensor_newFromOpenCLTensor(t);
	__fp16 *src = (__fp16 *)THFloatTensor_data(tmp);
	THFloatTensor *out = THFloatTensor_new();
	if(!cl_order)
	{
		THFloatTensor_resize3d(out, tmp->size[0], tmp->size[1], tmp->size[2]);
		float *dst = THFloatTensor_data(out);
		fromfp16(dst, src, THFloatTensor_nElement(t));
	} else {
		THFloatTensor_resize3d(out, tmp->size[2], tmp->size[0], tmp->size[1]);
		float *dst = THFloatTensor_data(out);
		for(c = 0; c < out->size[0]; c++)
			for(y = 0; y < out->size[1]; y++)
				for(x = 0; x < out->size[2]; x++)
					dst[c * out->stride[0] + y * out->stride[1] + x] =
						src[y * tmp->stride[0] + x * tmp->stride[1] + c];
	}
	THFloatTensor_free(tmp);
	return out;
}

THFloatTensor *THHalfOpenCLTensor_newFromWeightTensor(THFloatTensor *t, int nInputPlanes, int kW, int kH)
{
	int x, y, c, o;
	float *src = THFloatTensor_data(t);
	int nOutputPlanes = t->size[0];
	THFloatTensor *tmp = THFloatTensor_new();
	if(!cl_order)
	{
		THHalfFloatTensor_resize4d(tmp, nOutputPlanes, nInputPlanes, kH, kW);
		__fp16 *dst = (__fp16 *)THFloatTensor_data(tmp);
		tofp16(dst, src, THFloatTensor_nElement(t));
	} else {
		THHalfFloatTensor_resize4d(tmp, kH, kW, nInputPlanes, nOutputPlanes);
		__fp16 *dst = (__fp16 *)THFloatTensor_data(tmp);
		for(y = 0; y < kH; y++)
			for(x = 0; x < kW; x++)
				for(c = 0; c < nInputPlanes; c++)
					for(o = 0; o < nOutputPlanes; o++)
						dst[y * tmp->stride[0] + x * tmp->stride[1] + c * tmp->stride[2] + o] =
							src[y * kW + x + c * kW * kH + o * t->stride[0]];
	}
	THFloatTensor *out = THOpenCLTensor_newFromFloatTensor(tmp);
	THFloatTensor_free(tmp);
	return out;
}

#endif

THFloatStorage *THOpenCLStorage_new(long size)
{
	THFloatStorage *s = malloc(sizeof(*s));
	cl_int err;
	s->data = (float *)clCreateBuffer(ctx, CL_MEM_READ_ONLY, size * cl_datasize, NULL, &err);
	if(err)
		THError("clCreateBuffer(%d) failed with err=%d\n", size * cl_datasize, err);
	s->nref = 1;
	s->mustfree = 3;
	return s;
}

void THOpenCLTensor_resize3d(THFloatTensor *t, long size0, long size1, long size2)
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
		t->storage->mustfree = 3;
		int len = THFloatTensor_nElement(t) * cl_datasize;
		cl_int err;
		t->storage->data = (float *)clCreateBuffer(ctx, CL_MEM_READ_ONLY, len, NULL, &err);
		if(err)
			THError("clCreateBuffer(%d) failed with err=%d\n", len, err);
	} else if(nelemorig != THFloatTensor_nElement(t))
		THError("Resizing of CUDA tensors not supported");
}

void THOpenCLTensor_resizeAs(THFloatTensor *tdst, THFloatTensor *tsrc)
{
	if(tsrc == tdst)
		return;
	long nelemsrc = THFloatTensor_nElement(tsrc);
	tdst->nDimension = tsrc->nDimension;
	memcpy(tdst->size, tsrc->size, sizeof(tsrc->size));
	memcpy(tdst->stride, tsrc->stride, sizeof(tsrc->stride));
	if(!tdst->storage)
		tdst->storage = THOpenCLStorage_new(nelemsrc);
	else if(nelemsrc != THFloatTensor_nElement(tdst))
		THError("Resizing of OpenCL tensors not supported");
}

void OpenCL_GetTensorSizes(THFloatTensor *t, int *nplanes, int *W, int *H)
{
	if(t->nDimension != 3)
		THError("Only 3D tensors supported");
	if(cl_order)
	{
		*nplanes = t->size[2];
		*W = t->size[1];
		*H = t->size[0];
	} else {
		*nplanes = t->size[0];
		*W = t->size[2];
		*H = t->size[1];
	}
}

static char *build_src;
static const char *build_kernelname;

void OpenCL_AddSource(char *src, const char *kernelname)
{
	build_src = src;
	build_kernelname = kernelname;
}

cl_program OpenCL_BuildProgram(const char *src)
{
	cl_int err;
	cl_program program = clCreateProgramWithSource(ctx, 1, (const char **)&src, NULL, &err);
	if(!program)
	{
		THError("Error creating OpenCL program, err=%d\n", err);
		return 0;
	}
	err = clBuildProgram(program, 0, 0, 0, 0, 0);
	if(err != CL_SUCCESS)
	{
		char buffer[20000];
		size_t len;
		
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
		THError("OpenCL Program build failed:\n%s\n", buffer);
		return 0;
	}
	return program;
}

void OpenCL_Build(struct network *net, THFloatTensor *in)
{
	int i;
	char *src = malloc(100000);
	cl_program program;
	
#ifdef HAVEFP16
	if(cl_datasize == 2)
		strcpy(src, "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n#define THInf 1e4\n");
	else
#endif
		strcpy(src, "#define THInf 1e38\n");
	strcpy(src, "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n#define THInf 1e4\n");
	for(i = 0; i < net->nelem; i++)
	{
		build_src = 0;
		in = net->modules[i].updateOutput(&net->modules[i], in);
		if(build_src)
		{
			char s[100];
			sprintf(s, "%s%d", build_kernelname, i+1);
			subst(build_src, build_kernelname, s);
			net->modules[i].kernel = (cl_kernel)strdup(s);
			strcat(src, build_src);
			free(build_src);
		}
		if(i > 0)
		{
			THFloatTensor_free(net->modules[i-1].output);
			net->modules[i-1].output = THFloatTensor_new();
		}
	}
	program  = OpenCL_BuildProgram(src);
	for(i = 0; i < net->nelem; i++)
	{
		if(net->modules[i].kernel)
		{
			cl_int err;
			char *kernelname = (char *)net->modules[i].kernel;
			net->modules[i].kernel = clCreateKernel(program, kernelname, &err);
			if(!net->modules[i].kernel || err != CL_SUCCESS)
				THError("Error creating OpenCL kernel %s, err = %d\n", kernelname, err);
			free(kernelname);
		}
	}
	free(src);
	net->engine = ENGINE_OPENCLINIT;
}

struct network *THOpenCL_ToOpenCL(struct network *net)
{
	int i;
	struct network *nn = malloc(sizeof(*nn));

	nn->nelem = net->nelem;
	nn->modules = malloc(sizeof(net->modules[0]) * net->nelem);
	nn->engine = ENGINE_OPENCL;
	memcpy(nn->modules, net->modules, sizeof(net->modules[0]) * net->nelem);
	for(i = 0; i < net->nelem; i++)
	{
		nn->modules[i].output = THOpenCLTensor_newFromFloatTensor(net->modules[i].output);
		nn->modules[i].net = nn;
		switch(net->modules[i].type)
		{
		case MT_SpatialConvolutionMM:
		case MT_SpatialConvolution:
		case MT_SpatialConvolutionVirtMM:
			nn->modules[i].updateOutput = OpenCL_SpatialConvolution_updateOutput;
#ifdef HAVEFP16
			if(cl_datasize == 2)
			{
				nn->modules[i].SpatialConvolution.weight = THHalfOpenCLTensor_newFromWeightTensor(net->modules[i].SpatialConvolution.weight,
					nn->modules[i].SpatialConvolution.nInputPlane, nn->modules[i].SpatialConvolution.kH, nn->modules[i].SpatialConvolution.kW);
				nn->modules[i].SpatialConvolution.bias = THHalfOpenCLTensor_newFromFloatTensor(net->modules[i].SpatialConvolution.bias);
			} else
#endif
			{
				nn->modules[i].SpatialConvolution.weight = THOpenCLTensor_newFromWeightTensor(net->modules[i].SpatialConvolution.weight,
					nn->modules[i].SpatialConvolution.nInputPlane, nn->modules[i].SpatialConvolution.kH, nn->modules[i].SpatialConvolution.kW);
				nn->modules[i].SpatialConvolution.bias = THOpenCLTensor_newFromFloatTensor(net->modules[i].SpatialConvolution.bias);
			}
			nn->modules[i].SpatialConvolution.finput = 0;
			break;
		case MT_SpatialMaxPooling:
			nn->modules[i].SpatialMaxPooling.indices = 0;
			nn->modules[i].updateOutput = OpenCL_SpatialMaxPooling_updateOutput;
			break;
		case MT_SpatialMaxUnpooling:
			THError("MT_SpatialMaxUnpooling not supported in OpenCL");
			break;
		case MT_Threshold:
			nn->modules[i].updateOutput = OpenCL_Threshold_updateOutput;
			break;
		case MT_SoftMax:
			nn->modules[i].updateOutput = OpenCL_SoftMax_updateOutput;
			break;
		case MT_Dropout:
			if(!nn->modules[i].Dropout.v2)
				THError("Non v2 dropout not supported in OpenCL");
			break;
		case MT_SpatialZeroPadding:
			THError("SpatialZeroPadding not supported in OpenCL");
			break;
		case MT_Linear:
			nn->modules[i].type = MT_SpatialConvolutionMM;
			nn->modules[i].updateOutput = OpenCL_SpatialConvolution_updateOutput;
			struct SpatialConvolution *c = &nn->modules[i].SpatialConvolution;
			c->finput = 0;
			c->padW = c->padH = 0;
			c->dW = c->dH = 1;
			c->kW = c->kH = 1;
			c->nOutputPlane = c->weight->size[0];
			c->nInputPlane = c->weight->size[1];
#ifdef HAVEFP16
			if(cl_datasize == 2)
			{
				// kH and kW = 1, so it's not necessary to recorder the dimensions and use newFromWeightTensor
				nn->modules[i].SpatialConvolution.weight = THHalfOpenCLTensor_newFromFloatTensor(net->modules[i].SpatialConvolution.weight);
				nn->modules[i].SpatialConvolution.bias = THHalfOpenCLTensor_newFromFloatTensor(net->modules[i].SpatialConvolution.bias);
			} else
#endif
			{
				nn->modules[i].SpatialConvolution.weight = THOpenCLTensor_newFromFloatTensor(net->modules[i].SpatialConvolution.weight);
				nn->modules[i].SpatialConvolution.bias = THOpenCLTensor_newFromFloatTensor(net->modules[i].SpatialConvolution.bias);
			}
			break;
		case MT_SpatialBatchNormalization:
			THError("MT_SpatialBatchNormalization not supported in OpenCL");
			break;
		case MT_SpatialFullConvolution:
			THError("MT_SpatialFullConvolution not supported in OpenCL");
			break;
		case MT_SpatialAveragePooling:
			THError("MT_SpatialAveragePooling not supported in OpenCL");
			break;
		case MT_Sequential:
			THError("MT_Sequential not supported in OpenCL");
			break;
		case MT_Concat:
			THError("MT_Concat not supported in OpenCL");
			break;
		}
	}
	return nn;
}

void subst(char *buf, const char *from, const char *to)
{
	char *p = buf;
	int fromlen = strlen(from);
	int tolen = strlen(to);
	
	while( (p = strstr(p, from)) )
	{
		if(isalnum(p[-1]) || isalnum(p[fromlen]))
		{
			p++;
			continue;
		}
		memmove(p + tolen, p + fromlen, strlen(p + fromlen) + 1);
		memcpy(p, to, tolen);
	}
}

void substi(char *buf, const char *from, int to)
{
	char sto[10];
	sprintf(sto, "%d", to);
	subst(buf, from, sto);
}

void substf(char *buf, const char *from, float to)
{
	char sto[20];
	sprintf(sto, "%f", to);
	subst(buf, from, sto);
}

char *strdup_more(const char *src)
{
	char *dst = malloc(strlen(src) + 200);
	strcpy(dst, src);
	return dst;
}
