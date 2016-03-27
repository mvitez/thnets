#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../thnets.h"

static const char *source = 
"#define BYTE2FLOAT 0.003921568f // 1/255\n"
"__kernel void rgb2float_rgb(__global float *dst, __global const unsigned char *src, __global const float *mean, __global const float *std, int width, int srcstride)\n"
"{\n"
"	int c;\n"
"	int x = get_global_id(0);\n"
"	int y = get_global_id(1);\n"
"	for(c = 0; c < 3; c++)\n"
"	{\n"
"		dst[(x + y * width) * 3 + c] =\n"
"			(src[c + 3*x + srcstride*y] * BYTE2FLOAT - mean[c]) / std[c];\n"
"	}\n"
"}\n"
"__kernel void rgb2float_bgr(__global float *dst, __global const unsigned char *src, __global const float *mean, __global const float *std, int width, int srcstride)\n"
"{\n"
"	int c;\n"
"	int x = get_global_id(0);\n"
"	int y = get_global_id(1);\n"
"	for(c = 0; c < 3; c++)\n"
"	{\n"
"		dst[(x + y * width) * 3 + c] =\n"
"			(src[2 - c + 3*x + srcstride*y] * BYTE2FLOAT - mean[c]) / std[c];\n"
"	}\n"
"}\n";

THFloatTensor *OpenCL_LoadImage(const unsigned char *src, int width, int height, int srcstride, const float *mean, const float *std, int bgr)
{
	static cl_kernel kernel_rgb, kernel_bgr;
	THFloatTensor *out = THFloatTensor_new();
	THOpenCLTensor_resize3d(out, height, width, 3);
	if(!kernel_rgb)
	{
		cl_program program = OpenCL_BuildProgram(source);
		cl_int err;
		kernel_rgb = clCreateKernel(program, "rgb2float_rgb", &err);
		if(!kernel_rgb || err != CL_SUCCESS)
			THError("Error creating OpenCL kernel rgb2float_rgb, err = %d\n", err);
		kernel_bgr = clCreateKernel(program, "rgb2float_bgr", &err);
		if(!kernel_bgr || err != CL_SUCCESS)
			THError("Error creating OpenCL kernel rgb2float_bgr, err = %d\n", err);
	}
	cl_mem bufout = (cl_mem)THFloatTensor_data(out);
	cl_mem bufin = OpenCL_Buffer(src, srcstride * height);
	cl_mem bufmean = OpenCL_Buffer(mean, 12);
	cl_mem bufstd = OpenCL_Buffer(std, 12);
	cl_kernel kernel = bgr ? kernel_bgr : kernel_rgb;
	
	clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufout);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufin);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufmean);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), &bufstd);
	clSetKernelArg(kernel, 4, sizeof(cl_mem), &width);
	clSetKernelArg(kernel, 5, sizeof(cl_mem), &srcstride);
	size_t global[2], local[2];
	global[0] = width;
	global[1] = height;
	local[0] = global[0] % 2 == 0 ? 2 : 1;
	local[1] = global[1] % 2 == 0 ? 2 : 1;
	cl_int err = clEnqueueNDRangeKernel(cl_queue, kernel, 2, NULL, global, local, 0, NULL, NULL);
	if(err)
		THError("clEnqueueNDRangeKernel for SpatialConvolution failed with err=%d\n", err);
	clReleaseMemObject(bufin);
	clReleaseMemObject(bufmean);
	clReleaseMemObject(bufstd);
	return out;
}
