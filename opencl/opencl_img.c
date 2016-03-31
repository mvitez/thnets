#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../thnets.h"

static const char *source = 
"#define BYTE2FLOAT 0.003921568f // 1/255\n"
"__kernel void rgb2float_o1(__global float *dst, __global const unsigned char *src, __global const float *mean, __global const float *std, int srcstride, int width)\n"
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
"__kernel void bgr2float_o1(__global float *dst, __global const unsigned char *src, __global const float *mean, __global const float *std, int srcstride, int width)\n"
"{\n"
"	int c;\n"
"	int x = get_global_id(0);\n"
"	int y = get_global_id(1);\n"
"	for(c = 0; c < 3; c++)\n"
"	{\n"
"		dst[(x + y * width) * 3 + c] =\n"
"			(src[2 - c + 3*x + srcstride*y] * BYTE2FLOAT - mean[c]) / std[c];\n"
"	}\n"
"}\n"
"__kernel void rgb2float_o0(__global float *dst, __global const unsigned char *src, __global const float *mean, __global const float *std, int srcstride, int width, int height)\n"
"{\n"
"	int c;\n"
"	int x = get_global_id(0);\n"
"	int y = get_global_id(1);\n"
"	for(c = 0; c < 3; c++)\n"
"	{\n"
"		dst[x + y * width + c * width*height] =\n"
"			(src[c + 3*x + srcstride*y] * BYTE2FLOAT - mean[c]) / std[c];\n"
"	}\n"
"}\n"
"__kernel void bgr2float_o0(__global float *dst, __global const unsigned char *src, __global const float *mean, __global const float *std, int srcstride, int width, int height)\n"
"{\n"
"	int c;\n"
"	int x = get_global_id(0);\n"
"	int y = get_global_id(1);\n"
"	for(c = 0; c < 3; c++)\n"
"	{\n"
"		dst[x + y * width + c * width*height] =\n"
"			(src[2 - c + 3*x + srcstride*y] * BYTE2FLOAT - mean[c]) / std[c];\n"
"	}\n"
"}\n";

THFloatTensor *OpenCL_LoadImage(const unsigned char *src, int width, int height, int srcstride, const float *mean, const float *std, int bgr)
{
	static cl_kernel kernel_rgb_o1, kernel_bgr_o1, kernel_rgb_o0, kernel_bgr_o0;
	THFloatTensor *out = THFloatTensor_new();
	if(cl_order)
		THOpenCLTensor_resize3d(out, height, width, 3);
	else THOpenCLTensor_resize3d(out, 3, height, width);
	if(!kernel_rgb_o0)
	{
		cl_program program;
#ifdef HAVEFP16
		if(cl_datasize == 2)
		{
			char *source2 = malloc(strlen(source) + 100);
			strcpy(source2, "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n");
			strcat(source2, source);
			subst(source2, "float", "half");
			program = OpenCL_BuildProgram(source2);
		} else
#endif
			program = OpenCL_BuildProgram(source);
		cl_int err;
		kernel_rgb_o1 = clCreateKernel(program, "rgb2float_o1", &err);
		if(!kernel_rgb_o1 || err != CL_SUCCESS)
			THError("Error creating OpenCL kernel rgb2float_o1, err = %d\n", err);
		kernel_bgr_o1 = clCreateKernel(program, "bgr2float_o1", &err);
		if(!kernel_bgr_o1 || err != CL_SUCCESS)
			THError("Error creating OpenCL kernel bgr2float_o1, err = %d\n", err);
		kernel_rgb_o0 = clCreateKernel(program, "rgb2float_o0", &err);
		if(!kernel_rgb_o0 || err != CL_SUCCESS)
			THError("Error creating OpenCL kernel rgb2float_o0, err = %d\n", err);
		kernel_bgr_o0 = clCreateKernel(program, "bgr2float_o0", &err);
		if(!kernel_bgr_o0 || err != CL_SUCCESS)
			THError("Error creating OpenCL kernel bgr2float_o0, err = %d\n", err);
	}
	cl_mem bufout = (cl_mem)THFloatTensor_data(out);
	cl_mem bufin = OpenCL_Buffer(src, srcstride * height);
	cl_mem bufmean, bufstd;
#ifdef HAVEFP16
	if(cl_datasize == 2)
	{
		__fp16 mean2[3], std2[3];
		mean2[0] = mean[0];
		mean2[1] = mean[1];
		mean2[2] = mean[2];
		std2[0] = std[0];
		std2[1] = std[1];
		std2[2] = std[2];
		bufmean = OpenCL_Buffer(mean2, 6);
		bufstd = OpenCL_Buffer(std2, 6);
	} else
#endif
	{
		bufmean = OpenCL_Buffer(mean, 12);
		bufstd = OpenCL_Buffer(std, 12);
	}
	cl_kernel kernel = cl_order ? (bgr ? kernel_bgr_o1 : kernel_rgb_o1) :  (bgr ? kernel_bgr_o0 : kernel_rgb_o0);
	
	clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufout);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufin);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufmean);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), &bufstd);
	clSetKernelArg(kernel, 4, sizeof(cl_mem), &srcstride);
	clSetKernelArg(kernel, 5, sizeof(cl_mem), &width);
	if(!cl_order)
		clSetKernelArg(kernel, 6, sizeof(cl_mem), &height);
	size_t global[2], local[2];
	global[0] = width;
	global[1] = height;
	local[0] = global[0] % 2 == 0 ? 2 : 1;
	local[1] = global[1] % 2 == 0 ? 2 : 1;
	cl_int err = clEnqueueNDRangeKernel(cl_queue, kernel, 2, NULL, global, local, 0, NULL, NULL);
	if(err)
		THError("clEnqueueNDRangeKernel for rgb2float failed with err=%d\n", err);
	clReleaseMemObject(bufin);
	clReleaseMemObject(bufmean);
	clReleaseMemObject(bufstd);
	return out;
}
