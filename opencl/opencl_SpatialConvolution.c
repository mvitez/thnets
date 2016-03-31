#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../thnets.h"

static const char *source_o0a =
"__kernel void conv(__global const FLOAT *in, __global const FLOAT *filt, __global FLOAT *bias, __global FLOAT *out, int xoffs, int yoffs)\n"
"{\n"
"	int c, i, j, k, i1, inidx, inidx2, x, y, ix, iy;\n"
"	FLOAT sum;\n"
"	__global const FLOAT *filt2;\n"
"\n"
"	x = xoffs + get_global_id(0);\n"
"	y = yoffs + get_global_id(1);\n"
"	for(k = 0; k < nOutputPlanes; k++) {\n"
"		sum = bias[k];\n"
"		filt2 = filt + k * nInputPlanes * kW * kH;\n"
"		for(c = 0; c < nInputPlanes; c++)\n"
"		{\n"
"			for(i = 0; i < kH; i++)\n"
"			{\n"
"				iy = dH * y - padH + i;\n"
"				if(iy >= 0 && iy < iH)\n"
"				{\n"
"					i1 = kW*i + c * kW * kH;\n"
"					inidx = iW*iy + c * iW * iH;\n"
"					#pragma unroll\n"
"					for(j = 0; j < kW; j++)\n"
"					{\n"
"						ix = dW * x - padW + j;\n"
"						if(ix >= 0 && ix < iW)\n"
"							sum += in[inidx + ix] * filt2[i1 + j];\n"
"					}\n"
"				}\n"
"			}\n"
"		}\n"
"		out[(oW*y+x) + k * oW*oH] = sum;\n"
"	}\n"
"}\n";

static const char *source_o0b =
"__kernel void conv(__global const FLOAT *in, __global const FLOAT *filt, __global FLOAT *bias, __global FLOAT *out, int xoffs, int yoffs)\n"
"{\n"
"	int c, i, j, k, i1, inidx, inidx2, x, y, ix, iy;\n"
"	FLOAT sum;\n"
"	__global const FLOAT *filt2;\n"
"\n"
"	x = xoffs + get_global_id(0);\n"
"	y = yoffs + get_global_id(1);\n"
"	for(k = 0; k < nOutputPlanes; k++) {\n"
"		sum = bias[k];\n"
"		filt2 = filt + k * nInputPlanes * kW * kH;\n"
"		for(i = 0; i < kH; i++)\n"
"		{\n"
"			iy = dH * y - padH + i;\n"
"			if(iy >= 0 && iy < iH)\n"
"			{\n"
"				for(j = 0; j < kW; j++)\n"
"				{\n"
"					ix = dW * x - padW + j;\n"
"					if(ix >= 0 && ix < iW)\n"
"					{\n"
"						i1 = kW*i + j;\n"
"						inidx = iW*iy + ix;\n"
"						#pragma unroll\n"
"						for(c = 0; c < nInputPlanes; c++)\n"
"							sum += in[inidx + c * iW*iH] * filt2[i1 + c * kW*kH];\n"
"					}\n"
"				}\n"
"			}\n"
"		}\n"
"		out[(oW*y+x) + k * oW*oH] = sum;\n"
"	}\n"
"}\n";

static const char *source_o1a =
"__kernel void conv(__global const FLOAT *in, __global const FLOAT4 *filt, __global FLOAT4 *bias, __global FLOAT4 *out, int xoffs, int yoffs)\n"
"{\n"
"	int c, i, j, k, i1, inidx, inidx2, x, y, ix, iy;\n"
"	FLOAT4 sum;\n"
"\n"
"	x = xoffs + get_global_id(0);\n"
"	y = yoffs + get_global_id(1);\n"
"	for(k = 0; k < nOutputPlanes; k++) {\n"
"		sum = bias[k];\n"
"		for(i = 0; i < kH; i++) {\n"
"			iy = dH * y - padH + i;\n"
"			if(iy >= 0 && iy < iH) {\n"
"				i1 = kW*i*nInputPlanes;\n"
"				inidx = iW*iy*nInputPlanes;\n"
"				for(j = 0; j < kW; j++) {\n"
"					ix = dW * x - padW + j;\n"
"					if(ix >= 0 && ix < iW) {\n"
"						inidx2 = inidx + nInputPlanes * ix;\n"
"						#pragma unroll\n"
"						for(c = 0; c < nInputPlanes; c++)\n"
"							sum += in[inidx2 + c] * filt[(i1 + c) * nOutputPlanes + k];\n"
"					}\n"
"					i1 += nInputPlanes;\n"
"				}\n"
"			}\n"
"		}\n"
"		out[(oW*y+x)*nOutputPlanes+k] = sum;\n"
"	}\n"
"}\n";

static const char *source_o1b =
"__kernel void conv(__global const FLOAT *in, __global const FLOAT4 *filt, __global FLOAT4 *bias, __global FLOAT4 *out, int xoffs, int yoffs)\n"
"{\n"
"	int c, i, j, k, i1, inidx, inidx2, x, y, ix, iy;\n"
"	FLOAT4 sum;\n"
"\n"
"	x = xoffs + get_global_id(0);\n"
"	y = yoffs + get_global_id(1);\n"
"	k = get_global_id(2);\n"
"	sum = bias[k];\n"
"	for(i = 0; i < kH; i++) {\n"
"		iy = dH * y - padH + i;\n"
"		if(iy >= 0 && iy < iH) {\n"
"			i1 = kW*i*nInputPlanes;\n"
"			inidx = iW*iy*nInputPlanes;\n"
"			for(j = 0; j < kW; j++) {\n"
"				ix = dW * x - padW + j;\n"
"				if(ix >= 0 && ix < iW) {\n"
"					inidx2 = inidx + nInputPlanes * ix;\n"
"					#pragma unroll\n"
"					for(c = 0; c < nInputPlanes; c++)\n"
"						sum += in[inidx2 + c] * filt[(i1 + c) * nOutputPlanes + k];\n"
"				}\n"
"				i1 += nInputPlanes;\n"
"			}\n"
"		}\n"
"	}\n"
"	out[(oW*y+x)*nOutputPlanes+k] = sum;\n"
"}\n";

THFloatTensor *OpenCL_SpatialConvolution_updateOutput(struct module *module, THFloatTensor *input)
{
	const int alg = 1;
	int iplanes, iW, iH, oplanes;
	OpenCL_GetTensorSizes(input, &iplanes, &iW, &iH);
	int oH = (iH + 2*module->SpatialConvolution.padH - module->SpatialConvolution.kH) / module->SpatialConvolution.dH + 1;
	int oW = (iW + 2*module->SpatialConvolution.padW - module->SpatialConvolution.kW) / module->SpatialConvolution.dW + 1;

	oplanes = cl_order && module->SpatialConvolution.nOutputPlane % 4 == 0 ?
		module->SpatialConvolution.nOutputPlane / 4 : module->SpatialConvolution.nOutputPlane;
	
	if(cl_order)
		THOpenCLTensor_resize3d(module->output, oH, oW, module->SpatialConvolution.nOutputPlane);
	else THOpenCLTensor_resize3d(module->output, module->SpatialConvolution.nOutputPlane, oH, oW);
	if(!module->kernel)
	{
		char *src = strdup_more(cl_order ? (alg ? source_o1b : source_o1a) : (alg ? source_o0b : source_o0a));
		substi(src, "kW", module->SpatialConvolution.kW);
		substi(src, "kH", module->SpatialConvolution.kH);
		substi(src, "nInputPlanes", module->SpatialConvolution.nInputPlane);
		substi(src, "nOutputPlanes", oplanes);
		if(cl_order && module->SpatialConvolution.nOutputPlane % 4 > 0)
		{
			subst(src, "float4", "float");
			subst(src, "FLOAT4", "FLOAT");
		}
#ifdef HAVEFP16
		if(cl_datasize == 2)
		{
			subst(src, "FLOAT", "half");
			subst(src, "FLOAT4", "half4");
		} else
#endif
		{
			subst(src, "FLOAT", "float");
			subst(src, "FLOAT4", "float4");
		}
		substi(src, "iW", iW);
		substi(src, "iH", iH);
		substi(src, "dW", module->SpatialConvolution.dW);
		substi(src, "dH", module->SpatialConvolution.dH);
		substi(src, "oW", oW);
		substi(src, "oH", oH);
		substi(src, "padW", module->SpatialConvolution.padW);
		substi(src, "padH", module->SpatialConvolution.padH);
		OpenCL_AddSource(src, "conv");
		return module->output;
	}
	cl_mem bufin = (cl_mem)THFloatTensor_data(input);
	cl_mem bufweight = (cl_mem)THFloatTensor_data(module->SpatialConvolution.weight);
	cl_mem bufbias = (cl_mem)THFloatTensor_data(module->SpatialConvolution.bias);
	cl_mem bufout = (cl_mem)THFloatTensor_data(module->output);

	int x, y, step = iW > 1000 ? 64 : 8;
	for(y = 0; y < oH; y += step)
		for(x = 0; x < oW; x += step)
		{
			clSetKernelArg(module->kernel, 0, sizeof(cl_mem), &bufin);
			clSetKernelArg(module->kernel, 1, sizeof(cl_mem), &bufweight);
			clSetKernelArg(module->kernel, 2, sizeof(cl_mem), &bufbias);
			clSetKernelArg(module->kernel, 3, sizeof(cl_mem), &bufout);
			clSetKernelArg(module->kernel, 4, sizeof(cl_mem), &x);
			clSetKernelArg(module->kernel, 5, sizeof(cl_mem), &y);
			size_t local[3], global[3];
			int oW1 = oW - x > step ? step : oW - x;
			int oH1 = oH - y > step ? step : oH - y;
			local[0] = oW1 % 4 == 0 ? 4 : oW1 % 2 == 0 ? 2 : 1;
			local[1] = oH1 % 4 == 0 ? 4 : oH1 % 2 == 0 ? 2 : 1;
			local[2] = oplanes % 4 == 0 ? 4 : oplanes % 2 == 0 ? 2 : 1;
			global[0] = oW1;
			global[1] = oH1;
			global[2] = oplanes;
			cl_int err = clEnqueueNDRangeKernel(cl_queue, module->kernel, alg && cl_order ? 3 : 2, NULL, global, local, 0, NULL, NULL);
			if(err)
				THError("clEnqueueNDRangeKernel for SpatialConvolution failed with err=%d\n", err);
		}
	return module->output;
}
