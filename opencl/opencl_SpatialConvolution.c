#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../thnets.h"

static const char *source =
"__kernel void conv(__global const float *in, __global const float4 *filt, __global float4 *bias, __global float4 *out)\n"
"{\n"
"	int c, i, j, k, i1, inidx, inidx2, x, y, ix, iy;\n"
"	float4 sum;\n"
"\n"
"	x = get_global_id(0);\n"
"	y = get_global_id(1);\n"
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

THFloatTensor *OpenCL_SpatialConvolution_updateOutput(struct module *module, THFloatTensor *input)
{
	int iH = input->size[0];
	int iW = input->size[1];
	int oH = (iH + 2*module->SpatialConvolution.padH - module->SpatialConvolution.kH) / module->SpatialConvolution.dH + 1;
	int oW = (iW + 2*module->SpatialConvolution.padW - module->SpatialConvolution.kW) / module->SpatialConvolution.dW + 1;
	THOpenCLTensor_resize3d(module->output, oH, oW, module->SpatialConvolution.nOutputPlane);
	if(!module->kernel)
	{
		char *src = strdup_more(source);
		substi(src, "kW", module->SpatialConvolution.kW);
		substi(src, "kH", module->SpatialConvolution.kH);
		substi(src, "nInputPlanes", module->SpatialConvolution.nInputPlane);
		if(module->SpatialConvolution.nOutputPlane % 4 == 0)
			substi(src, "nOutputPlanes", module->SpatialConvolution.nOutputPlane / 4);
		else {
			substi(src, "nOutputPlanes", module->SpatialConvolution.nOutputPlane);
			subst(src, "float4", "float");
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
	
	clSetKernelArg(module->kernel, 0, sizeof(cl_mem), &bufin);
	clSetKernelArg(module->kernel, 1, sizeof(cl_mem), &bufweight);
	clSetKernelArg(module->kernel, 2, sizeof(cl_mem), &bufbias);
	clSetKernelArg(module->kernel, 3, sizeof(cl_mem), &bufout);
	size_t local[2], global[2];
	local[0] = oW % 2 == 0 ? 2 : 1;
	local[1] = oH % 2 == 0 ? 2 : 1;
	global[0] = oW;
	global[1] = oH;
	cl_int err = clEnqueueNDRangeKernel(cl_queue, module->kernel, 2, NULL, global, local, 0, NULL, NULL);
	if(err)
		THError("clEnqueueNDRangeKernel for SpatialConvolution failed with err=%d\n", err);
	return module->output;
}
