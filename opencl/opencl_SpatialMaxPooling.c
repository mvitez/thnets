#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../thnets.h"

static const char *source =
"__kernel void maxpooling(__global const float *in, __global float *out)\n"
"{\n"
"	int x = get_global_id(0);\n"
"	int y = get_global_id(1);\n"
"	int slice = get_global_id(2);\n"
"	float maxval = -1e38f;\n"
"	int i, j;\n"
"	int x1 = dW * x - padW;\n"
"	if(x1 < 0)\n"
"		x1 = 0;\n"
"	int x2 = dW * x - padW + kW;\n"
"	if(x2 > iW)\n"
"		x2 = iW;\n"
"	int y1 = dH * y - padH;\n"
"	if(y1 < 0)\n"
"		y1 = 0;\n"
"	int y2 = dH * y - padH + kH;\n"
"	if(y2 > iH)\n"
"		y2 = iH;\n"
"	for(i = y1; i < y2; i++)\n"
"	{\n"
"		int offs = (i * iW) * nslices + slice;\n"
"		for(j = x1; j < x2; j++)\n"
"			if(in[offs + j * nslices] > maxval)\n"
"				maxval = in[offs + j * nslices];\n"
"	}\n"
"	out[(x + y * oW) * nslices + slice] = maxval;\n"
"}\n";

THFloatTensor *OpenCL_SpatialMaxPooling_updateOutput(struct module *module, THFloatTensor *input)
{
	int kW = module->SpatialMaxPooling.kW;
	int kH = module->SpatialMaxPooling.kH;
	int dW = module->SpatialMaxPooling.dW;
	int dH = module->SpatialMaxPooling.dH;
	int padW = module->SpatialMaxPooling.padW;
	int padH = module->SpatialMaxPooling.padH;
	int iH = input->size[0];
	int iW = input->size[1];
	int nslices = input->size[2];
	int oW, oH;
	int ceil_mode = module->SpatialMaxPooling.ceil_mode;
	THFloatTensor *output = module->output;

	if(ceil_mode)
	{
		oH = ceil((float)(iH - kH + 2*padH) / dH) + 1;
		oW  = ceil((float)(iW  - kW + 2*padW) / dW) + 1;
	} else {
		oH = floor((float)(iH - kH + 2*padH) / dH) + 1;
		oW  = floor((float)(iW  - kW + 2*padW) / dW) + 1;
	}
	if (padW || padH)
	{
		// ensure that the last pooling starts inside the image
		if ((oH - 1)*dH >= iH + padH)
			--oH;
		if ((oW  - 1)*dW >= iW  + padW)
			--oW;
	}
	THOpenCLTensor_resize3d(output, oH, oW, nslices);

	if(!module->kernel)
	{
		char *src = strdup_more(source);
		substi(src, "kW", kW);
		substi(src, "kH", kH);
		substi(src, "iW", iW);
		substi(src, "iH", iH);
		substi(src, "nslices", nslices);
		substi(src, "dW", dW);
		substi(src, "dH", dH);
		substi(src, "oW", oW);
		substi(src, "oH", oH);
		substi(src, "padW", padW);
		substi(src, "padH", padH);
		OpenCL_AddSource(src, "maxpooling");
		return module->output;
	}
	cl_mem bufin = (cl_mem)THFloatTensor_data(input);
	cl_mem bufout = (cl_mem)THFloatTensor_data(output);
	clSetKernelArg(module->kernel, 0, sizeof(cl_mem), &bufin);
	clSetKernelArg(module->kernel, 1, sizeof(cl_mem), &bufout);
	size_t local[3], global[3];
	local[0] = oW % 2 == 0 ? 2 : 1;
	local[1] = oH % 2 == 0 ? 2 : 1;
	local[2] = nslices % 2 == 0 ? 2 : 1;
	global[0] = oW;
	global[1] = oH;
	global[2] = nslices;
	cl_int err = clEnqueueNDRangeKernel(cl_queue, module->kernel, 3, NULL, global, local, 0, NULL, NULL);
	if(err)
		THError("clEnqueueNDRangeKernel for SpatialMaxPooling failed with err=%d\n", err);
	return module->output;
}
