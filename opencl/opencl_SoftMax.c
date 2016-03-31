#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../thnets.h"

static const char *source_o0 =
"__kernel void softmax(__global const float *in, __global float *out)\n"
"{\n"
"	int x = get_global_id(0);\n"
"	int y = get_global_id(1);\n"
"	int i;\n"
"	float max = -THInf, sum = 0;\n"
"	in += x + y * iW;\n"
"	out += x + y * iW;\n"
"	for(i = 0; i < nplanes; i++)\n"
"		if(in[i*planestride] > max)\n"
"			max = in[i*planestride];\n"
"	for(i = 0; i < nplanes; i++)\n"
"	{\n"
"		out[i] = exp(in[i*planestride] - max);\n"
"		sum += out[i*planestride];\n"
"	}\n"
"	for(i = 0; i < nplanes; i++)\n"
"		out[i*planestride] *= 1 / sum;\n"
"}\n";
static const char *source_o1 =
"__kernel void softmax(__global const float *in, __global float *out)\n"
"{\n"
"	int x = get_global_id(0);\n"
"	int y = get_global_id(1);\n"
"	int i;\n"
"	float max = -THInf, sum = 0;\n"
"	in += (x + y * iW) * nplanes;\n"
"	out += (x + y * iW) * nplanes;\n"
"	for(i = 0; i < nplanes; i++)\n"
"		if(in[i] > max)\n"
"			max = in[i];\n"
"	for(i = 0; i < nplanes; i++)\n"
"	{\n"
"		out[i] = exp(in[i] - max);\n"
"		sum += out[i];\n"
"	}\n"
"	for(i = 0; i < nplanes; i++)\n"
"		out[i] *= 1 / sum;\n"
"}\n";

THFloatTensor *OpenCL_SoftMax_updateOutput(struct module *module, THFloatTensor *input)
{
	int nplanes, iW, iH;

	OpenCL_GetTensorSizes(input, &nplanes, &iW, &iH);
	THFloatTensor *output = module->output;
	THOpenCLTensor_resizeAs(output, input);
	if(!module->kernel)
	{
		char *src = strdup_more(cl_order ? source_o1 : source_o0);
#ifdef HAVEFP16
		if(cl_datasize == 2)
			subst(src, "float", "half");
#endif
		substi(src, "iH", iH);
		substi(src, "iW", iW);
		substi(src, "nplanes", nplanes);
		if(!cl_order)
			substi(src, "planestride", iH * iW);
		OpenCL_AddSource(src, "softmax");
		return output;
	}
	size_t local[2], global[2];

	cl_mem bufin = (cl_mem)THFloatTensor_data(input);
	cl_mem bufout = (cl_mem)THFloatTensor_data(output);
	clSetKernelArg(module->kernel, 0, sizeof(cl_mem), &bufin);
	clSetKernelArg(module->kernel, 1, sizeof(cl_mem), &bufout);
	global[0] = iW;
	global[1] = iH;
	local[0] = global[0] % 2 == 0 ? 2 : 1;
	local[1] = global[1] % 2 == 0 ? 2 : 1;
	cl_int err = clEnqueueNDRangeKernel(cl_queue, module->kernel, 2, NULL, global, local, 0, NULL, NULL);
	if(err)
		THError("clEnqueueNDRangeKernel for SoftMax failed with err=%d\n", err);
	return output;
}
