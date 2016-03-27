#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../thnets.h"

static const char *source =
"__kernel void softmax(__global const float *in, __global float *out)\n"
"{\n"
"	int x = get_global_id(0);\n"
"	int y = get_global_id(1);\n"
"	int i;\n"
"	float max = -1e38f, sum = 0;\n"
"	in += (x + y * iW) * nslices;\n"
"	out += (x + y * iW) * nslices;\n"
"	for(i = 0; i < nslices; i++)\n"
"		if(in[i] > max)\n"
"			max = in[i];\n"
"	for(i = 0; i < nslices; i++)\n"
"	{\n"
"		out[i] = exp(in[i] - max);\n"
"		sum += out[i];\n"
"	}\n"
"	for(i = 0; i < nslices; i++)\n"
"		out[i] *= 1 / sum;\n"
"}\n";

THFloatTensor *OpenCL_SoftMax_updateOutput(struct module *module, THFloatTensor *input)
{
	THFloatTensor *output = module->output;
	THOpenCLTensor_resizeAs(output, input);
	if(!module->kernel)
	{
		char *src = strdup_more(source);
		substi(src, "iH", input->size[0]);
		substi(src, "iW", input->size[1]);
		substi(src, "nslices", input->size[2]);
		OpenCL_AddSource(src, "softmax");
		return output;
	}
	size_t local[2], global[2];

	cl_mem bufin = (cl_mem)THFloatTensor_data(input);
	cl_mem bufout = (cl_mem)THFloatTensor_data(output);
	clSetKernelArg(module->kernel, 0, sizeof(cl_mem), &bufin);
	clSetKernelArg(module->kernel, 1, sizeof(cl_mem), &bufout);
	global[0] = input->size[1];
	global[1] = input->size[0];
	local[0] = global[0] % 2 == 0 ? 2 : 1;
	local[1] = global[1] % 2 == 0 ? 2 : 1;
	cl_int err = clEnqueueNDRangeKernel(cl_queue, module->kernel, 2, NULL, global, local, 0, NULL, NULL);
	if(err)
		THError("clEnqueueNDRangeKernel for SoftMax failed with err=%d\n", err);
	return output;
}
