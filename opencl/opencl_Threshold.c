#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../thnets.h"

static const char *source_inplace = 
"__kernel void thr(__global float *data)\n"
"{\n"
"	int i = get_global_id(0);\n"
"	if(data[i] <= threshold)\n"
"		data[i] = val;\n"
"}\n";

static const char *source_copy = 
"__kernel void thr(__global const float *in, __global float *out)\n"
"{\n"
"	int i = get_global_id(0);\n"
"	if(in[i] <= threshold)\n"
"		out[i] = val;\n"
"	else out[i] = in[i];\n"
"}\n";

THFloatTensor *OpenCL_Threshold_updateOutput(struct module *module, THFloatTensor *input)
{
	float val = module->Threshold.val;
	float threshold = module->Threshold.threshold;
	THFloatTensor *output = module->output;
	int inPlace = module->Threshold.inplace == 1;
	size_t local[1], global[1];

	if (inPlace)
	{
		THFloatTensor_set(output, input);
		if(!module->kernel)
		{
			char *src = strdup_more(source_inplace);
#ifdef HAVEFP16
			if(cl_datasize == 2)
				subst(src, "float", "half");
#endif
			substf(src, "threshold", threshold);
			substf(src, "val", val);
			OpenCL_AddSource(src, "thr");
			return output;
		}
		cl_mem buf = (cl_mem)THFloatTensor_data(input);
		clSetKernelArg(module->kernel, 0, sizeof(cl_mem), &buf);
		local[0] = 4;
		global[0] = THFloatTensor_nElement(input);
		cl_int err = clEnqueueNDRangeKernel(cl_queue, module->kernel, 1, NULL, global, local, 0, NULL, NULL);
		if(err)
			THError("clEnqueueNDRangeKernel for SpatialConvolution failed with err=%d\n", err);
		return output;
	} else {
		THOpenCLTensor_resizeAs(output, input);
		if(!module->kernel)
		{
			char *src = strdup_more(source_copy);
#ifdef HAVEFP16
			if(cl_datasize == 2)
				subst(src, "float", "half");
#endif
			substf(src, "threshold", threshold);
			substf(src, "val", val);
			OpenCL_AddSource(src, "thr");
			return output;
		}
		cl_mem bufin = (cl_mem)THFloatTensor_data(input);
		cl_mem bufout = (cl_mem)THFloatTensor_data(output);
		clSetKernelArg(module->kernel, 0, sizeof(cl_mem), &bufin);
		clSetKernelArg(module->kernel, 1, sizeof(cl_mem), &bufout);
		global[0] = THFloatTensor_nElement(input);
		if(global[0] % 4 == 0)
			local[0] = 4;
		else if(global[0] % 2 == 0)
			local[0] = 2;
		else local[0] = 1;
		cl_int err = clEnqueueNDRangeKernel(cl_queue, module->kernel, 1, NULL, global, local, 0, NULL, NULL);
		if(err)
			THError("clEnqueueNDRangeKernel for SpatialConvolution failed with err=%d\n", err);
		return output;
	}
}
