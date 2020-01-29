#include "../thnets.h"

#ifdef ONNX
void onnxload_Upsample(const void *graph, struct module *m, int nodeidx)
{
	m->updateOutput = nn_Upsample_updateOutput;
	m->type = MT_Upsample;
	struct Upsample *p = &m->Upsample;
	p->width_scale = onnx_getfloat(graph, nodeidx, "width_scale", -1);
	p->height_scale = onnx_getfloat(graph, nodeidx, "height_scale", -1);
	if(p->width_scale == 0 && p->height_scale == 0)
	{
		p->height_scale = onnx_getfloat(graph, nodeidx, "scales", 2);
		p->width_scale = onnx_getfloat(graph, nodeidx, "scales", 3);
	}
}
#endif

THNTensor *nn_Upsample_updateOutput(struct module *module, THNTensor *input)
{
	if(module->Upsample.height_scale != 2 || module->Upsample.width_scale != 2)
		THError("Only 2x upsampling is supported\n");
	int x, y, w, h, nb, np, b, p;
	int str0, str1, str2, str3;
	if(input->nDimension == 4)
	{
		THNTensor_resize4d(module->output, input->size[0], input->size[1], input->size[2] * 2, input->size[3] * 2);
		w = input->size[3];
		h = input->size[2];
		np = input->size[1];
		nb = input->size[0];
		str0 = input->stride[0];
		str1 = input->stride[1];
		str2 = input->stride[2];
		str3 = input->stride[3];
	} else if(input->nDimension == 3)
	{
		THNTensor_resize3d(module->output, input->size[0], input->size[1] * 2, input->size[2] * 2);
		w = input->size[2];
		h = input->size[1];
		np = input->size[0];
		nb = 1;
		str0 = 0;
		str1 = input->stride[0];
		str2 = input->stride[1];
		str3 = input->stride[2];
	} else {
		THError("Only 3D and 4D input is supported\n");
		return 0; // Just to avoid warnings, THError exits in any case
	}
	float *in = THNTensor_data(input);
	float *out = THNTensor_data(module->output);
	for(b = 0; b < nb; b++)
		for(p = 0; p < np; p++)
		{
			float *in1 = in + b * str0 + p * str1;
			float *out1 = out + b * w*h*4*np + p * w*h*4;
			for(y = 0; y < h; y++)
				for(x = 0; x < w; x++)
					out1[4*y*w + 2*x] = out1[4*y*w + 2*x + 1] = out1[4*y*w + 2*w + 2*x] = out1[4*y*w + 2*w + 2*x + 1] = in1[y*str2 + x*str3];
		}
	return module->output;
}
