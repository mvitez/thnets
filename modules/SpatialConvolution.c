#include "../thnets.h"
#include <string.h>

static void nnfree_SpatialConvolution(struct module *mod)
{
	THNTensor_free(mod->SpatialConvolution.bias);
	THNTensor_free(mod->SpatialConvolution.weight);
	THNTensor_free(mod->SpatialConvolution.finput);
}

int nnload_SpatialConvolution(struct module *mod, struct nnmodule *n)
{
	struct table *t = n->table;
	mod->type = MT_SpatialConvolutionMM;
	mod->updateOutput = nn_SpatialConvolutionMM_updateOutput;
	mod->nnfree = nnfree_SpatialConvolution;
	struct SpatialConvolution *m = &mod->SpatialConvolution;
	m->padW = TableGetNumber(t, "padW");
	m->padH = TableGetNumber(t, "padH");
	if(!m->padW && !m->padH)
		m->padW = m->padH = TableGetNumber(t, "padding");
	m->dW = TableGetNumber(t, "dW");
	m->dH = TableGetNumber(t, "dH");
	m->kW = TableGetNumber(t, "kW");
	m->kH = TableGetNumber(t, "kH");
	m->nInputPlane = TableGetNumber(t, "nInputPlane");
	m->nOutputPlane = TableGetNumber(t, "nOutputPlane");
	m->bias = TableGetTensor(t, "bias");
	m->weight = TableGetTensor(t, "weight");
	if(m->weight->nDimension == 4)
		THNTensor_resize2d(m->weight, m->weight->size[0], m->weight->size[1] * m->weight->size[2] * m->weight->size[3]);
	m->finput = THNTensor_new();
	return 0;
}

#ifdef ONNX
void onnxload_SpatialConvolution(const void *graph, struct module *m, int nodeidx)
{
	m->updateOutput = nn_SpatialConvolutionMM_updateOutput;
#ifdef USEBLAS
	m->type = MT_SpatialConvolutionMM;
#else
	m->type = MT_SpatialConvolutionVirtMM;
#endif
	m->nnfree = nnfree_SpatialConvolution;
	struct SpatialConvolution *p = &m->SpatialConvolution;
	p->refl_pad = 0;
	p->weight = onnx_gettensor(graph, nodeidx, 1);
	p->bias = onnx_gettensor(graph, nodeidx, 2);
	p->finput = THNTensor_new();
	p->nOutputPlane = (int)p->weight->size[0];
	p->nInputPlane = (int)p->weight->size[1];
	if(p->weight->nDimension == 5)
	{ //Conv3d
		p->kZ = (int)p->weight->size[2];
		p->kH = (int)p->weight->size[3];
		p->kW = (int)p->weight->size[4];
		p->padZ = onnx_getint(graph, nodeidx, "pads", 0);
		p->padH = onnx_getint(graph, nodeidx, "pads", 1);
		p->padW = onnx_getint(graph, nodeidx, "pads", 2);
		p->padZ2 = onnx_getint(graph, nodeidx, "pads", 3);
		p->padH2 = onnx_getint(graph, nodeidx, "pads", 4);
		p->padW2 = onnx_getint(graph, nodeidx, "pads", 5);
		p->dZ = onnx_getint(graph, nodeidx, "strides", 0);
		p->dH = onnx_getint(graph, nodeidx, "strides", 1);
		p->dW = onnx_getint(graph, nodeidx, "strides", 2);
		p->dlZ = onnx_getint(graph, nodeidx, "dilations", 0);
		p->dlH = onnx_getint(graph, nodeidx, "dilations", 1);
		p->dlW = onnx_getint(graph, nodeidx, "dilations", 2);
		if(p->kZ != onnx_getint(graph, nodeidx, "kernel_shape", 0) ||
				p->kH != onnx_getint(graph, nodeidx, "kernel_shape", 1) ||
				p->kW != onnx_getint(graph, nodeidx, "kernel_shape", 2))
			THError("Conflicting kernel sizes in proto file\n");
	} else { //Conv2d
		p->kZ = 1;
		p->kH = (int)p->weight->size[2];
		p->kW = (int)p->weight->size[3];
		p->padZ = 0;
		p->padH = onnx_getint(graph, nodeidx, "pads", 0);
		p->padW = onnx_getint(graph, nodeidx, "pads", 1);
		p->padZ2 = 0;
		p->padH2 = onnx_getint(graph, nodeidx, "pads", 2);
		p->padW2 = onnx_getint(graph, nodeidx, "pads", 3);
		p->dZ = 1;
		p->dH = onnx_getint(graph, nodeidx, "strides", 0);
		p->dW = onnx_getint(graph, nodeidx, "strides", 1);
		p->dlZ = 1;
		p->dlH = onnx_getint(graph, nodeidx, "dilations", 0);
		p->dlW = onnx_getint(graph, nodeidx, "dilations", 1);
		if(p->kH != onnx_getint(graph, nodeidx, "kernel_shape", 0) ||
				p->kW != onnx_getint(graph, nodeidx, "kernel_shape", 1))
			THError("Conflicting kernel sizes in proto file\n");
	}
	const char *autopad = onnx_getstring(graph, nodeidx, "auto_pad", -1);
	if(autopad && !strcmp(autopad, "SAME_UPPER"))
		p->autopad = 1;
	else if(autopad && !strcmp(autopad, "SAME_LOWER"))
		p->autopad = 2;
	else p->autopad = 0;
	if(p->dW == 0)
		p->dW = 1;
	if(p->dH == 0)
		p->dH = 1;
	if(p->dZ == 0)
		p->dZ = 1;
	if(p->dlW == 0)
		p->dlW = 1;
	if(p->dlH == 0)
		p->dlH = 1;
	if(p->dlZ == 0)
		p->dlZ = 1;
	int g = onnx_getint(graph, nodeidx, "group", -1);
	if(g == p->nOutputPlane && p->nInputPlane == 1 && g > 1)
	{
		m->type = MT_DepthwiseConvolution;
		p->nInputPlane = g;
	} else if(g > 1)
		THError("Group convolution not supported\n");
}
#endif

THNTensor *nn_SpatialConvolution_updateOutput(struct module *module, THNTensor *input)
{
	int dW = module->SpatialConvolution.dW;
	int dH = module->SpatialConvolution.dH;

	THNTensor *weight = module->SpatialConvolution.weight;
	THNTensor *bias = module->SpatialConvolution.bias;
	THNTensor *output = module->output;

	int dimw = 2;
	int dimh = 1;

	if (input->nDimension == 4)
	{
		dimw++;
		dimh++;
	}

	long nOutputPlane = weight->size[0];
	long kW           = weight->size[3];
	long kH           = weight->size[2];
	long inputWidth   = input->size[dimw];
	long inputHeight  = input->size[dimh];
	long outputWidth  = (inputWidth - kW) / dW + 1;
	long outputHeight = (inputHeight - kH) / dH + 1;

	if (input->nDimension == 3)
	{
		long i;
		float *bias_data;
		float *output_data;

		THNTensor_resize3d(output, nOutputPlane, outputHeight, outputWidth);
		/* add bias */
		bias_data = THNTensor_data(bias);
		output_data = THNTensor_data(output);

#pragma omp parallel for private(i)
		for (i=0; i<bias->size[0]; i++)
		{
			float *ptr_output = output_data + i*outputWidth*outputHeight;
			long j;
			for(j = 0; j < outputWidth*outputHeight; j++)
				ptr_output[j] = bias_data[i];
		}
		THNTensor_conv2Dmv(output, 1.0, 1.0, input, weight, dH, dW, "V","X");
	}
	else
	{
		float *bias_data;
		float *output_data;
		long p;

		THNTensor_resize4d(output, input->size[0], nOutputPlane, outputHeight, outputWidth);

		bias_data = THNTensor_data(bias);
		output_data = THNTensor_data(output);

#pragma omp parallel for private(p)
		for (p=0; p<input->size[0]; p++)
		{
			/* BIAS */
			long i;
			for (i=0; i<bias->size[0]; i++)
			{
				float *ptr_output = output_data + p*nOutputPlane*outputWidth*outputHeight + i*outputWidth*outputHeight;
				long j;
				for(j = 0; j < outputWidth*outputHeight; j++)
					ptr_output[j] = bias_data[i];
			}
		}

		/* do convolutions */
		THNTensor_conv2Dmm(output, 1.0, 1.0, input, weight, dH, dW, "V","X");
	}
	return output;
}
