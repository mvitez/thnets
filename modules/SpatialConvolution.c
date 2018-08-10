#include "../thnets.h"
#include <string.h>

static void nnfree_SpatialConvolution(struct module *mod)
{
	THFloatTensor_free(mod->SpatialConvolution.bias);
	THFloatTensor_free(mod->SpatialConvolution.weight);
	THFloatTensor_free(mod->SpatialConvolution.finput);
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
		THFloatTensor_resize2d(m->weight, m->weight->size[0], m->weight->size[1] * m->weight->size[2] * m->weight->size[3]);
	m->finput = THFloatTensor_new();
	return 0;
}

void pyload_SpatialConvolution(struct pyfunction *f)
{
	f->module.updateOutput = nn_SpatialConvolutionMM_updateOutput;
#ifdef USEBLAS
	f->module.type = MT_SpatialConvolutionMM;
#else
	f->module.type = MT_SpatialConvolutionVirtMM;
#endif
	f->module.nnfree = nnfree_SpatialConvolution;
	struct SpatialConvolution *p = &f->module.SpatialConvolution;
	struct pyelement *el;
	p->weight = pygettensor(f->params, "", 0);
	p->bias = pygettensor(f->params, "", 1);
	p->finput = THFloatTensor_new();
	p->nOutputPlane = (int)p->weight->size[0];
	p->nInputPlane = (int)p->weight->size[1];
	p->kH = (int)p->weight->size[2];
	p->kW = (int)p->weight->size[3];
	if( (el = findelement(f->params, "padding", 0)) && el->type == ELTYPE_INTVECT)
	{
		p->padH = el->ivect[0];
		p->padW = el->ivect[1];
	}
	if( (el = findelement(f->params, "stride", 0)) && el->type == ELTYPE_INTVECT)
	{
		p->dH = el->ivect[0];
		p->dW = el->ivect[1];
	}
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
	p->finput = THFloatTensor_new();
	p->nOutputPlane = (int)p->weight->size[0];
	p->nInputPlane = (int)p->weight->size[1];
	p->kH = (int)p->weight->size[2];
	p->kW = (int)p->weight->size[3];
	if(p->kH != onnx_getint(graph, nodeidx, "kernel_shape", 0) ||
			p->kW != onnx_getint(graph, nodeidx, "kernel_shape", 1))
		THError("Conflicting kernel sizes in proto file\n");
	const char *autopad = onnx_getstring(graph, nodeidx, "auto_pad", -1);
	if(autopad && !strcmp(autopad, "SAME_UPPER"))
		p->autopad = 1;
	else if(autopad && !strcmp(autopad, "SAME_LOWER"))
		p->autopad = 2;
	else p->autopad = 0;
	p->padH = onnx_getint(graph, nodeidx, "pads", 0);
	p->padW = onnx_getint(graph, nodeidx, "pads", 1);
	p->padH2 = onnx_getint(graph, nodeidx, "pads", 2);
	p->padW2 = onnx_getint(graph, nodeidx, "pads", 3);
	p->dH = onnx_getint(graph, nodeidx, "strides", 0);
	p->dW = onnx_getint(graph, nodeidx, "strides", 1);
	if(p->dW == 0)
		p->dW = 1;
	if(p->kW == 0)
		p->kW = 1;
	if(onnx_getint(graph, nodeidx, "dilations", 0) > 1 || onnx_getint(graph, nodeidx, "dilations", 1) > 1)
		THError("Dilation not supported\n");
	if(onnx_getint(graph, nodeidx, "group", -1) > 1)
		THError("Group convolution not supported\n");
}
#endif

THFloatTensor *nn_SpatialConvolution_updateOutput(struct module *module, THFloatTensor *input)
{
	int dW = module->SpatialConvolution.dW;
	int dH = module->SpatialConvolution.dH;

	THFloatTensor *weight = module->SpatialConvolution.weight;
	THFloatTensor *bias = module->SpatialConvolution.bias;
	THFloatTensor *output = module->output;

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

		THFloatTensor_resize3d(output, nOutputPlane, outputHeight, outputWidth);
		/* add bias */
		bias_data = THFloatTensor_data(bias);
		output_data = THFloatTensor_data(output);

#pragma omp parallel for private(i)
		for (i=0; i<bias->size[0]; i++)
		{
			float *ptr_output = output_data + i*outputWidth*outputHeight;
			long j;
			for(j = 0; j < outputWidth*outputHeight; j++)
				ptr_output[j] = bias_data[i];
		}
		THFloatTensor_conv2Dmv(output, 1.0, 1.0, input, weight, dH, dW, "V","X");
	}
	else
	{
		float *bias_data;
		float *output_data; 
		long p;

		THFloatTensor_resize4d(output, input->size[0], nOutputPlane, outputHeight, outputWidth);

		bias_data = THFloatTensor_data(bias);
		output_data = THFloatTensor_data(output);

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
		THFloatTensor_conv2Dmm(output, 1.0, 1.0, input, weight, dH, dW, "V","X");
	}
	return output;
}
