#include <math.h>
#include <string.h>
#include "../thnets.h"

int nnload_SpatialAveragePooling(struct module *mod, struct nnmodule *n)
{
	struct table *t = n->table;
	mod->type = MT_SpatialAveragePooling;
	mod->updateOutput = nn_SpatialAveragePooling_updateOutput;
	struct SpatialAveragePooling *m = &mod->SpatialAveragePooling;
	m->padW = TableGetNumber(t, "padW");
	m->padH = TableGetNumber(t, "padH");
	m->dW = TableGetNumber(t, "dW");
	m->dH = TableGetNumber(t, "dH");
	m->kW = TableGetNumber(t, "kW");
	m->kH = TableGetNumber(t, "kH");
	m->ceil_mode = TableGetNumber(t, "ceil_mode");
	m->count_include_pad= TableGetNumber(t, "count_include_pad");
	return 0;
}

#ifdef ONNX
void onnxload_SpatialAveragePooling(const void *graph, struct module *m, int nodeidx)
{
	int naxes = onnx_getint(graph, nodeidx, "axes", -2);
	if( !(naxes == 0 || (naxes == 2 && onnx_getint(graph, nodeidx, "axes", 0) == 2 && onnx_getint(graph, nodeidx, "axes", 1) == 3)))
		THError("ReduceMean along channel is not supported\n");
	m->updateOutput = nn_SpatialAveragePooling_updateOutput;
	m->type = MT_SpatialAveragePooling;
	struct SpatialAveragePooling *p = &m->SpatialAveragePooling;
	const char *autopad = onnx_getstring(graph, nodeidx, "auto_pad", -1);
	if(autopad && !strcmp(autopad, "SAME_UPPER"))
		p->autopad = 1;
	else if(autopad && !strcmp(autopad, "SAME_LOWER"))
		p->autopad = 2;
	else p->autopad = 0;
	p->ceil_mode = 0;
	if(onnx_getint(graph, nodeidx, "kernel_shape", -2) == 3)
	{
		p->kZ = onnx_getint(graph, nodeidx, "kernel_shape", 0);
		p->kH = onnx_getint(graph, nodeidx, "kernel_shape", 1);
		p->kW = onnx_getint(graph, nodeidx, "kernel_shape", 2);
		p->padZ = onnx_getint(graph, nodeidx, "pads", 0);
		p->padH = onnx_getint(graph, nodeidx, "pads", 1);
		p->padW = onnx_getint(graph, nodeidx, "pads", 2);
		p->padZ2 = onnx_getint(graph, nodeidx, "pads", 3);
		p->padH2 = onnx_getint(graph, nodeidx, "pads", 4);
		p->padW2 = onnx_getint(graph, nodeidx, "pads", 5);
		p->dZ = onnx_getint(graph, nodeidx, "strides", 0);
		p->dH = onnx_getint(graph, nodeidx, "strides", 1);
		p->dW = onnx_getint(graph, nodeidx, "strides", 2);
	} else {
		p->kZ = 1;
		p->kH = onnx_getint(graph, nodeidx, "kernel_shape", 0);
		p->kW = onnx_getint(graph, nodeidx, "kernel_shape", 1);
		p->padZ = 0;
		p->padH = onnx_getint(graph, nodeidx, "pads", 0);
		p->padW = onnx_getint(graph, nodeidx, "pads", 1);
		p->padZ2 = 0;
		p->padH2 = onnx_getint(graph, nodeidx, "pads", 2);
		p->padW2 = onnx_getint(graph, nodeidx, "pads", 3);
		p->dZ = 1;
		p->dH = onnx_getint(graph, nodeidx, "strides", 0);
		p->dW = onnx_getint(graph, nodeidx, "strides", 1);
	}
	if(p->dZ == 0)
		p->dZ = 1;
	if(p->dH == 0)
		p->dH = 1;
	if(p->dW == 0)
		p->dW = 1;
}
#endif

THNTensor *nn_SpatialAveragePooling_updateOutput(struct module *module, THNTensor *input)
{
	int kW = module->SpatialAveragePooling.kW;
	int kH = module->SpatialAveragePooling.kH;
	int dW = module->SpatialAveragePooling.dW;
	int dH = module->SpatialAveragePooling.dH;
	int padW = module->SpatialAveragePooling.padW;
	int padH = module->SpatialAveragePooling.padH;
	int ceil_mode = module->SpatialAveragePooling.ceil_mode;
	int count_include_pad = module->SpatialAveragePooling.count_include_pad;
	THNTensor *output = module->output;

	float *output_data;
	float *input_data;

	int dimw = 2;
	int dimh = 1;
	int dimc = 0;
	long nbatch = 1;

	long inputWidth;
	long inputHeight;
	long outputWidth;
	long outputHeight;
	long nInputPlane; // number of channels (or colors)

	long k;

	if(! (input->nDimension == 3 || input->nDimension == 4) )
		THError("3D or 4D (batch mode) tensor expected");
	if(! (kW/2 >= padW && kH/2 >= padH) )
		THError("pad should be smaller than half of kernel size");

	if (input->nDimension == 4)
	{
		nbatch = input->size[0];
		dimw++;
		dimh++;
		dimc++;
	}

	inputWidth = input->size[dimw];
	inputHeight = input->size[dimh];
	nInputPlane = input->size[dimc];

	if(kW == 0)
		kW = inputWidth;
	if(kH == 0)
		kH = inputHeight;
	if(ceil_mode)
	{
		outputWidth  = (long)(ceil((float)(inputWidth  - kW + 2*padW) / dW)) + 1;
		outputHeight = (long)(ceil((float)(inputHeight - kH + 2*padH) / dH)) + 1;
	}
	else
	{
		outputWidth  = (long)(floor((float)(inputWidth  - kW + 2*padW) / dW)) + 1;
		outputHeight = (long)(floor((float)(inputHeight - kH + 2*padH) / dH)) + 1;
	}
	if (padW || padH)
	{
	// ensure that the last pooling starts inside the image
	// needed to avoid problems in ceil mode
	if ((outputHeight - 1)*dH >= inputHeight + padH)
		--outputHeight;
	if ((outputWidth  - 1)*dW >= inputWidth  + padW)
		--outputWidth;
	}
	
	if( !(inputWidth >= kW - 2 * padW && inputHeight >= kH - 2 * padH) )
		THError("input image smaller than kernel size");

	if (input->nDimension == 3)
		THNTensor_resize3d(output, nInputPlane, outputHeight, outputWidth);
	else
		THNTensor_resize4d(output, input->size[0], nInputPlane, outputHeight, outputWidth);

	THNTensor *input2 = THNTensor_new();
	THNTensor_resizeAs(input2, input);
	THNTensor_copy(input2, input);
	input = input2;
	input_data = THNTensor_data(input);
	output_data = THNTensor_data(output);
  
#pragma omp parallel for private(k)
	for(k = 0; k < nInputPlane; k++)
	{
		long p;
		for(p = 0; p < nbatch; p++)
		{
			long xx, yy;
			/* For all output pixels... */
			float *ptr_output = output_data + p*nInputPlane*outputWidth*outputHeight + k*outputWidth*outputHeight;
			float *ptr_input = input_data + p*nInputPlane*inputWidth*inputHeight + k*inputWidth*inputHeight;
			long i;
			for(i = 0; i < outputWidth*outputHeight; i++)
				ptr_output[i] = 0;

			for(yy = 0; yy < outputHeight; yy++)
			{
				for(xx = 0; xx < outputWidth; xx++)
				{
					/* Compute the mean of the input image... */
					long hstart = yy * dH - padH;
					long wstart = xx * dW - padW;
					long hend = fminf(hstart + kH, inputHeight + padH);
					long wend = fminf(wstart + kW, inputWidth + padW);
					long pool_size = (hend - hstart) * (wend - wstart);
					hstart = fmaxf(hstart, 0);
					wstart = fmaxf(wstart, 0);
					hend = fminf(hend, inputHeight);
					wend = fminf(wend, inputWidth);

					float sum = 0;

					long divide_factor;
					if(count_include_pad)
						divide_factor = pool_size;
					else
						divide_factor = (hend - hstart) * (wend - wstart);

					long kx, ky;

					for(ky = hstart; ky < hend; ky++)
					{
						for(kx = wstart; kx < wend; kx++)
							sum += ptr_input[ky*inputWidth + kx];
					}
					/* Update output */
					*ptr_output++ += sum/divide_factor;
				}
			}
		}
	}
	THNTensor_free(input);
	return output;
}
