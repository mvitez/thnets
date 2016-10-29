#include <math.h>
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

THFloatTensor *nn_SpatialAveragePooling_updateOutput(struct module *module, THFloatTensor *input)
{
	int kW = module->SpatialAveragePooling.kW;
	int kH = module->SpatialAveragePooling.kH;
	int dW = module->SpatialAveragePooling.dW;
	int dH = module->SpatialAveragePooling.dH;
	int padW = module->SpatialAveragePooling.padW;
	int padH = module->SpatialAveragePooling.padH;
	int ceil_mode = module->SpatialAveragePooling.ceil_mode;
	int count_include_pad = module->SpatialAveragePooling.count_include_pad;
	THFloatTensor *output = module->output;

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
		THFloatTensor_resize3d(output, nInputPlane, outputHeight, outputWidth);
	else
		THFloatTensor_resize4d(output, input->size[0], nInputPlane, outputHeight, outputWidth);

	THFloatTensor *input2 = THFloatTensor_new();
	THFloatTensor_resizeAs(input2, input);
	THFloatTensor_copy(input2, input);
	input = input2;
	input_data = THFloatTensor_data(input);
	output_data = THFloatTensor_data(output);
  
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
					int pool_size = (hend - hstart) * (wend - wstart);
					hstart = fmaxf(hstart, 0);
					wstart = fmaxf(wstart, 0);
					hend = fminf(hend, inputHeight);
					wend = fminf(wend, inputWidth);

					float sum = 0;

					int divide_factor;
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
	THFloatTensor_free(input);
	return output;
}
