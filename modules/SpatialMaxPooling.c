#include <math.h>
#include "../thnets.h"

static void nnfree_SpatialMaxPooling(struct module *mod)
{
	THFloatTensor_free(mod->SpatialMaxPooling.indices);
}

int nnload_SpatialMaxPooling(struct module *mod, struct nnmodule *n)
{
	struct table *t = n->table;
	mod->type = MT_SpatialMaxPooling;
	mod->updateOutput = nn_SpatialMaxPooling_updateOutput;
	mod->nnfree = nnfree_SpatialMaxPooling;
	struct SpatialMaxPooling *m = &mod->SpatialMaxPooling;
	m->padW = TableGetNumber(t, "padW");
	m->padH = TableGetNumber(t, "padH");
	m->dW = TableGetNumber(t, "dW");
	m->dH = TableGetNumber(t, "dH");
	m->kW = TableGetNumber(t, "kW");
	m->kH = TableGetNumber(t, "kH");
	m->ceil_mode = TableGetNumber(t, "ceil_mode");
	m->indices = THFloatTensor_new();
	return 0;
}

static void nn_SpatialMaxPooling_updateOutput_frame(float *input_p, float *output_p, float *ind_p,
	long nslices,
	long iwidth, long iheight,
	long owidth, long oheight,
	int kW, int kH, int dW, int dH,
	int padW, int padH)
{
	long k;
#pragma omp parallel for private(k)
	for (k = 0; k < nslices; k++) {
		float *ip = input_p + k*iwidth*iheight;
		float *op = output_p + k*owidth*oheight;
		float *indp = ind_p + k*owidth*oheight;

		long i, j;
		for (i = 0; i < oheight; i++) {
			for (j = 0; j < owidth; j++) {

				long hstart = i * dH - padH;
				long wstart = j * dW - padW;
				long hend = thfminf(hstart + kH, iheight);
				long wend = thfminf(wstart + kW, iwidth);
				hstart = thfmaxf(hstart, 0);
				wstart = thfmaxf(wstart, 0);

				long maxindex = -1;
				float maxval = -THInf;

				long x, y;
				for (y = hstart; y < hend; y++) {
					for (x = wstart; x < wend; x++) {
						float val = *(ip + y*iwidth + x);

						if (val > maxval)
						{
							maxval = val;
							maxindex = y*iwidth + x;
						}
					}
				}
				*(op + i*owidth + j) = maxval;
				*(indp + i*owidth + j) = maxindex + 1;
			}
		}
	}
}

THFloatTensor *nn_SpatialMaxPooling_updateOutput(struct module *module, THFloatTensor *input)
{
	int kW = module->SpatialMaxPooling.kW;
	int kH = module->SpatialMaxPooling.kH;
	int dW = module->SpatialMaxPooling.dW;
	int dH = module->SpatialMaxPooling.dH;
	int padW = module->SpatialMaxPooling.padW;
	int padH = module->SpatialMaxPooling.padH;
	int ceil_mode = module->SpatialMaxPooling.ceil_mode;
	THFloatTensor *output = module->output;
	THFloatTensor *indices = module->SpatialMaxPooling.indices;

	int batch = 1;
	if (input->nDimension == 3) {
		batch = 0;
		THFloatTensor_resize4d(input, 1, input->size[0], input->size[1], input->size[2]);
	}

	long batchSize = input->size[0];
	long nslices = input->size[1];
	long iheight = input->size[2];
	long iwidth = input->size[3];
	module->SpatialMaxPooling.iwidth = iwidth;
	module->SpatialMaxPooling.iheight = iheight;

	long oheight;
	long owidth;
	if (ceil_mode) {
		oheight = (long)(ceil((float)(iheight - kH + 2*padH) / dH)) + 1;
		owidth  = (long)(ceil((float)(iwidth  - kW + 2*padW) / dW)) + 1;
	} else {
		oheight = (long)(floor((float)(iheight - kH + 2*padH) / dH)) + 1;
		owidth  = (long)(floor((float)(iwidth  - kW + 2*padW) / dW)) + 1;
	}

	if (padW || padH) {
		// ensure that the last pooling starts inside the image
		if ((oheight - 1)*dH >= iheight + padH)
			--oheight;
		if ((owidth  - 1)*dW >= iwidth  + padW)
			--owidth;
	}

	THFloatTensor_resize4d(output, batchSize, nslices, oheight, owidth);
	THFloatTensor_resize4d(indices, batchSize, nslices, oheight, owidth);

	float *input_data = THFloatTensor_data(input);
	float *output_data = THFloatTensor_data(output);
	float *indices_data = THFloatTensor_data(indices);

	long p;
#pragma omp parallel for private(p)
	for (p = 0; p < batchSize; p++) {
		nn_SpatialMaxPooling_updateOutput_frame(input_data+p*nslices*iwidth*iheight,
			output_data+p*nslices*owidth*oheight, indices_data+p*nslices*owidth*oheight,
			nslices, iwidth, iheight, owidth, oheight,
			kW, kH, dW, dH, padW, padH);
	}

	if (batch == 0) {
		THFloatTensor_resize3d(output, nslices, oheight, owidth);
		THFloatTensor_resize3d(indices, nslices, oheight, owidth);
		THFloatTensor_resize3d(input, nslices, iheight, iwidth);
	}

	return output;
}
