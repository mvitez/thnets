#include <string.h>
#include <math.h>
#include "../thnets.h"

static void nn_unfolded_copy(THFloatTensor *finput, THFloatTensor *input,
	int kW, int kH, int dW, int dH, int padW, int padH,
	int nInputPlane, int inputWidth, int inputHeight,
	int outputWidth, int outputHeight, unsigned char zero)
{
	long k;
	unsigned char *input_data = (unsigned char *)THFloatTensor_data(input);
	unsigned char *finput_data = (unsigned char *)THFloatTensor_data(finput);

#pragma omp parallel for private(k)
	for(k = 0; k < nInputPlane*kH*kW; k++) {
		long nip = k / (kH*kW);
		long rest = k % (kH*kW);
		long kh = rest / kW;
		long kw = rest % kW;
		long x,y;
		long long ix,iy;
		unsigned char *dst = finput_data + nip*(kH*kW*outputHeight*outputWidth) + kh*(kW*outputHeight*outputWidth) + kw*(outputHeight*outputWidth);
		unsigned char *src = input_data + nip*(inputHeight*inputWidth);
		if (padW > 0 || padH > 0) {
			long lpad,rpad;
			for(y = 0; y < outputHeight; y++) {
				iy = (long long)(y*dH - padH + kh);
				if (iy < 0 || iy >= inputHeight) {
					memset(dst+y*outputWidth, zero, outputWidth);
				} else {
					if (dW==1){
						ix = (long long)(0 - padW + kw);
						lpad = thfmaxf(0,padW-kw);
						rpad = thfmaxf(0,padW-(kW-kw-1));
						if (outputWidth-rpad-lpad <= 0) {
							memset(dst+(y*outputWidth), zero, outputWidth);
						} else {
							if (lpad > 0) memset(dst+y*outputWidth, zero, lpad);
							memcpy(dst+(y*outputWidth+lpad), src+(iy*inputWidth+ix+lpad), (outputWidth-rpad-lpad));
							if (rpad > 0) memset(dst+y*outputWidth + outputWidth - rpad, zero, rpad);
						}
					}
					else{
						for (x=0; x<outputWidth; x++){
							ix = (long long)(x*dW - padW + kw);
							if (ix < 0 || ix >= inputWidth)
								memset(dst+(y*outputWidth+x), zero, 1);
							else
								memcpy(dst+(y*outputWidth+x), src+(iy*inputWidth+ix), 1);
						}
					}
				}
			}
		} else {
			for(y = 0; y < outputHeight; y++) {
				iy = (long long)(y*dH + kh);
				ix = (long long)(0 + kw);
				if (dW == 1)
					memcpy(dst+(y*outputWidth), src+(iy*inputWidth+ix), outputWidth);
				else{
					for (x=0; x<outputWidth; x++)
						memcpy(dst+(y*outputWidth+x), src+(iy*inputWidth+ix+x*dW), 1);
				}
			}
		}
	}
	finput->mult = input->mult;
	finput->sub = input->sub;
}

static void lowpadd(unsigned char *dst, int add, int len)
{
	int i;

	for(i = 0; i < len - 7; i += 8)
	{
		dst[i+0] = dst[i+0] + add > 255 ? 255 : dst[i+0] + add < 0 ? 0 : dst[i+0] + add;
		dst[i+1] = dst[i+1] + add > 255 ? 255 : dst[i+1] + add < 0 ? 0 : dst[i+1] + add;
		dst[i+2] = dst[i+2] + add > 255 ? 255 : dst[i+2] + add < 0 ? 0 : dst[i+2] + add;
		dst[i+3] = dst[i+3] + add > 255 ? 255 : dst[i+3] + add < 0 ? 0 : dst[i+3] + add;
		dst[i+4] = dst[i+4] + add > 255 ? 255 : dst[i+4] + add < 0 ? 0 : dst[i+4] + add;
		dst[i+5] = dst[i+5] + add > 255 ? 255 : dst[i+5] + add < 0 ? 0 : dst[i+5] + add;
		dst[i+6] = dst[i+6] + add > 255 ? 255 : dst[i+6] + add < 0 ? 0 : dst[i+6] + add;
		dst[i+7] = dst[i+7] + add > 255 ? 255 : dst[i+7] + add < 0 ? 0 : dst[i+7] + add;
	}
	for(; i < len; i++)
		dst[i] = dst[i] + add > 255 ? 255 : dst[i] + add;
}

static void nn_SpatialConvolutionMM_updateOutput_frame(THFloatTensor *input, THFloatTensor *output,
	THFloatTensor *weight, THFloatTensor *bias,
	THFloatTensor *finput,
	int kW, int kH, int dW, int dH, int padW, int padH,
	long nInputPlane, long inputWidth, long inputHeight,
	long nOutputPlane, long outputWidth, long outputHeight)
{
	THFloatTensor *output2d = 0;

	if(finput)
	{
		int zero = roundf(-input->sub * input->mult);
		nn_unfolded_copy(finput, input, kW, kH, dW, dH, padW, padH,
			nInputPlane, inputWidth, inputHeight, outputWidth, outputHeight, zero);
		output2d = THFloatTensor_newWithStorage2d(output->storage, output->storageOffset,
			nOutputPlane, -1, outputHeight*outputWidth, -1);
	}

	output2d->mult = output->mult;
	output2d->sub = output->sub;
	THLowpTensor_mm(output2d, weight, finput);
	long i;
	float mult = output->mult / bias->mult;
	int add = roundf(bias->sub * output->mult);
	for(i = 0; i < nOutputPlane; i++)
	{
		unsigned char *data = (unsigned char *)output->storage->data + output->storageOffset + output->stride[0]*i;
		int addend = ((unsigned char *)bias->storage->data)[bias->storageOffset + i];
		addend = roundf(addend * mult) + add;
		lowpadd(data, addend, outputHeight*outputWidth);
	}
	THFloatTensor_free(output2d);
}

THFloatTensor *Lowp_SpatialConvolution_updateOutput(struct module *module, THFloatTensor *input)
{
	int kW = module->SpatialConvolution.kW;
	int kH = module->SpatialConvolution.kH;
	int dW = module->SpatialConvolution.dW;
	int dH = module->SpatialConvolution.dH;
	int padW = module->SpatialConvolution.padW;
	int padH = module->SpatialConvolution.padH;

	THFloatTensor *finput = module->SpatialConvolution.finput;
	THFloatTensor *weight = module->SpatialConvolution.weight;
	THFloatTensor *bias   = module->SpatialConvolution.bias;
	THFloatTensor *output = module->output;

	int batch = 1;
	if(input->nDimension == 3)
	{
		batch = 0;
		THLowpTensor_resize4d(input, 1, input->size[0], input->size[1], input->size[2]);
	} else if(input->nDimension == 1)
	{
		batch = 0;
		THLowpTensor_resize4d(input, 1, input->size[0], 1, 1);
	}
	
	long batchSize = input->size[0];
	long nInputPlane  = module->SpatialConvolution.nInputPlane;
	long nOutputPlane = module->SpatialConvolution.nOutputPlane;
	long inputWidth   = input->size[3];
	long inputHeight  = input->size[2];
	long outputWidth  = (inputWidth + 2*padW - kW) / dW + 1;
	long outputHeight = (inputHeight + 2*padH - kH) / dH + 1;

	if (outputWidth < 1 || outputHeight < 1)
		THError("Given input size: (%dx%dx%d). Calculated output size: (%dx%dx%d). Output size is too small",
		nInputPlane,inputHeight,inputWidth,nOutputPlane,outputHeight,outputWidth);

	THLowpTensor_resize3d(finput, batchSize, kW*kH*nInputPlane, outputHeight*outputWidth);
	THLowpTensor_resize4d(output, batchSize, nOutputPlane, outputHeight, outputWidth);

	long t;
#pragma omp parallel for if(batchSize >= 4) private(t)
	for (t = 0; t < batchSize; t++) {
		THFloatTensor *input_t = THFloatTensor_newSelect(input, 0, t);
		THFloatTensor *output_t = THFloatTensor_newSelect(output, 0, t);
		THFloatTensor *finput_t = THFloatTensor_newSelect(finput, 0, t);

		nn_SpatialConvolutionMM_updateOutput_frame(input_t, output_t, weight, bias, finput_t,
			kW, kH, dW, dH, padW, padH,
			nInputPlane, inputWidth, inputHeight,
			nOutputPlane, outputWidth, outputHeight);

		THFloatTensor_free(input_t);
		THFloatTensor_free(output_t);
		THFloatTensor_free(finput_t);
	}

	if (batch == 0) {
		THLowpTensor_resize3d(output, nOutputPlane, outputHeight, outputWidth);
		THLowpTensor_resize3d(input, nInputPlane, inputHeight, inputWidth);
	}

	return output;
}
