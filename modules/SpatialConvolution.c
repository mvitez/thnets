#include "../thnets.h"

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
