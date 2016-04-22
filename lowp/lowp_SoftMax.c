#include "../thnets.h"

THFloatTensor *Lowp_SoftMax_updateOutput(struct module *module, THFloatTensor *input)
{
	THFloatTensor *output = module->output;
	unsigned char *input_data, *output_data;
	long nframe = 0, dim = 0, stride = 0;
	long t;

	if(input->nDimension == 1)
	{
		nframe = 1;
		dim = input->size[0];
		stride = 1;
	}
	else if(input->nDimension == 2)
	{
		nframe = input->size[0];
		dim = input->size[1];
		stride = 1;
	}
	else if(input->nDimension == 3)
	{
		nframe = 1;
		dim = input->size[0];
		stride = input->size[1]*input->size[2];
	}
	else if(input->nDimension == 4)
	{
		nframe = input->size[0];
		dim = input->size[1];
		stride = input->size[2]*input->size[3];
	}
	else
		THError("1D, 2D, 3D or 4D tensor expected");

	THLowpTensor_resizeAs(output, input);

	input_data = (unsigned char *)THFloatTensor_data(input);
	output_data = (unsigned char *)THFloatTensor_data(output);
	float invmult = input->mult ? 1 / input->mult : 0;
	float *tmp = malloc(dim * sizeof(*tmp));

#pragma omp parallel for private(t)
	for(t = 0; t < stride*nframe; t++)
	{
		unsigned char *input_ptr = input_data + (t/stride)*dim*stride + t % stride;
		unsigned char *output_ptr = output_data + (t/stride)*dim*stride + t % stride;

		unsigned char inputMax = 0;
		float sum, max;

		long d;
		for(d = 0; d < dim; d++)
			if (input_ptr[d*stride] >= inputMax)
				inputMax = input_ptr[d*stride];
		
		sum = 0;
		for(d = 0; d < dim; d++)
		{
			tmp[d] = THExpMinusApprox((inputMax - input_ptr[d*stride]) * invmult + input->sub);
			sum += tmp[d];
		}

		if(stride * nframe > 1)
		{
			for(d = 0; d < dim; d++)
				tmp[d] *= 1/sum;
			for(d = 0; d < dim; d++)
				output_ptr[d*stride] = 255 * tmp[d];
		} else {
			max = 0;
			for(d = 0; d < dim; d++)
			{
				tmp[d] *= 1/sum;
				if(tmp[d] > max)
					max = tmp[d];
			}
			for(d = 0; d < dim; d++)
				output_ptr[d*stride] = 255 * tmp[d] * (1/max);
			output->mult = 255/max;
		}
	}
	free(tmp);
	output->sub = 0;
	if(stride * nframe > 1)
		output->mult = 255;
	return output;
}
