#include <math.h>
#include "../thnets.h"

int nnload_Normalize(struct module *mod, struct nnmodule *n)
{
	mod->type = MT_Normalize;
	mod->updateOutput = nn_Normalize_updateOutput;
	return 0;
}

THFloatTensor *nn_Normalize_updateOutput(struct module *module, THFloatTensor *input)
{
	THFloatTensor *output = module->output;
	long i, j, n;
	float sum;
	float *idata, *odata;

	THFloatTensor_resizeAs(output, input);
	if(input->nDimension == 2)
	{
		for(i = 0; i < input->size[0]; i++)
		{
			sum = 0;
			idata = input->storage->data + input->storageOffset + i * input->stride[0];
			odata = output->storage->data + output->storageOffset + i * output->stride[0];
			for(j = 0; j < input->size[1]; j++)
				sum += idata[j] * idata[j];
			sum = sqrtf(sum);
			for(j = 0; j < input->size[1]; j++)
				odata[j] = idata[j] / sum;
		}
	} else {
		sum = 0;
		idata = input->storage->data + input->storageOffset;
		odata = output->storage->data + output->storageOffset;
		n = THFloatTensor_nElement(input);
		for(j = 0; j < n; j++)
			sum += idata[j] * idata[j];
		sum = sqrtf(sum);
		for(j = 0; j < n; j++)
			odata[j] = idata[j] / sum;
	}
	return output;
}
