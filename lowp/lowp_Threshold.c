#include "../thnets.h"

THFloatTensor *Lowp_Threshold_updateOutput(struct module *module, THFloatTensor *input)
{
	THFloatTensor *output = module->output;
	int inPlace = module->Threshold.inplace == 1;
	unsigned char val = THLowp_ScaleFloat(input, module->Threshold.val);
	unsigned char threshold = THLowp_ScaleFloat(input, module->Threshold.threshold);

	long i, n = THFloatTensor_nElement(input);
	if (inPlace)
	{
		unsigned char *data = (unsigned char *)input->storage->data;
		for(i = 0; i < n; i++)
			if (data[i] <= threshold)
				data[i] = val;
		THFloatTensor_set(output, input);
	} else {
		THLowpTensor_resizeAs(output, input);
		unsigned char *src = (unsigned char *)input->storage->data;
		unsigned char *dst = (unsigned char *)output->storage->data;
		for(i = 0; i < n; i++)
			dst[i] = src[i] > threshold ? src[i] : val;
		output->mult = input->mult;
		output->sub = input->sub;
	}
	return output;
}
