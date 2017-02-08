#include <math.h>
#include <string.h>
#include <stdio.h>
#include "../thnets.h"

static void nnfree_Sequential(struct module *mod)
{
	freenetwork(mod->Sequential.net);
}

int nnload_Sequential(struct module *mod, struct nnmodule *n)
{
	mod->type = MT_Sequential;
	mod->Sequential.net = Module2Network(n);
	mod->nnfree = nnfree_Sequential;
	mod->updateOutput = nn_Sequential_updateOutput;
	return 0;
}

THFloatTensor *nn_Sequential_updateOutput(struct module *module, THFloatTensor *input)
{
	int nelem = module->Sequential.net->nelem;
	int i;
	struct module *modules = module->Sequential.net->modules;
	double t = 0;

	for(i = 0; i < nelem; i++)
	{
		if(th_profile)
			t = th_seconds();
		input = modules[i].updateOutput(&modules[i], input);
		if(th_profile)
		{
			t = th_seconds() - t;
			if(modules[i].type == MT_SpatialConvolutionMM ||
				modules[i].type == MT_SpatialConvolutionVirtMM ||
				modules[i].type == MT_SpatialConvolution)
			{
				double flops = 2.0 * THFloatTensor_nElement(input) * modules[i].SpatialConvolution.nInputPlane *
					modules[i].SpatialConvolution.kW * modules[i].SpatialConvolution.kH;
				printf("%f seconds for module %d, %f Gflops/s\n", t, i+1, flops * 1e-9 / t);
				th_convtot += t;
				th_convflops += flops;
			} else printf("%f seconds for module %d\n", t, i+1);
		}
		if(th_debug > 1)
			printf("  %d) %d %d %ld %ld %ld %ld\n", i+1, modules[i].type, input->nDimension, input->size[0], input->size[1], input->size[2], input->size[3]);
	}
	THFloatTensor_set(module->output, input);
	return module->output;
}
