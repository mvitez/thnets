#include <string.h>
#include "../thnets.h"

int nnload_SpatialMaxUnpooling(struct module *mod, struct nnmodule *n)
{
	struct table *t = n->table;
	mod->type = MT_SpatialMaxUnpooling;
	mod->updateOutput = nn_SpatialMaxUnpooling_updateOutput;
	struct SpatialMaxUnpooling *m = &mod->SpatialMaxUnpooling;
	m->pooling = TableGetNNModule(t, "pooling");
	return 0;
}

static void SpatialMaxUnpooling_updateOutput_frame(float *input_p, float *output_p,
	float *ind_p,
	long nslices,
	long iwidth, long iheight,
	long owidth, long oheight)
{
	long k;
#pragma omp parallel for private(k)
	for (k = 0; k < nslices; k++)
	{    
		float *output_p_k = output_p + k*owidth*oheight;
		float *input_p_k = input_p + k*iwidth*iheight;
		float *ind_p_k = ind_p + k*iwidth*iheight;

		long i, j, maxp;
		for(i = 0; i < iheight; i++)
		{
			for(j = 0; j < iwidth; j++)
			{
				maxp = ind_p_k[i*iwidth + j] - 1;  /* retrieve position of max */
				if(maxp<0 || maxp>=owidth*oheight)
					THError("invalid max index %d, owidth= %d, oheight= %d",maxp,owidth,oheight);
				output_p_k[maxp] = input_p_k[i*iwidth + j]; /* update output */
			}
		}
	}
}

THFloatTensor *nn_SpatialMaxUnpooling_updateOutput(struct module *module, THFloatTensor *input)
{
	THFloatTensor *output = module->output;
	THFloatTensor *indices = 0;
	int dimw = 2;
	int dimh = 1;
	int nbatch = 1;
	int nslices;
	int iheight;
	int iwidth;
	int oheight;
	int owidth;
	int i;
	float *input_data;
	float *output_data;
	float *indices_data;

	if(input->nDimension != 3 && input->nDimension != 4)
		THError("3D or 4D (batch mode) tensor is expected");

	struct network *net = module->net;
	for(i = 0; i < net->nelem; i++)
		if(net->modules[i].type == MT_SpatialMaxPooling &&
			net->modules[i].nnmodule == module->SpatialMaxUnpooling.pooling)
		{
			owidth = net->modules[i].SpatialMaxPooling.iwidth;
			oheight = net->modules[i].SpatialMaxPooling.iheight;
			indices = net->modules[i].SpatialMaxPooling.indices;
			break;
		}
	if (!indices || !THFloatTensor_isSameSizeAs(input, indices))
		THError("Invalid input size w.r.t current indices size");
	if (input->nDimension == 4) 
	{
		nbatch = input->size[0];
		dimw++;
		dimh++;
	}

	/* sizes */
	nslices = input->size[dimh-1];
	iheight = input->size[dimh];
	iwidth = input->size[dimw];

	/* resize output */
	if (input->nDimension == 3)
	{
		THFloatTensor_resize3d(output, nslices, oheight, owidth);
		THFloatTensor_zero(output);

		input_data = THFloatTensor_data(input);
		output_data = THFloatTensor_data(output);
		indices_data = THFloatTensor_data(indices);
		SpatialMaxUnpooling_updateOutput_frame(input_data, output_data,
			indices_data,
			nslices,
			iwidth, iheight,
			owidth, oheight);
	} else {
		long p;

		THFloatTensor_resize4d(output, nbatch, nslices, oheight, owidth);
		THFloatTensor_zero(output);

		input_data = THFloatTensor_data(input);
		output_data = THFloatTensor_data(output);
		indices_data = THFloatTensor_data(indices);

		#pragma omp parallel for private(p)
		for (p = 0; p < nbatch; p++)
		{
			SpatialMaxUnpooling_updateOutput_frame(input_data+p*nslices*iwidth*iheight, output_data+p*nslices*owidth*oheight,
				indices_data+p*nslices*iwidth*iheight,
				nslices,
				iwidth, iheight,
				owidth, oheight);
		}
	}
	return output;
}
