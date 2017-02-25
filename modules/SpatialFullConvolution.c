#include <string.h>
#include "../thnets.h"

static void nnfree_SpatialFullConvolution(struct module *mod)
{
	THFloatTensor_free(mod->SpatialFullConvolution.bias);
	THFloatTensor_free(mod->SpatialFullConvolution.weight);
	THFloatTensor_free(mod->SpatialFullConvolution.ones);
	THFloatTensor_free(mod->SpatialFullConvolution.columns);
}

int nnload_SpatialFullConvolution(struct module *mod, struct nnmodule *n)
{
	struct table *t = n->table;
	mod->type = MT_SpatialFullConvolution;
	mod->updateOutput = nn_SpatialFullConvolution_updateOutput;
	mod->nnfree = nnfree_SpatialFullConvolution;
	struct SpatialFullConvolution *m = &mod->SpatialFullConvolution;
	m->padW = TableGetNumber(t, "padW");
	m->padH = TableGetNumber(t, "padH");
	m->adjW = TableGetNumber(t, "adjW");
	m->adjH = TableGetNumber(t, "adjH");
	m->dW = TableGetNumber(t, "dW");
	m->dH = TableGetNumber(t, "dH");
	m->kW = TableGetNumber(t, "kW");
	m->kH = TableGetNumber(t, "kH");
	m->nInputPlane = TableGetNumber(t, "nInputPlane");
	m->nOutputPlane = TableGetNumber(t, "nOutputPlane");
	m->bias = TableGetTensor(t, "bias");
	m->weight = TableGetTensor(t, "weight");
	m->columns = THFloatTensor_new();
	m->ones = THFloatTensor_new();
	return 0;
}

static void col2im(const float *data_col, const int channels,
	const int height, const int width, const int patch_h, const int patch_w,
	const int pad_h, const int pad_w,
	const int stride_h, const int stride_w,
	float *data_im)
{
	int c, h, w;
	memset(data_im, 0, sizeof(float)*height * width * channels);
	int height_col = (height + 2 * pad_h - patch_h) / stride_h + 1;
	int width_col = (width + 2 * pad_w - patch_w) / stride_w + 1;
	int channels_col = channels * patch_h * patch_w;
	for (c = 0; c < channels_col; ++c)
	{
		int w_offset = c % patch_w;
		int h_offset = (c / patch_w) % patch_h;
		int c_im = c / patch_h / patch_w;
		for (h = 0; h < height_col; ++h)
		{
			for (w = 0; w < width_col; ++w)
			{
				int h_pad = h * stride_h - pad_h + h_offset;
				int w_pad = w * stride_w - pad_w + w_offset;
				if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
					data_im[(c_im * height + h_pad) * width + w_pad] +=
						data_col[(c * height_col + h) * width_col + w];
			}
		}
	}
}

THFloatTensor *nn_SpatialFullConvolution_updateOutput(struct module *module, THFloatTensor *input)
{
	int dW = module->SpatialFullConvolution.dW;
	int dH = module->SpatialFullConvolution.dH;
	int kW = module->SpatialFullConvolution.kW;
	int kH = module->SpatialFullConvolution.kH;
	int padW = module->SpatialFullConvolution.padW;
	int padH = module->SpatialFullConvolution.padH;
	int adjW = module->SpatialFullConvolution.adjW;
	int adjH = module->SpatialFullConvolution.adjH;

	THFloatTensor *weight = module->SpatialFullConvolution.weight;
	THFloatTensor *bias = module->SpatialFullConvolution.bias;
	THFloatTensor *output = module->output;
	THFloatTensor *columns = module->SpatialFullConvolution.columns;
	THFloatTensor *ones = module->SpatialFullConvolution.ones;

	int nInputPlane = (int)weight->size[0];
	int nOutputPlane = (int)weight->size[1];

	if(input->nDimension != 3 && input->nDimension != 4)
		THError("3D or 4D (batch mode) tensor is expected");

	int batch = 1;
	if(input->nDimension == 3)
	{
		if(input->size[0] != nInputPlane)
			THError("input channels and nInputPlane dont match");
		// Force batch
		batch = 0;
		THFloatTensor_resize4d(input, 1, input->size[0], input->size[1], input->size[2]);
	} else if(input->size[1] != nInputPlane)
		THError("input channels and nInputPlane dont match");

	long inputWidth   = input->size[3];
	long inputHeight  = input->size[2];
	long outputWidth  = (inputWidth - 1) * dW - 2*padW + kW + adjW;
	long outputHeight = (inputHeight - 1) * dH - 2*padH + kH + adjH;

	// Batch size + input planes
	long batchSize = input->size[0];

	// Resize output
	THFloatTensor_resize4d(output, batchSize, nOutputPlane, outputHeight, outputWidth);

	// Resize temporary columns
	THFloatTensor_resize2d(columns, nOutputPlane*kW*kH, inputHeight*inputWidth);

	// Define a buffer of ones, for bias accumulation
	// Note: this buffer can be shared with other modules, it only ever gets increased,
	// and always contains ones.
	if (ones->nDimension != 2 || ones->size[0]*ones->size[1] < outputHeight*outputWidth)
	{
		// Resize plane and fill with ones...
		THFloatTensor_resize2d(ones, outputHeight, outputWidth);
		THFloatTensor_fill(ones, 1);
	}

	int elt;
	// For each elt in batch, do:
	for (elt = 0; elt < batchSize; elt ++)
	{
		// Matrix mulitply per output:

		// Helpers
		THFloatTensor *input_n = THFloatTensor_newSelect(input, 0, elt);
		THFloatTensor *output_n = THFloatTensor_newSelect(output, 0, elt);

		// M,N,K are dims of matrix A and B
		// (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
		long m = weight->size[1] * weight->size[2] * weight->size[3];
		long n = columns->size[1];
		long k = weight->size[0];

		// Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
		THBlas_gemm(
			'n', 't',
			n, m, k,
			1,
			THFloatTensor_data(input_n), n,
			THFloatTensor_data(weight), m,
			0,
			THFloatTensor_data(columns), n
		);

		// Unpack columns back into input:
		col2im(
			THFloatTensor_data(columns),
			nOutputPlane, (int)outputHeight, (int)outputWidth, kH, kW, padH, padW, dH, dW,
			THFloatTensor_data(output_n)
		);

		// Do Bias after:
		// M,N,K are dims of matrix A and B
		// (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
		long m_ = nOutputPlane;
		long n_ = outputHeight * outputWidth;
		long k_ = 1;

		// Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
		THBlas_gemm(
			't', 'n',
			n_, m_, k_,
			1,
			THFloatTensor_data(ones), k_,
			THFloatTensor_data(bias), k_,
			1,
			THFloatTensor_data(output_n), n_
		);

		THFloatTensor_free(input_n);
		THFloatTensor_free(output_n);
	}

	// Resize output
	if (batch == 0)
	{
		THFloatTensor_resize3d(output, nOutputPlane, outputHeight, outputWidth);
		THFloatTensor_resize3d(input, nInputPlane, inputHeight, inputWidth);
	}
	return output;
}
