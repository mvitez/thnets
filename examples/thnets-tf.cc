// Build with
// TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
// g++ -std=c++11 -shared thnets-tf.cc -o thnets-tf.so -fPIC -I $TF_INC -lthnets

extern "C" {
#include "thnets.h"
};

#include "tensorflow/core/framework/op.h"

REGISTER_OP("Conv")
    .Input("input: float")
	.Input("weight: float")
	.Input("bias: float")
    .Attr("strides: list(int)")
	.Attr("padding: string")
    .Output("output: float");

#include "tensorflow/core/framework/op_kernel.h"

// Initialize thnets on library loading
__attribute__((constructor)) void init()
{
    THInit();
}

using namespace tensorflow;

// Manually build a THFloatTensor struct with associated THFloatStorage from a Tensor
// We do this to keep preallocated structures
void Tensor2THFloatTensor(const Tensor *in, THFloatTensor *out, THFloatStorage *storage)
{
	int i;
	long stride = 1;

	storage->data = (float *)in->tensor_data().data();
	storage->nref = 1;
	storage->mustfree = 0;
	out->nDimension = in->dims();
	for(i = 0; i < out->nDimension; i++)
		out->size[i] = in->dim_size(i);
	for(i = out->nDimension - 1; i >= 0; i--)
	{
		out->stride[i] = stride;
		stride *= out->size[i];
	}
	out->storage = storage;
	out->storageOffset = 0;
}

class ConvOp : public OpKernel {
public:
	explicit ConvOp(OpKernelConstruction* context) : OpKernel(context)
	{
		OP_REQUIRES_OK(context, context->GetAttr("strides", &strides));
		OP_REQUIRES_OK(context, context->GetAttr("padding", &padding));
	}

	void Compute(OpKernelContext* context) override
	{
		const Tensor& input_tensor = context->input(0);
		OP_REQUIRES(context, input_tensor.dims() == 4,
			errors::InvalidArgument("Conv expects a 4-D input tensor."));

		const Tensor& weight_tensor = context->input(1);
		OP_REQUIRES(context, weight_tensor.dims() == 4,
			errors::InvalidArgument("Conv expects a 4-D weight tensor."));

		const Tensor& bias_tensor = context->input(2);
		OP_REQUIRES(context, bias_tensor.dims() == 1,
			errors::InvalidArgument("Conv expects a 1-D bias tensor."));

		// Create the output tensor
		Tensor* output_tensor = NULL;
		int dW = strides[3];
		int dH = strides[2];
		int padW, padH;
		int kW = weight_tensor.dim_size(3);
		int kH = weight_tensor.dim_size(2);
		if(padding == "VALID")
		{
			padW = 0;
			padH = 0;
		} else {
			padW = (kW - 1) / 2;
			padH = (kH - 1) / 2;
		}

		long outputHeight = floor((input_tensor.dim_size(2) + 2 * padW - kH) / dW) + 1;
		long outputWidth = floor((input_tensor.dim_size(3) + 2 * padH - kW) / dH) + 1;
		TensorShape outshape({input_tensor.dim_size(0), weight_tensor.dim_size(0), outputHeight, outputWidth});
		OP_REQUIRES_OK(context, context->allocate_output(0, outshape, &output_tensor));

		// Build thnets tensors
		THFloatTensor input_thtensor;
		THFloatTensor weight_thtensor;
		THFloatTensor bias_thtensor;
		THFloatTensor output_thtensor;
		THFloatStorage input_thstorage;
		THFloatStorage weight_thstorage;
		THFloatStorage bias_thstorage;
		THFloatStorage output_thstorage;
		
		Tensor2THFloatTensor(&input_tensor, &input_thtensor, &input_thstorage);
		Tensor2THFloatTensor(&weight_tensor, &weight_thtensor, &weight_thstorage);
		Tensor2THFloatTensor(&bias_tensor, &bias_thtensor, &bias_thstorage);
		Tensor2THFloatTensor(output_tensor, &output_thtensor, &output_thstorage);
		// thnets wants a 2-D tensor for weights
		THFloatTensor_resize2d(&weight_thtensor, weight_thtensor.size[0],
			weight_thtensor.size[1] * weight_thtensor.size[2] * weight_thtensor.size[3]);
		
		// Build the convolution module
		struct module mod;

		mod.SpatialConvolution.kW = kW;
		mod.SpatialConvolution.kH = kH;
		mod.SpatialConvolution.padW = padW;
		mod.SpatialConvolution.padH = padH;
		mod.SpatialConvolution.dW = dW;
		mod.SpatialConvolution.dH = dH;
		mod.SpatialConvolution.nInputPlane = weight_tensor.dim_size(1);
		mod.SpatialConvolution.nOutputPlane = weight_tensor.dim_size(0);
		if(padding == "VALID")
		{
			mod.SpatialConvolution.padW = padH;
			mod.SpatialConvolution.padH = padW;
		} else {
			mod.SpatialConvolution.padW = (mod.SpatialConvolution.kW - 1) / 2;
			mod.SpatialConvolution.padH = (mod.SpatialConvolution.kH - 1) / 2;
		}
		mod.SpatialConvolution.weight = &weight_thtensor;
		mod.SpatialConvolution.bias = &bias_thtensor;
		mod.SpatialConvolution.finput = 0;
		mod.type = MT_SpatialConvolutionVirtMM;
		mod.output = &output_thtensor;

		// run the convolution, the output is written directly into the TensorFlow tensor
		nn_SpatialConvolutionMM_updateOutput(&mod, &input_thtensor);
		
		// Everything has been allocated on the stack, so no freeing is necessary
	}
private:
	std::vector<int32> strides;
	string padding;
};

REGISTER_KERNEL_BUILDER(Name("Conv").Device(DEVICE_CPU), ConvOp);