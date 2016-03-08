#ifdef HAVEHALF
#include "cuda_fp16.h"
#endif

#define CUDA_KERNEL_LOOP(i, n) \
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

// Use 1024 threads per block, which requires cuda sm_2x or above
static int CUDA_NUM_THREADS = 1024;

// CUDA: number of blocks for threads.
static int GET_BLOCKS(const int N)
{
	return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

__global__ void col2im_kernel(const int n, const float* data_col,
	const int height, const int width, const int channels, const int patch_h, const int patch_w,
	const int pad_h, const int pad_w, const int stride_h, const int stride_w, const int height_col, const int width_col,
	float *data_im)
{
	CUDA_KERNEL_LOOP(index, n)
	{
		float val = 0;
		int w = index % width + pad_w;
		int h = (index / width) % height + pad_h;
		int c = index / (width * height);
		// compute the start and end of the output
		int w_col_start = (w < patch_w) ? 0 : (w - patch_w) / stride_w + 1;
		int w_col_end = min(w / stride_w + 1, width_col);
		int h_col_start = (h < patch_h) ? 0 : (h - patch_h) / stride_h + 1;
		int h_col_end = min(h / stride_h + 1, height_col);
		int offset = (c * patch_h * patch_w + h * patch_w + w) * height_col * width_col;
		int coeff_h_col = (1 - stride_h * patch_w * height_col) * width_col;
		int coeff_w_col = (1 - stride_w * height_col * width_col);
		for (int h_col = h_col_start; h_col < h_col_end; ++h_col)
			for (int w_col = w_col_start; w_col < w_col_end; ++w_col)
				val += data_col[offset + h_col * coeff_h_col + w_col * coeff_w_col];
		data_im[index] = val;
	}
}

__global__ void col2im_kernelH(const int n, const __half *data_col,
	const int height, const int width, const int channels, const int patch_h, const int patch_w,
	const int pad_h, const int pad_w, const int stride_h, const int stride_w, const int height_col, const int width_col,
	__half *data_im)
{
	CUDA_KERNEL_LOOP(index, n)
	{
		float val = 0;
		int w = index % width + pad_w;
		int h = (index / width) % height + pad_h;
		int c = index / (width * height);
		// compute the start and end of the output
		int w_col_start = (w < patch_w) ? 0 : (w - patch_w) / stride_w + 1;
		int w_col_end = min(w / stride_w + 1, width_col);
		int h_col_start = (h < patch_h) ? 0 : (h - patch_h) / stride_h + 1;
		int h_col_end = min(h / stride_h + 1, height_col);
		int offset = (c * patch_h * patch_w + h * patch_w + w) * height_col * width_col;
		int coeff_h_col = (1 - stride_h * patch_w * height_col) * width_col;
		int coeff_w_col = (1 - stride_w * height_col * width_col);
		for (int h_col = h_col_start; h_col < h_col_end; ++h_col)
			for (int w_col = w_col_start; w_col < w_col_end; ++w_col)
				val = val + __half2float(data_col[offset + h_col * coeff_h_col + w_col * coeff_w_col]);
		data_im[index] = __float2half(val);
	}
}

void col2im(const float *data_col, const int channels,
    const int height, const int width, const int patch_h, const int patch_w, const int pad_h,
    const int pad_w, const int stride_h, const int stride_w, float *data_im)
{
	int height_col = (height + 2 * pad_h - patch_h) / stride_h + 1;
	int width_col = (width + 2 * pad_w - patch_w) / stride_w + 1;
	int num_kernels = channels * height * width;
	// To avoid involving atomic operations, we will launch one kernel per
	// bottom dimension, and then in the kernel add up the top dimensions.
	col2im_kernel <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>> (
		num_kernels, data_col, height, width, channels,
		patch_h, patch_w, pad_h, pad_w, stride_h, stride_w,
		height_col, width_col, data_im);
}

void col2imH(const __half *data_col, const int channels,
    const int height, const int width, const int patch_h, const int patch_w, const int pad_h,
    const int pad_w, const int stride_h, const int stride_w, __half *data_im)
{
	int height_col = (height + 2 * pad_h - patch_h) / stride_h + 1;
	int width_col = (width + 2 * pad_w - patch_w) / stride_w + 1;
	int num_kernels = channels * height * width;
	// To avoid involving atomic operations, we will launch one kernel per
	// bottom dimension, and then in the kernel add up the top dimensions.
	col2im_kernelH <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>> (
		num_kernels, data_col, height, width, channels,
		patch_h, patch_w, pad_h, pad_w, stride_h, stride_w,
		height_col, width_col, data_im);
}
