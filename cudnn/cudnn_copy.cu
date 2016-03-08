#include <stdio.h>
#ifdef HAVEHALF
#include "cuda_fp16.h"
#endif

extern "C" void THError(const char *fmt, ...);
extern "C" int cuda_maphostmem;

#define errcheck(f) do {int rc = f; if(rc) THError("Error %d in line %s:%d", rc, __FILE__, __LINE__); } while(0)

#define BYTE2FLOAT 0.003921568f // 1/255

__global__ void rgb2float_kernel(float *dst, const unsigned char *src, int width, int height, int srcstride, const float *mean, const float *std)
{
	int c;

	for(c = 0; c < 3; c++)
	{
		dst[4*threadIdx.x + (blockIdx.x + c * height) * width] =
			(src[c + 3*4*threadIdx.x + srcstride*blockIdx.x] * BYTE2FLOAT - mean[c]) / std[c];
		dst[4*threadIdx.x+1 + (blockIdx.x + c * height) * width] =
			(src[c + 3*(4*threadIdx.x+1) + srcstride*blockIdx.x] * BYTE2FLOAT - mean[c]) / std[c];
		dst[4*threadIdx.x+2 + (blockIdx.x + c * height) * width] =
			(src[c + 3*(4*threadIdx.x+2) + srcstride*blockIdx.x] * BYTE2FLOAT - mean[c]) / std[c];
		dst[4*threadIdx.x+3 + (blockIdx.x + c * height) * width] =
			(src[c + 3*(4*threadIdx.x+3) + srcstride*blockIdx.x] * BYTE2FLOAT - mean[c]) / std[c];
	}
}

__global__ void bgr2float_kernel(float *dst, const unsigned char *src, int width, int height, int srcstride, const float *mean, const float *std)
{
	int c;

	for(c = 0; c < 3; c++)
	{
		dst[4*threadIdx.x + (blockIdx.x + c * height) * width] =
			(src[2-c + 3*4*threadIdx.x + srcstride*blockIdx.x] * BYTE2FLOAT - mean[c]) / std[c];
		dst[4*threadIdx.x+1 + (blockIdx.x + c * height) * width] =
			(src[2-c + 3*(4*threadIdx.x+1) + srcstride*blockIdx.x] * BYTE2FLOAT - mean[c]) / std[c];
		dst[4*threadIdx.x+2 + (blockIdx.x + c * height) * width] =
			(src[2-c + 3*(4*threadIdx.x+2) + srcstride*blockIdx.x] * BYTE2FLOAT - mean[c]) / std[c];
		dst[4*threadIdx.x+3 + (blockIdx.x + c * height) * width] =
			(src[2-c + 3*(4*threadIdx.x+3) + srcstride*blockIdx.x] * BYTE2FLOAT - mean[c]) / std[c];
	}
}

__global__ void rgb2half_kernel(unsigned short *dst, const unsigned char *src, int width, int height, int srcstride, const float *mean, const float *std)
{
	int c;

	for(c = 0; c < 3; c++)
	{
		dst[4*threadIdx.x + (blockIdx.x + c * height) * width] =
			__float2half_rn((src[c + 3*4*threadIdx.x + srcstride*blockIdx.x] * BYTE2FLOAT - mean[c]) / std[c]);
		dst[4*threadIdx.x+1 + (blockIdx.x + c * height) * width] =
			__float2half_rn((src[c + 3*(4*threadIdx.x+1) + srcstride*blockIdx.x] * BYTE2FLOAT - mean[c]) / std[c]);
		dst[4*threadIdx.x+2 + (blockIdx.x + c * height) * width] =
			__float2half_rn((src[c + 3*(4*threadIdx.x+2) + srcstride*blockIdx.x] * BYTE2FLOAT - mean[c]) / std[c]);
		dst[4*threadIdx.x+3 + (blockIdx.x + c * height) * width] =
			__float2half_rn((src[c + 3*(4*threadIdx.x+3) + srcstride*blockIdx.x] * BYTE2FLOAT - mean[c]) / std[c]);
	}
}

__global__ void bgr2half_kernel(unsigned short *dst, const unsigned char *src, int width, int height, int srcstride, const float *mean, const float *std)
{
	int c;

	for(c = 0; c < 3; c++)
	{
		dst[4*threadIdx.x + (blockIdx.x + c * height) * width] =
			__float2half_rn((src[2-c + 3*4*threadIdx.x + srcstride*blockIdx.x] * BYTE2FLOAT - mean[c]) / std[c]);
		dst[4*threadIdx.x+1 + (blockIdx.x + c * height) * width] =
			__float2half_rn((src[2-c + 3*(4*threadIdx.x+1) + srcstride*blockIdx.x] * BYTE2FLOAT - mean[c]) / std[c]);
		dst[4*threadIdx.x+2 + (blockIdx.x + c * height) * width] =
			__float2half_rn((src[2-c + 3*(4*threadIdx.x+2) + srcstride*blockIdx.x] * BYTE2FLOAT - mean[c]) / std[c]);
		dst[4*threadIdx.x+3 + (blockIdx.x + c * height) * width] =
			__float2half_rn((src[2-c + 3*(4*threadIdx.x+3) + srcstride*blockIdx.x] * BYTE2FLOAT - mean[c]) / std[c]);
	}
}

extern "C" float *cuda_rgb2float(float *dst, const unsigned char *src, int width, int height, int srcstride, const float *mean, const float *std, int bgr);
extern "C" unsigned short *cuda_rgb2half(unsigned short *dst, const unsigned char *src, int width, int height, int srcstride, const float *mean, const float *std, int bgr);

float *cuda_rgb2float(float *dst, const unsigned char *src, int width, int height, int srcstride, const float *mean, const float *std, int bgr)
{
	unsigned char *csrc;
	float *cmean, *cstd;
	
	if(cuda_maphostmem)
	{
		if(cuda_maphostmem == 2)
			errcheck(cudaHostRegister((void *)src, height*srcstride, cudaHostRegisterMapped));
		errcheck(cudaHostGetDevicePointer((void **)&csrc, (void *)src, 0));
	} else {
		errcheck(cudaMalloc((void **)&csrc, height * srcstride));
		errcheck(cudaMemcpy(csrc, src, height * srcstride, cudaMemcpyHostToDevice));
	}
	errcheck(cudaMalloc((void **)&cmean, 3 * sizeof(*cmean)));
	errcheck(cudaMemcpy(cmean, mean, 3 * sizeof(*cmean), cudaMemcpyHostToDevice));
	errcheck(cudaMalloc((void **)&cstd, 3 * sizeof(*cstd)));
	errcheck(cudaMemcpy(cstd, std, 3 * sizeof(*std), cudaMemcpyHostToDevice));

	if(bgr)
		bgr2float_kernel<<<height, width/4>>>(dst, csrc, width, height, srcstride, cmean, cstd);
	else rgb2float_kernel<<<height, width/4>>>(dst, csrc, width, height, srcstride, cmean, cstd);
	errcheck(cudaDeviceSynchronize());
	
	if(cuda_maphostmem == 2)
		cudaHostUnregister((void *)src);
	else if(cuda_maphostmem == 0)
		cudaFree(csrc);
	cudaFree(cmean);
	cudaFree(cstd);
	
	return dst;
}

unsigned short *cuda_rgb2half(unsigned short *dst, const unsigned char *src, int width, int height, int srcstride, const float *mean, const float *std, int bgr)
{
	unsigned char *csrc;
	float *cmean, *cstd;
	
	if(cuda_maphostmem)
	{
		if(cuda_maphostmem == 2)
			errcheck(cudaHostRegister((void *)src, height*srcstride, cudaHostRegisterMapped));
		errcheck(cudaHostGetDevicePointer((void **)&csrc, (void *)src, 0));
	} else {
		errcheck(cudaMalloc((void **)&csrc, height * srcstride));
		errcheck(cudaMemcpy(csrc, src, height * srcstride, cudaMemcpyHostToDevice));
	}
	errcheck(cudaMalloc((void **)&cmean, 3 * sizeof(*cmean)));
	errcheck(cudaMemcpy(cmean, mean, 3 * sizeof(*cmean), cudaMemcpyHostToDevice));
	errcheck(cudaMalloc((void **)&cstd, 3 * sizeof(*cstd)));
	errcheck(cudaMemcpy(cstd, std, 3 * sizeof(*std), cudaMemcpyHostToDevice));

	if(bgr)
		bgr2half_kernel<<<height, width/4>>>(dst, csrc, width, height, srcstride, cmean, cstd);
	else rgb2half_kernel<<<height, width/4>>>(dst, csrc, width, height, srcstride, cmean, cstd);
	errcheck(cudaDeviceSynchronize());
	
	if(cuda_maphostmem)
		cudaHostUnregister((void *)src);
	else cudaFree(csrc);
	cudaFree(cmean);
	cudaFree(cstd);
	
	return dst;
}

__global__ void fillwithone(float *dst, int stride)
{
	dst[threadIdx.x + blockIdx.x * stride] = 1;
}

__global__ void fillwithoneH(__half *dst, int stride)
{
	dst[threadIdx.x + blockIdx.x * stride] = __float2half(1);
}

extern "C" void cuda_fillwithone(int n1, int n2, float *data, int stride);
extern "C" void cuda_fillwithoneH(int n1, int n2, float *data, int stride);

void cuda_fillwithone(int n1, int n2, float *data, int stride)
{
	fillwithone<<<n1, n2>>>(data, stride);
}

void cuda_fillwithoneH(int n1, int n2, float *data, int stride)
{
	fillwithoneH<<<n1, n2>>>((__half *)data, stride);
}
