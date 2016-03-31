#include "cudnn.h"

#define errcheck(f) do {int rc = f; if(rc) THError("Error %d in line %s:%d", rc, __FILE__, __LINE__); } while(0)
cudnnHandle_t THcudnn_getHandle();
int THcudnn_TensorDescriptor(cudnnTensorDescriptor_t *d, THFloatTensor *t);
THFloatStorage *THCudaStorage_new(long size);
THFloatTensor *THCudaTensor_newFromFloatTensor(THFloatTensor *t);
THFloatTensor *THFloatTensor_newFromCudaTensor(THFloatTensor *t);
void THCudaTensor_resize4d(THFloatTensor *t, long size0, long size1, long size2, long size3);
void THCudaTensor_resize3d(THFloatTensor *t, long size0, long size1, long size2);
void THCudaTensor_resize2d(THFloatTensor *t, long size0, long size1);
void THCudaTensor_resize1d(THFloatTensor *t, long size0);
void THCudaTensor_resizeAs(THFloatTensor *tdst, THFloatTensor *tsrc);
struct network *THcudnn_ToCUDNN(struct network *net);
void THCudaTensor_Ones(THFloatTensor *t);
void THCudaBlas_gemm(char transa, char transb, long m, long n, long k, float alpha, float *a, long lda, float *b, long ldb, float beta, float *c, long ldc);
float *cuda_rgb2float(float *dst, const unsigned char *src, int width, int height, int srcstride, const float *mean, const float *std, int bgr);
unsigned short *cuda_rgb2half(unsigned short *dst, const unsigned char *src, int width, int height, int srcstride, const float *mean, const float *std, int bgr);
void cuda_fillwithone(int n1, int n2, float *data, int stride);
void cuda_fillwithoneH(int n1, int n2, float *data, int stride);
#ifdef HAVEFP16
THFloatTensor *THHalfCudaTensor_newFromFloatTensor(THFloatTensor *t);
THFloatTensor *THFloatTensor_newFromHalfCudaTensor(THFloatTensor *t);
#endif

THFloatTensor *cudnn_SpatialConvolution_updateOutput(struct module *module, THFloatTensor *input);
THFloatTensor *cudnn_SpatialMaxPooling_updateOutput(struct module *module, THFloatTensor *input);
THFloatTensor *cunn_SpatialMaxPooling_updateOutput(struct module *module, THFloatTensor *input);
THFloatTensor *cunn_SpatialMaxUnpooling_updateOutput(struct module *module, THFloatTensor *input);
THFloatTensor *cudnn_SpatialBatchNormalization_updateOutput(struct module *module, THFloatTensor *input);
THFloatTensor *cudnn_Threshold_updateOutput(struct module *module, THFloatTensor *input);
THFloatTensor *cudnn_SoftMax_updateOutput(struct module *module, THFloatTensor *input);
THFloatTensor *cunn_SpatialFullConvolution_updateOutput(struct module *module, THFloatTensor *input);

extern int floattype;
