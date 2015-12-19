#include "cudnn.h"

#define errcheck(f) do {int rc = f; if(rc) THError("Error %d in line %s:%d", rc, __FILE__, __LINE__); } while(0)
cudnnHandle_t THcudnn_getHandle();
int THcudnn_TensorDescriptor(cudnnTensorDescriptor_t *d, THFloatTensor *t);
THFloatTensor *THCudaTensor_newFromFloatTensor(THFloatTensor *t);
THFloatTensor *THFloatTensor_newFromCudaTensor(THFloatTensor *t);
void THCudatTensor_resize4d(THFloatTensor *t, long size0, long size1, long size2, long size3);
struct network *THcudnn_ToCUDNN(struct network *net);
void THCudaTensor_resize4d(THFloatTensor *t, long size0, long size1, long size2, long size3);

THFloatTensor *cudnn_SpatialConvolution_updateOutput(struct module *module, THFloatTensor *input);
THFloatTensor *cudnn_SpatialMaxPooling_updateOutput(struct module *module, THFloatTensor *input);
THFloatTensor *cudnn_Threshold_updateOutput(struct module *module, THFloatTensor *input);
THFloatTensor *cudnn_SoftMax_updateOutput(struct module *module, THFloatTensor *input);
