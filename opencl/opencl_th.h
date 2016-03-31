int thopencl_init();
cl_mem OpenCL_Buffer(const void *data, size_t len);
THFloatTensor *THOpenCLTensor_newFromFloatTensor(THFloatTensor *t);
THFloatTensor *THFloatTensor_newFromOpenCLTensor(THFloatTensor *t);
THFloatTensor *THFloatTensor_newFromOpenCLImageTensor(THFloatTensor *t);
THFloatTensor *THOpenCLTensor_newFromImageTensor(THFloatTensor *t);
THFloatTensor *THOpenCLTensor_newFromWeightTensor(THFloatTensor *t, int nInputPlanes, int kW, int kH);
#ifdef HAVEFP16
THFloatTensor *THHalfOpenCLTensor_newFromFloatTensor(THFloatTensor *t);
THFloatTensor *THFloatTensor_newFromHalfOpenCLTensor(THFloatTensor *t);
THFloatTensor *THFloatTensor_newFromHalfOpenCLImageTensor(THFloatTensor *t);
THFloatTensor *THHalfOpenCLTensor_newFromImageTensor(THFloatTensor *t);
THFloatTensor *THHalfOpenCLTensor_newFromWeightTensor(THFloatTensor *t, int nInputPlanes, int kW, int kH);
#endif
THFloatTensor *OpenCL_LoadImage(const unsigned char *src, int width, int height, int srcstride, const float *mean, const float *std, int bgr);
struct network *THOpenCL_ToOpenCL(struct network *net);
void OpenCL_GetTensorSizes(THFloatTensor *t, int *nplanes, int *W, int *H);
void subst(char *buf, const char *from, const char* to);
void substi(char *buf, const char *from, int to);
void substf(char *buf, const char *from, float to);
THFloatStorage *THOpenCLStorage_new(long size);
void THOpenCLTensor_resize3d(THFloatTensor *t, long size0, long size1, long size2);
void THOpenCLTensor_resizeAs(THFloatTensor *tdst, THFloatTensor *tsrc);
char *strdup_more(const char *src);
cl_program OpenCL_BuildProgram(const char *src);
void OpenCL_AddSource(char *src, const char *kernelname);
void OpenCL_Build(struct network *net, THFloatTensor *in);

THFloatTensor *OpenCL_SpatialConvolution_updateOutput(struct module *module, THFloatTensor *input);
THFloatTensor *OpenCL_SpatialMaxPooling_updateOutput(struct module *module, THFloatTensor *input);
THFloatTensor *OpenCL_Threshold_updateOutput(struct module *module, THFloatTensor *input);
THFloatTensor *OpenCL_SoftMax_updateOutput(struct module *module, THFloatTensor *input);

extern cl_command_queue cl_queue;
extern int cl_order, cl_datasize;
