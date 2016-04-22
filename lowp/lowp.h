struct network *THLowp_ToLowp(struct network *net, float range);
THFloatTensor *THFloatTensor_newFromLowpTensor(THFloatTensor *t);
THFloatTensor *THLowpTensor_newFromFloatTensor(THFloatTensor *t);
THFloatTensor *Lowp_LoadImages(unsigned char **src, int nimages, int width, int height, int srcstride, const float *mean, const float *std, int bgr);
unsigned char THLowp_ScaleFloat(THFloatTensor *t, float value);
THFloatStorage *THLowpStorage_new(long size);
void THLowpTensor_resizeAs(THFloatTensor *tdst, THFloatTensor *tsrc);
void THLowpTensor_resize4d(THFloatTensor *t, long size0, long size1, long size2, long size3);
void THLowpTensor_resize3d(THFloatTensor *t, long size0, long size1, long size2);
void THLowpTensor_mm(THFloatTensor *r_, THFloatTensor *m1, THFloatTensor *m2);

THFloatTensor *Lowp_SpatialConvolution_updateOutput(struct module *module, THFloatTensor *input);
THFloatTensor *Lowp_SpatialMaxPooling_updateOutput(struct module *module, THFloatTensor *input);
THFloatTensor *Lowp_Threshold_updateOutput(struct module *module, THFloatTensor *input);
THFloatTensor *Lowp_SoftMax_updateOutput(struct module *module, THFloatTensor *input);
