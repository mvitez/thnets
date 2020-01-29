#include <float.h>
#include <stdlib.h>
#include "thvector.h"
#ifdef MEMORYDEBUG
#include "memorydebug.h"
#endif

#ifdef OPENCL
#include "CL/opencl.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum therror {
	ERR_OPENFILE = -1,
	ERR_READFILE = -2,
	ERR_NOTIMPLEMENTED = -3,
	ERR_CORRUPTED = -4,
	ERR_WRONGOBJECT = -5
};

enum thtype {
	TYPE_NIL = 0,
	TYPE_NUMBER = 1,
	TYPE_STRING = 2,
	TYPE_TABLE = 3,
	TYPE_TORCH = 4,
	TYPE_BOOLEAN = 5,
	TYPE_FUNCTION = 6,
	LEGACY_TYPE_RECUR_FUNCTION = 7,
	TYPE_RECUR_FUNCTION = 8,
	TYPE_BYTE = 100,
	TYPE_CHAR = 101,
	TYPE_SHORT = 102,
	TYPE_INT = 103,
	TYPE_LONG = 104,
	TYPE_FLOAT = 105,
	TYPE_DOUBLE = 106,
	TYPE_STORAGE = 200,
	TYPE_TENSOR = 201,
	TYPE_NNMODULE = 202
};

struct thobject;
struct threcord;

struct table {
	int idx;
	int nrefs;
	int nelem;
	struct threcord *records;
};

struct nnmodule {
	int idx;
	int nrefs;
	char *name;
	struct table *table;
};

struct storage {
	int idx;
	int nrefs;
	int scalartype;
	long nelem;
	void *data;
};

struct tensor {
	int idx;
	int nrefs;
	int scalartype;
	int ndim;
	long *size;
	long *stride;
	long storageoffset;
	struct storage *storage;
};

struct thobject
{
	int type;
	union {
		double number;
		struct {
			int size;
			char *data;
		} string;
		struct table *table;
		struct storage *storage;
		struct tensor *tensor;
		struct nnmodule *nnmodule;
		int boolean;
	};
};

struct threcord {
	struct thobject name;
	struct thobject value;
};

typedef struct THNStorage
{
	float *data;
	int nref, mustfree;	// mustfree = 0 (allocated somewhere else), 1 (free), 2 (cuda free)
} THNStorage;

typedef struct THNTensor
{
	long size[5];
	long stride[5];
	int nDimension;
	THNStorage *storage;
	long storageOffset;
#ifdef LOWP
	float sub, mult;
#endif
} THNTensor;

struct SpatialConvolution
{
	THNTensor *bias, *weight, *finput;
	int dW, dH, dZ, padW, padH, padZ, kW, kH, kZ, nInputPlane, nOutputPlane;
	int refl_pad;
	int padW2, padH2, padZ2; // right and bottom, if different
	int autopad; // ONNX, 0 = VALID, 1 = SAME_UPPER, 2 = SAME_LOWER
	int dlH, dlW, dlZ; // Dilations
};

struct SpatialFullConvolution
{
	THNTensor *bias, *weight;
	int dW, dH, dZ, padW, padH, padZ, kW, kH, kZ, nInputPlane, nOutputPlane;
	int adjW, adjH, adjZ;
	THNTensor *ones, *columns;
};

struct SpatialMaxPooling
{
	int padW, padH, padZ, dW, dH, dZ, kW, kH, kZ, ceil_mode;
	int iwidth, iheight;
	THNTensor *indices;
	int padW2, padH2, padZ2; // right and bottom, if different
	int autopad; // ONNX, 0 = VALID, 1 = SAME_UPPER, 2 = SAME_LOWER
};

struct SpatialAveragePooling
{
	int padW, padH, padZ, dW, dH, dZ, kW, kH, kZ, ceil_mode;
	int count_include_pad;
	int padW2, padH2, padZ2; // right and bottom, if different
	int autopad; // ONNX, 0 = VALID, 1 = SAME_UPPER, 2 = SAME_LOWER
};

struct Linear
{
	THNTensor *bias, *weight, *addBuffer;
	int commute;	// Used for ONNX, if 1, invert A and B
};

struct Threshold
{
	float threshold, val, alpha;
	int inplace;
};

struct View
{
	int numElements, nDimension;
	long size[5];
};

struct Dropout
{
	float p;
	int inplace, v2;
};

struct SpatialZeroPadding
{
	int pad_l, pad_r, pad_t, pad_b;
};

struct Reshape
{
	int numElements, batchMode;
	long size[5], batchsize[5];
	int nsize, nbatchsize;
};

struct SpatialMaxUnpooling
{
	struct nnmodule *pooling;
};

struct SpatialBatchNormalization
{
	THNTensor *running_mean, *running_var, *weight, *bias;
	double eps;
};

struct Concat
{
	struct network *net;
	int dimension;
};

struct Sequential
{
	struct network *net;
};

struct PReLU
{
	THNTensor *weight;
	int nOutputPlane;
};

struct Padding
{
	float dim, pad, nInputDim, index, value;
};

struct Slice
{
	int axis, from, to;
};

struct Upsample
{
	float width_scale, height_scale;
};

struct LSTM
{
	THNTensor *W, *R, *B;
};

struct GRU
{
	THNTensor *W, *R, *B;
};

struct Squeeze
{
	int naxes;
	int axes[4];
};

enum moduletype {
	MT_Nil,
	MT_SpatialConvolutionMM,
	MT_SpatialConvolutionVirtMM,
	MT_SpatialConvolution,
	MT_SpatialMaxPooling,
	MT_SpatialAveragePooling,
	MT_Linear,
	MT_SoftMax,
	MT_Threshold,
	MT_View,
	MT_Dropout,
	MT_SpatialZeroPadding,
	MT_Reshape,
	MT_Normalize,
	MT_SpatialFullConvolution,
	MT_SpatialMaxUnpooling,
	MT_SpatialBatchNormalization,
	MT_Sequential,
	MT_Concat,
	MT_ConcatTable,
	MT_JoinTable,
	MT_CAddTable,
	MT_CSubTable,
	MT_PReLU,
	MT_Identity,
	MT_Padding,
	MT_LogSoftMax,
	MT_Slice,
	MT_Cmax,
	MT_Upsample,
	MT_LSTM,
	MT_GRU,
	MT_Squeeze,
	MT_Unsqueeze,
	MT_Sigmoid,
	MT_Tanh,
	MT_Transpose,
	MT_DepthwiseConvolution,
	MT_CMulTable,
	MT_Elu
};

struct network;

struct module
{
	int type;
	THNTensor *(*updateOutput)(struct module *m, THNTensor *in);
	void (*nnfree)(struct module *m);
	THNTensor *output;
	struct network *net;
	struct nnmodule *nnmodule;
#ifdef OPENCL
	cl_kernel kernel;
	int clstatus;
#endif
	// These are currently used only by ONNX
	// They are always present in order not to require to define ONNX
	// when including this header
	char *outputname;
	int ninputs;
#define MAXMODULEINPUTS 16
	int inputs[MAXMODULEINPUTS];
	char *inputnames[MAXMODULEINPUTS];
	// End ONNX
	union {
		struct SpatialConvolution SpatialConvolution;
		struct SpatialMaxPooling SpatialMaxPooling;
		struct SpatialAveragePooling SpatialAveragePooling;
		struct Linear Linear;
		struct Threshold Threshold;
		struct View View;
		struct Dropout Dropout;
		struct SpatialZeroPadding SpatialZeroPadding;
		struct Reshape Reshape;
		struct SpatialFullConvolution SpatialFullConvolution;
		struct SpatialMaxUnpooling SpatialMaxUnpooling;
		struct SpatialBatchNormalization SpatialBatchNormalization;
		struct Sequential Sequential;
		struct Concat Concat;
		struct Sequential ConcatTable;
		struct Concat JoinTable;
		struct PReLU PReLU;
		struct Slice Slice;
		struct Upsample Upsample;
		struct LSTM LSTM;
		struct GRU GRU;
		struct Squeeze Squeeze;
	};
};

// ONNX stuff
#ifdef ONNX
struct network *loadonnx(const char *path);
int onnx_isinitializer(const void *graph, int nodeidx, int inputidx);
THNTensor *onnx_gettensor(const void *graph, int nodeidx, int inputidx);
THNTensor *onnx_getshapetensor(const void *graph, int nodeidx, int inputidx);
int onnx_getint(const void *graph, int nodeidx, const char *attrname, int idx);
float onnx_getfloat(const void *graph, int nodeidx, const char *attrname, int idx);
const char *onnx_getstring(const void *graph, int nodeidx, const char *attrname, int idx);
void onnx_printintslist(const void *graph, int nodeidx, const char *name);
#endif
// End ONNX

enum {ENGINE_CPU, ENGINE_CUDA, ENGINE_OPENCL, ENGINE_OPENCLINIT, ENGINE_LOWP};

struct network
{
	int nelem, engine;
	struct module *modules;
};

struct object2module
{
	const char *name;
	int (*func)(struct module *mod, struct nnmodule *n);
};

extern struct object2module object2module[];

double TableGetNumber(struct table *t, const char *name);
int TableGetBoolean(struct table *t, const char *name);
THNTensor *TableGetTensor(struct table *t, const char *name);
void *TableGetStorage(struct table *t, const char *name, int *nelem);
struct nnmodule *TableGetNNModule(struct table *t, const char *name);
void THError(const char *fmt, ...);
THNTensor *THNTensor_new();
THNStorage *THNStorage_new(long size);
THNStorage *THNStorage_newwithbuffer(void *buffer);
THNTensor *THNTensor_newWithStorage1d(THNStorage *storage, long storageOffset, long size0, long stride0);
THNTensor *THNTensor_newWithStorage2d(THNStorage *storage, long storageOffset, long size0, long stride0, long size1, long stride1);
THNTensor *THNTensor_newWithStorage3d(THNStorage *storage, long storageOffset, long size0, long stride0, long size1, long stride1, long size2, long stride2);
THNTensor *THNTensor_newWithTensor(THNTensor *tensor);
void THNTensor_transpose(THNTensor *tdst, THNTensor *tsrc, int dimension1, int dimension2);
THNTensor *THNTensor_newTranspose(THNTensor *tensor, int dimension1_, int dimension2_);
float *THNTensor_data(THNTensor *tensor);
int THNTensor_isSameSizeAs(const THNTensor *self, const THNTensor* src);
void THNTensor_resize(THNTensor *t, long *size, int nDimension);
void THNTensor_resize4d(THNTensor *t, long size0, long size1, long size2, long size3);
void THNTensor_resize3d(THNTensor *t, long size0, long size1, long size2);
void THNTensor_resize2d(THNTensor *t, long size0, long size1);
void THNTensor_resize1d(THNTensor *t, long size0);
void THNTensor_resizeAs(THNTensor *tdst, THNTensor *tsrc);
long THNTensor_nElement(THNTensor *t);
void THNTensor_set(THNTensor *tdst, THNTensor *tsrc);
void THNTensor_zero(THNTensor *t);
void THNTensor_fill(THNTensor *t, float value);
void THNTensor_copy(THNTensor *tdst, THNTensor *tsrc);
void THNTensor_safecopy(THNTensor *tdst, THNTensor *tsrc);
void THNTensor_slice(THNTensor *dst, THNTensor *src, int dimension, long from, long to);
void THNTensor_free(THNTensor *t);
THNTensor *THNTensor_newSelect(THNTensor *tensor, int dimension, long sliceIndex);
THNTensor *THNTensor_squeeze(THNTensor *t);
double THExpMinusApprox(double x);
void THBlas_gemm(char transa, char transb, long m, long n, long k, float alpha, float *a, long lda, float *b, long ldb, float beta, float *c, long ldc);
void THNTensor_addmm(THNTensor *r_, float beta, THNTensor *t, float alpha, THNTensor *m1, THNTensor *m2);
void THNTensor_convmm(THNTensor *r, float beta, float alpha, THNTensor *filt, THNTensor *m,
	int kH, int kW, int dH, int dW, int padH, int padW);
void THNTensor_addr(THNTensor *r_, float beta, THNTensor *t, float alpha, THNTensor *vec1, THNTensor *vec2);
void THNTensor_addmv(THNTensor *r_, float beta, THNTensor *t, float alpha, THNTensor *mat, THNTensor *vec);
void THNTensor_conv2Dmm(THNTensor *r_, float beta, float alpha, THNTensor *t_, THNTensor *k_, long srow, long scol, const char *vf, const char *xc);
void THNTensor_conv2Dmv(THNTensor *r_, float beta, float alpha, THNTensor *t_, THNTensor *k_, long srow, long scol, const char *vf, const char *xc);

#define thfmaxf(a,b) ((a) > (b) ? (a) : (b))
#define thfminf(a,b) ((a) < (b) ? (a) : (b))

#define THInf FLT_MAX

#ifdef HAVEFP16
void tofp16(__fp16 *dst, const float *src, size_t len);
void fromfp16(float *dst, const __fp16 *src, size_t len);
#endif

int loadtorch(const char *path, struct thobject *obj, int longsize);
int printobject(struct thobject *obj, int indent);
int freeobject(struct thobject *obj);
void freemodule(struct module *m);
void freenetwork(struct network *net);
THNTensor *forward(struct network *net, THNTensor *in);
THNTensor *THNTensor_newFromObject(struct thobject *obj);
struct network *Module2Network(struct nnmodule *obj);
void printtensor(THNTensor *t);
void blas_init();
double th_seconds();

THNTensor *nn_SpatialConvolutionMM_updateOutput(struct module *module, THNTensor *input);
THNTensor *nn_SpatialConvolution_updateOutput(struct module *module, THNTensor *input);
THNTensor *nn_SpatialMaxPooling_updateOutput(struct module *module, THNTensor *input);
THNTensor *nn_SpatialAveragePooling_updateOutput(struct module *module, THNTensor *input);
THNTensor *nn_Threshold_updateOutput(struct module *module, THNTensor *input);
THNTensor *nn_View_updateOutput(struct module *module, THNTensor *input);
THNTensor *nn_SoftMax_updateOutput(struct module *module, THNTensor *input);
THNTensor *nn_Linear_updateOutput(struct module *module, THNTensor *input);
THNTensor *nn_Dropout_updateOutput(struct module *module, THNTensor *input);
THNTensor *nn_SpatialZeroPadding_updateOutput(struct module *module, THNTensor *input);
THNTensor *nn_Reshape_updateOutput(struct module *module, THNTensor *input);
THNTensor *nn_Normalize_updateOutput(struct module *module, THNTensor *input);
THNTensor *nn_SpatialFullConvolution_updateOutput(struct module *module, THNTensor *input);
THNTensor *nn_SpatialMaxUnpooling_updateOutput(struct module *module, THNTensor *input);
THNTensor *nn_SpatialBatchNormalization_updateOutput(struct module *module, THNTensor *input);
THNTensor *nn_Sequential_updateOutput(struct module *module, THNTensor *input);
THNTensor *nn_Concat_updateOutput(struct module *module, THNTensor *input);
THNTensor *nn_ConcatTable_updateOutput(struct module *module, THNTensor *input);
THNTensor *nn_JoinTable_updateOutput(struct module *module, THNTensor *input);
THNTensor *nn_CAddTable_updateOutput(struct module *module, THNTensor *input);
THNTensor *nn_PReLU_updateOutput(struct module *module, THNTensor *input);
THNTensor *nn_Identity_updateOutput(struct module *module, THNTensor *input);
THNTensor *nn_LogSoftMax_updateOutput(struct module *module, THNTensor *input);
THNTensor *nn_Slice_updateOutput(struct module *module, THNTensor *input);
THNTensor *nn_Cmax_updateOutput(struct module *module, THNTensor *input);
THNTensor *nn_Sigmoid_updateOutput(struct module *module, THNTensor *input);
THNTensor *nn_Tanh_updateOutput(struct module *module, THNTensor *input);
THNTensor *nn_Upsample_updateOutput(struct module *module, THNTensor *input);

int nnload_SpatialConvolution(struct module *mod, struct nnmodule *n);
int nnload_SpatialMaxPooling(struct module *mod, struct nnmodule *n);
int nnload_SpatialAveragePooling(struct module *mod, struct nnmodule *n);
int nnload_Threshold(struct module *mod, struct nnmodule *n);
int nnload_View(struct module *mod, struct nnmodule *n);
int nnload_SoftMax(struct module *mod, struct nnmodule *n);
int nnload_Linear(struct module *mod, struct nnmodule *n);
int nnload_Dropout(struct module *mod, struct nnmodule *n);
int nnload_SpatialZeroPadding(struct module *mod, struct nnmodule *n);
int nnload_Reshape(struct module *mod, struct nnmodule *n);
int nnload_Normalize(struct module *mod, struct nnmodule *n);
int nnload_SpatialFullConvolution(struct module *mod, struct nnmodule *n);
int nnload_SpatialMaxUnpooling(struct module *mod, struct nnmodule *n);
int nnload_SpatialBatchNormalization(struct module *mod, struct nnmodule *n);
int nnload_Sequential(struct module *mod, struct nnmodule *n);
int nnload_Concat(struct module *mod, struct nnmodule *n);
int nnload_ConcatTable(struct module *mod, struct nnmodule *n);
int nnload_JoinTable(struct module *mod, struct nnmodule *n);
int nnload_CAddTable(struct module *mod, struct nnmodule *n);
int nnload_PReLU(struct module *mod, struct nnmodule *n);
int nnload_Identity(struct module *mod, struct nnmodule *n);
int nnload_LogSoftMax(struct module *mod, struct nnmodule *n);

void onnxload_SpatialConvolution(const void *graph, struct module *m, int nodeidx);
void onnxload_SpatialConvolutionTransposed(const void *graph, struct module *m, int nodeidx);
void onnxload_Linear(const void *graph, struct module *m, int nodeidx);
void onnxload_SpatialBatchNormalization(const void *graph, struct module *m, int nodeidx);
void onnxload_SpatialMaxPooling(const void *graph, struct module *m, int nodeidx);
void onnxload_SpatialAveragePooling(const void *graph, struct module *m, int nodeidx);
void onnxload_Threshold(const void *graph, struct module *m, int nodeidx);
void onnxload_PReLU(const void *graph, struct module *m, int nodeidx);
void onnxload_Dropout(const void *graph, struct module *m, int nodeidx);
void onnxload_SoftMax(const void *graph, struct module *m, int nodeidx);
void onnxload_LogSoftMax(const void *graph, struct module *m, int nodeidx);
void onnxload_View(const void *graph, struct module *m, int nodeidx);
void onnxload_Add(const void *graph, struct module *m, int nodeidx);
void onnxload_Sub(const void *graph, struct module *m, int nodeidx);
void onnxload_Concat(const void *graph, struct module *m, int nodeidx);
void onnxload_Slice(const void *graph, struct module *m, int nodeidx);
void onnxload_Cmax(const void *graph, struct module *m, int nodeidx);
void onnxload_Sigmoid(const void *graph, struct module *m, int nodeidx);
void onnxload_Tanh(const void *graph, struct module *m, int nodeidx);
void onnxload_Upsample(const void *graph, struct module *m, int nodeidx);

/* High level API */

typedef struct thnetwork
{
	struct thobject *netobj;
	struct thobject *statobj;
	struct network *net;
	THNTensor *out;
	float mean[3], std[3];
} THNETWORK;

void THInit();
THNETWORK *THLoadNetwork(const char *path);
THNTensor *THForward(THNETWORK *net, THNTensor *in);
void THMakeSpatial(THNETWORK *network, int size);
int THProcessFloat(THNETWORK *network, float *data, int batchsize, int width, int height, int nplanes, float **result, int *outwidth, int *outheight);
int THProcessImages(THNETWORK *network, unsigned char **images, int batchsize, int width, int height, int stride, float **result, int *outwidth, int *outheight, int bgr);
int THProcessYUYV(THNETWORK *network, unsigned char *image, int width, int height, float **results, int *outwidth, int *outheight);
THNETWORK *THCreateCudaNetwork(THNETWORK *net);
THNETWORK *THCreateOpenCLNetwork(THNETWORK *net);
THNETWORK *THCreateLowpNetwork(THNETWORK *net, float range);
int THCudaHalfFloat(int enable);
int THOpenCLHalfFloat(int enable);
int THUseSpatialConvolutionMM(THNETWORK *network, int mm_type);
void THFreeNetwork(THNETWORK *network);
int THLastError();
extern int th_debug, th_profile, th_minmax;
extern double th_convtot, th_convflops;

#ifdef CUDNN
#include "cudnn/cudnn_th.h"
#endif

#ifdef OPENCL
#include "opencl/opencl_th.h"
#endif

#ifdef LOWP
#include "lowp/lowp.h"
#endif

#ifdef USEQSML
void init_thnets4qsml_conv(THNETWORK *network);
void transform_mem(struct module newmod, int col, int row, int plane, int outp);
float* transform_mem_input(float* in1, int col, int row, int plane);
#endif

#ifdef __cplusplus
}
#endif
