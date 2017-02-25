LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

LOCAL_CFLAGS += -Wall -c -fopenmp -fPIC -DARM -D__NEON__ -mcpu=cortex-a9 -mfpu=neon -O3

LOCAL_MODULE := thnets

LOCAL_SRC_FILES := thload.c thbasic.c thapi.c pytorch.c images.c test.c \
	modules/CAddTable.c \
	modules/Concat.c \
	modules/ConcatTable.c \
	modules/Dropout.c \
	modules/JoinTable.c \
	modules/Linear.c \
	modules/Normalize.c \
	modules/PReLU.c \
	modules/Reshape.c \
	modules/Sequential.c \
	modules/SoftMax.c \
	modules/SpatialAveragePooling.c \
	modules/SpatialBatchNormalization.c \
	modules/SpatialConvolution.c \
	modules/SpatialConvolutionMM.c \
	modules/SpatialFullConvolution.c \
	modules/SpatialMaxPooling.c \
	modules/SpatialMaxUnpooling.c \
	modules/SpatialZeroPadding.c \
	modules/Threshold.c \
	modules/View.c \
	OpenBLAS-stripped/sgemm.c \
	OpenBLAS-stripped/sger.c \
	OpenBLAS-stripped/sgemv.c \
	OpenBLAS-stripped/gemm_beta.c \
	OpenBLAS-stripped/gemv_t.c \
	OpenBLAS-stripped/copy.c \
	OpenBLAS-stripped/arm/axpy_vfp.S \
	OpenBLAS-stripped/arm/sgemm_kernel_4x4_vfpv3.S \
	OpenBLAS-stripped/arm/sgemm_ncopy_4_vfp.S \
	OpenBLAS-stripped/arm/sgemm_tcopy_4_vfp.S

LOCAL_LDLIBS := -landroid -lm -llog

include $(BUILD_EXECUTABLE)
