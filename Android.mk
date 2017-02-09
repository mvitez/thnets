LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

LOCAL_MODULE    := OpenBLAS
LOCAL_SRC_FILES :=\
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

LOCAL_CFLAGS += -Wall -c -fopenmp -fPIC -DARM -D__NEON__ -mcpu=cortex-a9 -mfpu=neon -O3
include $(BUILD_STATIC_LIBRARY)
 


include $(CLEAR_VARS)

LOCAL_MODULE := thnets

LOCAL_CFLAGS += -fopenmp

LOCAL_SRC_FILES := thload.c thbasic.c thapi.c images.c test.c \
	modules/CAddTable.c \
	modules/Concat.c \
	modules/ConcatTable.c \
	modules/Dropout.c \
	modules/JoinTable.c \
	modules/Linear.c \
	modules/Normalize.c \
	modules/PReLu.c \
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
	modules/View.c

LOCAL_LDLIBS := -lz -landroid -lm -llog

LOCAL_C_INCLUDES := $(LOCAL_PATH)/OpenBLAS-stripped/

LOCAL_STATIC_LIBRARIES := OpenBLAS

include $(BUILD_EXECUTABLE)

