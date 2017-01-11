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


include $(CLEAR_VARS)

LOCAL_CFLAGS :=

LOCAL_MODULE    := libpng
LOCAL_SRC_FILES :=\
	libpng/png.c \
	libpng/pngerror.c \
	libpng/pngget.c \
	libpng/pngmem.c \
	libpng/pngpread.c \
	libpng/pngread.c \
	libpng/pngrio.c \
	libpng/pngrtran.c \
	libpng/pngrutil.c \
	libpng/pngset.c \
	libpng/pngtrans.c \
	libpng/pngwio.c \
	libpng/pngwrite.c \
	libpng/pngwtran.c \
	libpng/pngwutil.c

LOCAL_SHARED_LIBRARIES := -lz
include $(BUILD_STATIC_LIBRARY)


include $(CLEAR_VARS)

LOCAL_MODULE := thnets

LOCAL_CFLAGS += -fopenmp -DNOJPEG

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

LOCAL_C_INCLUDES := $(LOCAL_PATH)/libpng/ $(LOCAL_PATH)/libjpeg/ $(LOCAL_PATH)/OpenBLAS-stripped/

LOCAL_STATIC_LIBRARIES := OpenBLAS JPEG-Turbo libpng

include $(BUILD_EXECUTABLE)

