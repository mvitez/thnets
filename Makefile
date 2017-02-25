#Variables that users can change

# Can be no, 0 or 1
MEMORYDEBUG = no
# Can be 0 or 1
DEBUG = 0
OPENCL = 0
LOWP = 0
PENRYN = 0
SANDYBRIDGE = 0
HASWELL = 0
USEBLAS = 0
#Can be no, 4 or 5 (version)
CUDNN = no

CUDAPATH=/usr/local/cuda
CUDAPATH2=/usr/local/cuda/targets/aarch64-linux
OPENBLASPATH=/opt/OpenBLAS/lib
#End of variables users can change

UNAME_M := $(shell uname -m)
UNAME_S := $(shell uname -s)
CFLAGS = -Wall -c -fPIC
CPPFLAGS = -Wall -c -fPIC -std=c++11
LIBS = -lm
CC = gcc
CXX = g++
VPATH = modules cudnn OpenBLAS-stripped opencl lowp lowp/gemmlowp/eight_bit_int_gemm \
	OpenBLAS-stripped/generic
LIBOBJS = thload.o thbasic.o thapi.o SpatialConvolutionMM.o SpatialMaxPooling.o Threshold.o \
	View.o SoftMax.o Linear.o Dropout.o SpatialZeroPadding.o Reshape.o SpatialConvolution.o \
	Normalize.o SpatialFullConvolution.o SpatialMaxUnpooling.o SpatialBatchNormalization.o \
	SpatialAveragePooling.o Sequential.o Concat.o ConcatTable.o JoinTable.o CAddTable.o \
	PReLU.o pytorch.o

ifeq ($(UNAME_S),Darwin)
	CFLAGS += -DACCELERATE
	USEBLAS = 1
else
	CFLAGS += -fopenmp
	CPPFLAGS += -fopenmp
	LDFLAGS += -fopenmp
endif

ifneq ($(USEBLAS),1)

	LIBOBJS += sgemm.o sger.o sgemv.o gemm_beta.o gemv_t.o copy.o
#ARM 32 bit
ifneq ($(filter arm%,$(UNAME_M)),)
	CFLAGS += -DARM -D__NEON__ -mcpu=cortex-a9 -mfpu=neon -DHAVEFP16 -mfp16-format=ieee
	LIBOBJS += axpy_vfp.o sgemm_kernel_4x4_vfpv3.o sgemm_ncopy_4_vfp.o sgemm_tcopy_4_vfp.o
	VPATH += OpenBLAS-stripped/arm
endif

#ARM 64 bit
ifneq ($(filter aarc%,$(UNAME_M)),)
	CFLAGS += -DARM -DHAVEFP16
	CUFLAGS += -DHAVEHALF --gpu-architecture=compute_53
	LIBOBJS += axpy.o sgemm_kernel_4x4.o gemm_ncopy_4.o gemm_tcopy_4.o
	VPATH += OpenBLAS-stripped/arm64
endif

#Intel 64 bit
ifneq ($(filter x86_64%,$(UNAME_M)),)

ifeq ($(PENRYN)$(SANDYBRIDGE)$(HASWELL),000)
	PENRYN=1
endif

	CFLAGS += -DX86_64
	CPPFLAGS += -DX86_64
	LIBOBJS += axpy_sse2.o
ifeq ($(PENRYN),1)
	CFLAGS += -DPENRYN
	CPPFLAGS += -DPENRYN
	LIBOBJS += gemm_kernel_8x4_penryn.o gemm_ncopy_8.o gemm_tcopy_8.o gemm_ncopy_4_penryn.o gemm_tcopy_4_penryn.o
endif
ifeq ($(SANDYBRIDGE),1)
	CFLAGS += -DSANDYBRIDGE
	CPPFLAGS += -DSANDYBRIDGE
	LIBOBJS += sgemm_kernel_16x4_sandy.o gemm_ncopy_16.o gemm_tcopy_16.o gemm_ncopy_4.o gemm_tcopy_4.o
endif
ifeq ($(HASWELL),1)
	CFLAGS += -DHASWELL
	CPPFLAGS += -DHASWELL
	LIBOBJS += sgemm_kernel_16x4_haswell.o gemm_ncopy_16.o gemm_tcopy_16.o gemm_ncopy_4.o gemm_tcopy_4.o
endif

	VPATH += OpenBLAS-stripped/x86_64
endif

else #USEBLAS

	CFLAGS += -DUSEBLAS
ifeq ($(UNAME_S),Darwin)
	LIBS += -framework Accelerate
else
	LIBS += -L$(OPENBLASPATH) -lopenblas
endif

#ARM 32 bit
ifneq ($(filter arm%,$(UNAME_M)),)
	CFLAGS += -DARM -D__NEON__ -mcpu=cortex-a9 -mfpu=neon -DHAVEFP16 -mfp16-format=ieee
endif

#ARM 64 bit
ifneq ($(filter aarc%,$(UNAME_M)),)
	CFLAGS += -DARM -DHAVEFP16
	CUFLAGS += -DHAVEHALF --gpu-architecture=compute_53
endif

#Intel 64 bit
ifneq ($(filter x86_64%,$(UNAME_M)),)
	CFLAGS += -DX86_64
	CPPFLAGS += -DX86_64
endif

endif #USEBLAS

#Memory leaks debugging
ifneq ($(MEMORYDEBUG),no)
	LIBOBJS += memorydebug.o
	CFLAGS += -DMEMORYDEBUG=$(MEMORYDEBUG)
endif

#Debug or release compilation
ifeq ($(DEBUG),1)
	CFLAGS += -g
	CUFLAGS += -g
	CFLAGS += -g
	ASFLAGS += -g
else
	CFLAGS += -O3
	CUFLAGS += -O3
	CPPFLAGS += -O3
endif

#CUDNN
ifeq ($(CUDNN),$(filter $(CUDNN),4 5))
	LIBOBJS += cudnn_basic.o cudnn_SpatialConvolution.o cunn_SpatialMaxPooling.o cudnn_Threshold.o \
		cudnn_SoftMax.o cudnn_copy.o cunn_SpatialMaxUnpooling.o cudnn_SpatialBatchNormalization.o \
		cunn_SpatialFullConvolution.o
	CFLAGS += -DCUDNN -I$(CUDAPATH)/include -I$(CUDAPATH2)/include
	LIBS += -L$(CUDAPATH2)/lib -L$(CUDAPATH2)/lib -lcudart -lcudnn -lcublas
ifeq ($(CUDNN),5)
	CFLAGS += -DCUDNN5
endif

endif

#OpenCL
ifeq ($(OPENCL),1)
	LIBOBJS += opencl_basic.o opencl_SpatialConvolution.o opencl_SpatialMaxPooling.o \
		opencl_Threshold.o opencl_SoftMax.o opencl_img.o
	CFLAGS += -DOPENCL
	LIBS += -lOpenCL
endif

#Lowp
ifeq ($(LOWP),1)
	LIBOBJS += lowp_basic.o lowp_SpatialConvolution.o lowp_SpatialMaxPooling.o \
		lowp_Threshold.o lowp_SoftMax.o gemm8.o eight_bit_int_gemm.o
	CFLAGS += -DLOWP
endif

all : libthnets.so test

*.o: thnets.h

.c.o:
	$(CC) $(CFLAGS) $<

%.o: %.cu
	nvcc -c -Xcompiler -fPIC $(CUFLAGS) -DCUDNN $<

libthnets.so: $(LIBOBJS)
	$(CXX) -o $@ $(LIBOBJS) -shared $(LDFLAGS) $(LIBS)

test: $(LIBOBJS) test.o images.o
	$(CC) -o $@ test.o images.o libthnets.so $(LDFLAGS) $(LIBS)

clean :
	rm -f *.o libthnets.so test

install:
	cp libthnets.so /usr/local/lib
	cp thnets.h thvector.h /usr/local/include

uninstall:
	rm /usr/local/lib/libthnets.so
	rm /usr/local/include/thnets.h /usr/local/include/thvector.h
