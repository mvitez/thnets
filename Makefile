# Can be no, 0 or 1
MEMORYDEBUG = no
# Can be 0 or 1
DEBUG = 0
# Can be 0 or 1
CUDNN = 0

CUDAPATH=/usr/local/cuda
CUDNNPATH=/home/ubuntu/cudnn/cuda
OPENBLASPATH=/opt/OpenBLAS/lib

UNAME_P := $(shell uname -p)
CFLAGS = -Wall -c -fopenmp -fPIC
LIBS = -lm
CC = gcc
VPATH = modules cudnn OpenBLAS-stripped
LIBOBJS = thload.o thbasic.o thapi.o SpatialConvolutionMM.o SpatialMaxPooling.o Threshold.o \
	View.o SoftMax.o Linear.o Dropout.o SpatialZeroPadding.o Reshape.o SpatialConvolution.o \
	sgemm.o sger.o sgemv.o gemm_beta.o gemv_t.o copy.o

ifneq ($(filter arm%,$(UNAME_P)),)
	CFLAGS += -DARM -D__NEON__ -mcpu=cortex-a9 -mfpu=neon
	LIBOBJS += axpy_vfp.o sgemm_kernel_4x4_vfpv3.o sgemm_ncopy_4_vfp.o sgemm_tcopy_4_vfp.o
	VPATH += OpenBLAS-stripped/arm
endif
ifneq ($(filter aarc%,$(UNAME_P)),)
	CFLAGS += -DARM -D__NEON__ -mcpu=cortex-a53 -mfpu=neon-vfpv4 -DHAVEFP16 -mfp16-format=ieee
	LIBOBJS += axpy_vfp.o sgemm_kernel_4x4_vfpv3.o sgemm_ncopy_4_vfp.o sgemm_tcopy_4_vfp.o
	VPATH += OpenBLAS-stripped/arm
endif
ifneq ($(filter x86_64%,$(UNAME_P)),)
	CFLAGS += -DX86_64
	LIBOBJS += axpy_sse.o gemm_kernel_8x4_penryn.o gemm_ncopy_4.o gemm_tcopy_4.o \
		gemm_ncopy_8.o gemm_tcopy_8.o
	VPATH += OpenBLAS-stripped/x86_64
endif

ifeq ($(filter x86_64% arm% aarc%,$(UNAME_P)),)
	CFLAGS += -DUSEBLAS
	LIBS += -L$(OPENBLASPATH) -lopenblas
endif
	
ifneq ($(MEMORYDEBUG),no)
	LIBOBJS += memory.o
	CFLAGS += -DMEMORYDEBUG=$(MEMORYDEBUG)
endif

ifeq ($(DEBUG),1)
	CFLAGS += -g
	ASFLAGS += -g
else
	CFLAGS += -O3
endif

ifeq ($(CUDNN),1)
	LIBOBJS += cudnn_basic.o cudnn_SpatialConvolution.o cudnn_SpatialMaxPooling.o cudnn_Threshold.o \
		cudnn_SoftMax.o cudnn_copy.o
	CFLAGS += -DCUDNN -I$(CUDAPATH)/include -I$(CUDNNPATH)/include
	LIBS += -L$(CUDAPATH)/lib -L$(CUDNNPATH)/lib -lcudart -lcudnn
endif

.PHONY : all
all : libthnets.so test

*.o: thnets.h

.c.o:
	$(CC) $(CFLAGS) $<

cudnn_copy.o: cudnn_copy.cu
	nvcc -c -Xcompiler -fPIC -O3 cudnn/cudnn_copy.cu

libthnets.so: $(LIBOBJS)
	$(CC) -o $@ $(LIBOBJS) -shared -fopenmp $(LIBS)

test: $(LIBOBJS) test.o images.o
	$(CC) -o $@ test.o images.o libthnets.so $(LIBS) -lpng -ljpeg

.PHONY : clean
clean :
	rm -f *.o libthnets.so test

install:
	sudo cp libthnets.so /usr/local/lib
	sudo cp thnets.h thvector.h /usr/local/include

uninstall:
	sudo rm /usr/local/lib/libthnets.so
	sudo rm /usr/local/include/thnets.h /usr/local/include/thvector.h
