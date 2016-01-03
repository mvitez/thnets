# Can be no, 0 or 1
MEMORYDEBUG = no
# Can be 0 or 1
DEBUG = 0
# Can be 0 or 1
CUDNN = 0

CUDAPATH=/usr/local/cuda
CUDNNPATH=/home/ubuntu/cudnn/cuda

UNAME_P := $(shell uname -p)
CFLAGS = -Wall -c -fopenmp -fPIC
LIBS = -L/opt/OpenBLAS/lib -lopenblas -lm
CC = gcc
VPATH = modules cudnn
LIBOBJS = thload.o thbasic.o thapi.o SpatialConvolutionMM.o SpatialMaxPooling.o Threshold.o \
	View.o SoftMax.o Linear.o Dropout.o SpatialZeroPadding.o Reshape.o SpatialConvolution.o

ifneq ($(filter arm%,$(UNAME_P)),)
	CFLAGS += -D__NEON__ -mcpu=cortex-a9 -mfpu=neon
endif
ifneq ($(filter aarc%,$(UNAME_P)),)
	CFLAGS += -D__NEON__ -mcpu=cortex-a53 -mfpu=neon-vfpv4 -DHAVEFP16 -mfp16-format=ieee
endif

ifneq ($(filter x86%,$(UNAME_P)),)
#	CFLAGS += -DUSE_SSE4_2 -march=corei7	#it's slower!
endif
	
ifneq ($(MEMORYDEBUG),no)
	LIBOBJS += memory.o
	CFLAGS += -DMEMORYDEBUG=$(MEMORYDEBUG)
endif

ifeq ($(DEBUG),1)
	CFLAGS += -g
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
all : thnets.so test

*.o: thnets.h

.c.o:
	$(CC) $(CFLAGS) $<

cudnn_copy.o: cudnn_copy.cu
	nvcc -c -Xcompiler -fPIC -O3 cudnn/cudnn_copy.cu

thnets.so: $(LIBOBJS)
	$(CC) -o $@ $(LIBOBJS) -shared -fopenmp $(LIBS)

test: $(LIBOBJS) test.o images.o
	$(CC) -o $@ test.o images.o thnets.so $(LIBS) -lpng -ljpeg

.PHONY : clean
clean :
	rm -f *.o thnets.so test
