#CFLAGS = -Wall -c -fopenmp -fPIC -DUSE_OPENMP -DSMP_SERVER -DTHREADED_LEVEL3 -DNN
CFLAGS = -O2 -DMAX_STACK_ALLOC=2048 -fopenmp -Wall -DF_INTERFACE_GFORT -fPIC -DSMP_SERVER -DUSE_OPENMP -DNO_WARMUP -DMAX_CPU_NUMBER=4 -DNO_AFFINITY -marm -mfpu=vfpv3  -mfloat-abi=hard -march=armv7-a
CC = gcc
#OBJS = blas_server_omp.o gemm.o l3_gemm.o sgemm_ncopy_4_vfp.o sgemm_tcopy_4_vfp.o strmm_kernel_4x4_vfpv3.o
OBJS = memory.o blas_server.o openblas_error_handle.o xerbla.o sgemm_beta.o sgemm_kernel.o sgemm_oncopy.o sgemm_otcopy.o \
	sgemm.o sgemm_nn.o sgemm_nt.o sgemm_tn.o sgemm_tt.o sgemm_thread_nn.o sgemm_thread_nt.o sgemm_thread_tn.o sgemm_thread_tt.o \
	sgemv.o sgemv_n.o sgemv_t.o sgemv_thread_n.o sgemv_thread_t.o sscal_k.o sger.o sger_k.o scopy_k.o saxpy_k.o

all: libopenblas.so

memory.o:
	$(CC) $(CFLAGS) -DASMNAME=memory -DASMFNAME=memory_ -DNAME=memory_ -DCNAME=memory -DCHAR_NAME=\"memory_\" -DCHAR_CNAME=\"memory\" -DNO_AFFINITY -c memory.c -o memory.o
blas_server.o:
	$(CC) $(CFLAGS) -DASMNAME=blas_server -DASMFNAME=blas_server_ -DNAME=blas_server_ -DCNAME=blas_server -DCHAR_NAME=\"blas_server_\" -DCHAR_CNAME=\"blas_server\" -DNO_AFFINITY -c blas_server_omp.c -o blas_server.o
openblas_error_handle.o:
	$(CC) $(CFLAGS) -DASMNAME=openblas_error_handle -DASMFNAME=openblas_error_handle_ -DNAME=openblas_error_handle_ -DCNAME=openblas_error_handle -DCHAR_NAME=\"openblas_error_handle_\" -DCHAR_CNAME=\"openblas_error_handle\" -DNO_AFFINITY -c openblas_error_handle.c -o openblas_error_handle.o
xerbla.o:
	$(CC) $(CFLAGS) -DASMNAME=xerbla -DASMFNAME=xerbla_ -DNAME=xerbla_ -DCNAME=xerbla -DCHAR_NAME=\"xerbla_\" -DCHAR_CNAME=\"xerbla\" -DNO_AFFINITY -c xerbla.c -o xerbla.o
sgemm_beta.o:
	$(CC) $(CLFAGS) -DASMNAME=sgemm_beta -DASMFNAME=sgemm_beta_ -DNAME=sgemm_beta_ -DCNAME=sgemm_beta -DCHAR_NAME=\"sgemm_beta_\" -DCHAR_CNAME=\"sgemm_beta\" -DNO_AFFINITY -UDOUBLE  -UCOMPLEX -c -UDOUBLE -UCOMPLEX -I. generic/gemm_beta.c -o sgemm_beta.o	
sgemm_kernel.o:
	$(CC) $(CLFAGS) -DASMNAME=sgemm_kernel -DASMFNAME=sgemm_kernel_ -DNAME=sgemm_kernel_ -DCNAME=sgemm_kernel -DCHAR_NAME=\"sgemm_kernel_\" -DCHAR_CNAME=\"sgemm_kernel\" -DNO_AFFINITY -UDOUBLE  -UCOMPLEX -c -UDOUBLE -UCOMPLEX -I. arm/sgemm_kernel_4x4_vfpv3.S -o sgemm_kernel.o
sgemm_oncopy.o:
	$(CC) $(CLFAGS) -DASMNAME=sgemm_oncopy -DASMFNAME=sgemm_oncopy_ -DNAME=sgemm_oncopy_ -DCNAME=sgemm_oncopy -DCHAR_NAME=\"sgemm_oncopy_\" -DCHAR_CNAME=\"sgemm_oncopy\" -DNO_AFFINITY -UDOUBLE  -UCOMPLEX -c -UDOUBLE -UCOMPLEX -I. arm/sgemm_ncopy_4_vfp.S -o sgemm_oncopy.o
sgemm_otcopy.o:
	$(CC) $(CLFAGS) -DASMNAME=sgemm_otcopy -DASMFNAME=sgemm_otcopy_ -DNAME=sgemm_otcopy_ -DCNAME=sgemm_otcopy -DCHAR_NAME=\"sgemm_otcopy_\" -DCHAR_CNAME=\"sgemm_otcopy\" -DNO_AFFINITY -UDOUBLE  -UCOMPLEX -c -UDOUBLE -UCOMPLEX -I. arm/sgemm_tcopy_4_vfp.S -o sgemm_otcopy.o

sgemm.o:
	$(CC) -c $(CFLAGS) -DASMNAME=sgemm -DASMFNAME=sgemm_ -DNAME=sgemm_ -DCNAME=sgemm -DCHAR_NAME=\"sgemm_\" -DCHAR_CNAME=\"sgemm\" -DNO_AFFINITY -UDOUBLE  -UCOMPLEX gemm1.c -o sgemm.o
sgemm_nn.o:
	$(CC) $(CFLAGS) -DASMNAME=sgemm_nn -DASMFNAME=sgemm_nn_ -DNAME=sgemm_nn_ -DCNAME=sgemm_nn -DCHAR_NAME=\"sgemm_nn_\" -DCHAR_CNAME=\"sgemm_nn\" -DNO_AFFINITY -UDOUBLE  -UCOMPLEX  -c -UDOUBLE -UCOMPLEX -DNN gemm.c -o sgemm_nn.o
sgemm_nt.o:                                                                                                                                                     
	$(CC) $(CFLAGS) -DASMNAME=sgemm_nt -DASMFNAME=sgemm_nt_ -DNAME=sgemm_nt_ -DCNAME=sgemm_nt -DCHAR_NAME=\"sgemm_nt_\" -DCHAR_CNAME=\"sgemm_nt\" -DNO_AFFINITY -UDOUBLE  -UCOMPLEX  -c -UDOUBLE -UCOMPLEX -DNT gemm.c -o sgemm_nt.o
sgemm_tn.o:                                                                                                                                                     
	$(CC) $(CFLAGS) -DASMNAME=sgemm_tn -DASMFNAME=sgemm_tn_ -DNAME=sgemm_tn_ -DCNAME=sgemm_tn -DCHAR_NAME=\"sgemm_tn_\" -DCHAR_CNAME=\"sgemm_tn\" -DNO_AFFINITY -UDOUBLE  -UCOMPLEX  -c -UDOUBLE -UCOMPLEX -DTN gemm.c -o sgemm_tn.o
sgemm_tt.o:                                                                                                                                                     
	$(CC) $(CFLAGS) -DASMNAME=sgemm_tt -DASMFNAME=sgemm_tt_ -DNAME=sgemm_tt_ -DCNAME=sgemm_tt -DCHAR_NAME=\"sgemm_tt_\" -DCHAR_CNAME=\"sgemm_tt\" -DNO_AFFINITY -UDOUBLE  -UCOMPLEX  -c -UDOUBLE -UCOMPLEX -DTT gemm.c -o sgemm_tt.o
sgemm_thread_nn.o:
	$(CC) $(CFLAGS) -DASMNAME=sgemm_thread_nn -DASMFNAME=sgemm_thread_nn_ -DNAME=sgemm_thread_nn_ -DCNAME=sgemm_thread_nn -DCHAR_NAME=\"sgemm_thread_nn_\" -DCHAR_CNAME=\"sgemm_thread_nn\" -DNO_AFFINITY -UDOUBLE  -UCOMPLEX  -c -DTHREADED_LEVEL3 -UDOUBLE -UCOMPLEX -DNN gemm.c -o sgemm_thread_nn.o
sgemm_thread_nt.o:
	$(CC) $(CFLAGS) -DASMNAME=sgemm_thread_nt -DASMFNAME=sgemm_thread_nt_ -DNAME=sgemm_thread_nt_ -DCNAME=sgemm_thread_nt -DCHAR_NAME=\"sgemm_thread_nt_\" -DCHAR_CNAME=\"sgemm_thread_nt\" -DNO_AFFINITY -UDOUBLE  -UCOMPLEX  -c -DTHREADED_LEVEL3 -UDOUBLE -UCOMPLEX -DNT gemm.c -o sgemm_thread_nt.o
sgemm_thread_tn.o:
	$(CC) $(CFLAGS) -DASMNAME=sgemm_thread_tn -DASMFNAME=sgemm_thread_tn_ -DNAME=sgemm_thread_tn_ -DCNAME=sgemm_thread_tn -DCHAR_NAME=\"sgemm_thread_tn_\" -DCHAR_CNAME=\"sgemm_thread_tn\" -DNO_AFFINITY -UDOUBLE  -UCOMPLEX  -c -DTHREADED_LEVEL3 -UDOUBLE -UCOMPLEX -DTN gemm.c -o sgemm_thread_tn.o
sgemm_thread_tt.o:
	$(CC) $(CFLAGS) -DASMNAME=sgemm_thread_tt -DASMFNAME=sgemm_thread_tt_ -DNAME=sgemm_thread_tt_ -DCNAME=sgemm_thread_tt -DCHAR_NAME=\"sgemm_thread_tt_\" -DCHAR_CNAME=\"sgemm_thread_tt\" -DNO_AFFINITY -UDOUBLE  -UCOMPLEX  -c -DTHREADED_LEVEL3 -UDOUBLE -UCOMPLEX -DTT gemm.c -o sgemm_thread_tt.o

sgemv.o:	
	$(CC) $(CFLAGS) -c -DASMNAME=sgemv -DASMFNAME=sgemv_ -DNAME=sgemv_ -DCNAME=sgemv -DCHAR_NAME=\"sgemv_\" -DCHAR_CNAME=\"sgemv\" -DNO_AFFINITY -UDOUBLE  -UCOMPLEX -o sgemv.o gemv.c
sgemv_n.o:
	$(CC) $(CFLAGS) -c -DASMNAME=sgemv_n -DASMFNAME=sgemv_n_ -DNAME=sgemv_n_ -DCNAME=sgemv_n -DCHAR_NAME=\"sgemv_n_\" -DCHAR_CNAME=\"sgemv_n\" -DNO_AFFINITY -I. -UDOUBLE  -UCOMPLEX -UDOUBLE -UCOMPLEX  -UTRANS arm/gemv_n.c -o sgemv_n.o
sgemv_t.o:
	$(CC) $(CFLAGS) -c -DASMNAME=sgemv_t -DASMFNAME=sgemv_t_ -DNAME=sgemv_t_ -DCNAME=sgemv_t -DCHAR_NAME=\"sgemv_t_\" -DCHAR_CNAME=\"sgemv_t\" -DNO_AFFINITY -I. -UDOUBLE  -UCOMPLEX -UDOUBLE -UCOMPLEX  -DTRANS  arm/gemv_t.c -o sgemv_t.o
sgemv_thread_n.o:
	$(CC) $(CFLAGS) -c -DASMNAME=sgemv_thread_n -DASMFNAME=sgemv_thread_n_ -DNAME=sgemv_thread_n_ -DCNAME=sgemv_thread_n -DCHAR_NAME=\"sgemv_thread_n_\" -DCHAR_CNAME=\"sgemv_thread_n\" -DNO_AFFINITY -UDOUBLE  -UCOMPLEX -UCOMPLEX -UDOUBLE  -UTRANSA -UCONJ -UXCONJ gemv_thread.c -o sgemv_thread_n.o
sgemv_thread_t.o:
	$(CC) $(CFLAGS) -c -DASMNAME=sgemv_thread_t -DASMFNAME=sgemv_thread_t_ -DNAME=sgemv_thread_t_ -DCNAME=sgemv_thread_t -DCHAR_NAME=\"sgemv_thread_t_\" -DCHAR_CNAME=\"sgemv_thread_t\" -DNO_AFFINITY -UDOUBLE  -UCOMPLEX -UCOMPLEX -UDOUBLE  -DTRANSA -UCONJ -UXCONJ gemv_thread.c -o sgemv_thread_t.o
sscal_k.o:
	$(CC) $(CFLAGS) -c -DASMNAME=sscal_k -DASMFNAME=sscal_k_ -DNAME=sscal_k_ -DCNAME=sscal_k -DCHAR_NAME=\"sscal_k_\" -DCHAR_CNAME=\"sscal_k\" -DNO_AFFINITY -I. -UDOUBLE  -UCOMPLEX -UCOMPLEX -UDOUBLE arm/scal.c -o sscal_k.o
sger.o:
	$(CC) $(CFLAGS) -c -DASMNAME=sger -DASMFNAME=sger_ -DNAME=sger_ -DCNAME=sger -DCHAR_NAME=\"sger_\" -DCHAR_CNAME=\"sger\" -DNO_AFFINITY -UDOUBLE  -UCOMPLEX ger1.c -o sger.o
sger_k.o:	
	$(CC) $(CFLAGS) -c -DASMNAME=sger_k -DASMFNAME=sger_k_ -DNAME=sger_k_ -DCNAME=sger_k -DCHAR_NAME=\"sger_k_\" -DCHAR_CNAME=\"sger_k\" -DNO_AFFINITY -I. -UDOUBLE  -UCOMPLEX -UDOUBLE generic/ger.c -o sger_k.o	
saxpy_k.o:	
	$(CC) $(CFLAGS) -c -DASMNAME=saxpy_k -DASMFNAME=saxpy_k_ -DNAME=saxpy_k_ -DCNAME=saxpy_k -DCHAR_NAME=\"saxpy_k_\" -DCHAR_CNAME=\"saxpy_k\" -DNO_AFFINITY -I. -UDOUBLE  -UCOMPLEX -UCOMPLEX -UDOUBLE arm/axpy_vfp.S -o saxpy_k.o	
scopy_k.o:	
	$(CC) $(CFLAGS) -c -DASMNAME=scopy_k -DASMFNAME=scopy_k_ -DNAME=scopy_k_ -DCNAME=scopy_k -DCHAR_NAME=\"scopy_k_\" -DCHAR_CNAME=\"scopy_k\" -DNO_AFFINITY -I. -UDOUBLE  -UCOMPLEX -UCOMPLEX -UDOUBLE -DC_INTERFACE arm/copy.c -o scopy_k.o	
libopenblas.so: $(OBJS)
	$(CC) -o $@ $(OBJS) $(CFLAGS) -shared $(LIBS) -lm -lpthread

clean:
	rm *.o libopenblas.so
