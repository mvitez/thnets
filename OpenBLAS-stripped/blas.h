#define MAX_CPU_NUMBER 8

// ARM assembly functions use -mfloat-abi=hard calling convention, Android does not,
// unless armeabi-v7a-hard is specified in Application.mk, but this is not supported anymore
#ifdef ANDROID
#define FCALL __attribute__((pcs("aapcs-vfp")))
#else
#define FCALL
#endif

#define ZERO 0
typedef long BLASLONG;
typedef float FLOAT;

extern float *saa[MAX_CPU_NUMBER], *sba[MAX_CPU_NUMBER];
extern int threads_num;

int sgemm_incopy(long m, long n, float *a, long lda, float *b);
int sgemm_itcopy(long m, long n, float *a, long lda, float *b);
int sgemm_oncopy(long m, long n, float *a, long lda, float *b);
int sgemm_otcopy(long m, long n, float *a, long lda, float *b);
int FCALL sgemm_kernel(long m, long n, long k, float alpha, float *sa, float *sb, float *c, long ldc);
int sgemm_beta(long m, long n, long dummy1, float beta, float *dummy2, long dummy3, float *dummy4, long dummy5, float *c, long ldc);
int scopy_k(long n, float *x, long incx, float *y, long incy);
void FCALL saxpy_k(long, long, long, float, float *, long, float *, long, float *, long);
int sgemv_t(long m, long n, long dummy1, float alpha, float *a, long lda, float *x, long inc_x, float *y, long inc_y, float *buffer);
