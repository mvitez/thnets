#define MAX_CPU_NUMBER 8

#define ZERO 0
typedef long BLASLONG;
typedef float FLOAT;

extern float *saa[MAX_CPU_NUMBER], *sba[MAX_CPU_NUMBER];
extern int threads_num;

int sgemm_incopy(long m, long n, float *a, long lda, float *b);
int sgemm_itcopy(long m, long n, float *a, long lda, float *b);
int sgemm_oncopy(long m, long n, float *a, long lda, float *b);
int sgemm_otcopy(long m, long n, float *a, long lda, float *b);
int sgemm_kernel(long m, long n, long k, float alpha, float *sa, float *sb, float *c, long ldc);
int sgemm_beta(long m, long n, long dummy1, float beta, float *dummy2, long dummy3, float *dummy4, long dummy5, float *c, long ldc);
int scopy_k(long n, float *x, long incx, float *y, long incy);
void saxpy_k(long, long, long, float, float *, long, float *, long, float *, long);
int sgemv_t(long m, long n, long dummy1, float alpha, float *a, long lda, float *x, long inc_x, float *y, long inc_y, float *buffer);
