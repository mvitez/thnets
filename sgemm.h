struct sgemmargs {
	char transa, transb;
	long m, n, k;
	long lda, ldb, ldc;
	float alpha, beta;
	float *a, *b, *c;
	long os0, os1, ks0, ks1, is0, is1, ih;
	long dW, dH, padW, padH;
};

void sgemm(char transa, char transb, long m, long n, long k, float alpha, float *a, long lda, float *b, long ldb, float beta, float *c, long ldc);
void sgemmargs(struct sgemmargs *args);
