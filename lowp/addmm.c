void THLowpTensor_addmm(THFloatTensor *r_, float beta, THFloatTensor *t, float alpha, THFloatTensor *m1, THFloatTensor *m2)
{
	const int transpose_m1 = 1;
	const int transpose_m2 = 1;
	const int transpose_r_ = 1;

	const int m = m1->size[0];
	const int n = m2->size[1];
	const int k = m1->size[1];

	THByteBlas_gemm8(transpose_m1, transpose_m2, transpose_r_,
		m, n, k,
		THByteTensor_data(m1),
		THByteTensor_data(m2),
		THByteTensor_data(r_),
		k, n, n,
		m1_offset, m2_offset, r_offset, r_mult, r_shift);
}
