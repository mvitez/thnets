#include <stdint.h>
#include "gemmlowp/eight_bit_int_gemm/eight_bit_int_gemm.h"

extern "C"  void lowpgemm(const int is_a_transposed,
	const int is_b_transposed,
	const int is_c_transposed,
	const int m, const int n, const int k,
	const uint8_t* a, const uint8_t* b, uint8_t* c,
	const int lda, const int ldb, const int ldc,
	const int a_offset, const int b_offset, const int c_offset,
	const int c_mult, const int c_shift) {

	gemmlowp::eight_bit_int_gemm::EightBitIntGemm(
		is_a_transposed, is_b_transposed, is_c_transposed, m, n, k, a,
		a_offset, lda, b, b_offset, ldb, c, c_offset, c_mult,
		c_shift, ldc, gemmlowp::eight_bit_int_gemm::BitDepthSetting::A8B8);
}
