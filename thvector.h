#if defined USE_SSE2 || defined USE_SSE3 || defined USE_SSSE3 \
  || defined USE_SSE4_1 || defined USE_SSE4_2

#ifdef USE_SSE2
#include <emmintrin.h>
#endif

#ifdef USE_SSE3
#include <pmmintrin.h>
#endif

#ifdef USE_SSSE3
#include <tmmintrin.h>
#endif

#if defined (USE_SSE4_2) || defined (USE_SSE4_1)
#include <smmintrin.h>
#endif

#define THFloatVector_fill(x, c, n) {           \
    long i;                                     \
    __m128 XMM0 = _mm_set_ps1(c);               \
    long off;                                   \
    for (i=0; i<=((n)-16); i+=16) {             \
      _mm_storeu_ps((x)+i  ,  XMM0);            \
      _mm_storeu_ps((x)+i+4,  XMM0);            \
      _mm_storeu_ps((x)+i+8,  XMM0);            \
      _mm_storeu_ps((x)+i+12, XMM0);            \
    }                                           \
    off = (n) - ((n)%16);                       \
    for (i=0; i<((n)%16); i++) {                \
      x[off+i] = c;                             \
    }                                           \
  }

#define THFloatVector_add(y, x, c, n) {         \
    long i = 0;                                 \
    __m128 XMM7 = _mm_set_ps1(c);               \
    __m128 XMM0,XMM2;                           \
    for (; i<=((n)-4); i+=4) {                  \
      XMM0 = _mm_loadu_ps((x)+i);               \
      XMM2 = _mm_loadu_ps((y)+i);               \
      XMM0 = _mm_mul_ps(XMM0, XMM7);            \
      XMM2 = _mm_add_ps(XMM2, XMM0);            \
      _mm_storeu_ps((y)+i  , XMM2);             \
    }                                           \
    for (; i<(n); i++) {                        \
      y[i] += c * x[i];                         \
    }                                           \
  }

#elif defined __NEON__

#define THFloatVector_fill(x, c, n) {                   \
        float ctemp = c;                                \
        float * caddr = &ctemp;                         \
        __asm__ __volatile__ (                          \
            "mov         r0, %0           @ \n\t"       \
            "ldr         r4, [%1]         @ \n\t"       \
            "vdup.32     q12, r4          @ \n\t"       \
            "vdup.32     q13, r4          @ \n\t"       \
            "lsrs        r4, %2, #3       @ \n\t"       \
            "beq         3f               @ \n\t"       \
            "1:                           @ \n\t"       \
            "vst1.32     {d24-d27}, [r0]! @ \n\t"       \
            "subs        r4, r4, #1       @ \n\t"       \
            "bne         1b               @ \n\t"       \
            "3:                           @ \n\t"       \
            "ands        r4, %2, #7       @ \n\t"       \
            "beq         5f               @ \n\t"       \
            "4:                           @ \n\t"       \
            "subs        r4, r4, #1       @ \n\t"       \
            "vst1.32     {d24[0]}, [r0]!  @ \n\t"       \
            "bne         4b               @ \n\t"       \
            "5:                           @ "           \
            :                                           \
            :"r" (x), "r"(caddr),"r"(n)                 \
            : "cc", "r0", "r4",  "memory",              \
              "q12",                                    \
              "d24", "d25", "d26", "d27"                \
            );                                          \
    }

#define THFloatVector_add(y, x, c, n) {                                 \
        float ctemp = c;                                                \
        float * caddr = &ctemp;                                         \
        __asm__ __volatile__ (                                          \
            "mov         r0, %0           @ \n\t"                       \
            "mov         r1, %1           @ \n\t"                       \
            "mov         r2, r0           @ \n\t"                       \
            "ldr         r5, [%2]         @ \n\t"                       \
            "vdup.32     q14, r5          @ \n\t"                       \
            "lsrs        r5, %3, #4       @ \n\t"                       \
            "beq         3f               @ \n\t"                       \
            "vld1.32     {d16-d19}, [r1]! @ \n\t"                       \
            "vld1.32     {d0-d3}, [r0]!   @ \n\t"                       \
            "vld1.32     {d20-d23}, [r1]! @ \n\t"                       \
            "vld1.32     {d4-d7}, [r0]!   @ \n\t"                       \
            "1:                           @ \n\t"                       \
            "vmla.f32    q0, q8, q14      @ \n\t"                       \
            "vmla.f32    q1, q9, q14      @ \n\t"                       \
            "vmla.f32    q2, q10, q14     @ \n\t"                       \
            "vmla.f32    q3, q11, q14     @ \n\t"                       \
            "subs        r5, r5, #1       @ \n\t"                       \
            "beq         2f               @ \n\t"                       \
            "vld1.32     {d16-d19}, [r1]! @ \n\t"                       \
            "vld1.32     {d20-d23}, [r1]! @ \n\t"                       \
            "vst1.32     {d0-d3}, [r2]!   @ \n\t"                       \
            "vld1.32     {d0-d3}, [r0]!   @ \n\t"                       \
            "vst1.32     {d4-d7}, [r2]!   @ \n\t"                       \
            "vld1.32     {d4-d7}, [r0]!   @ \n\t"                       \
            "b           1b               @ \n\t"                       \
            "2:                           @ \n\t"                       \
            "vst1.32     {d0-d3}, [r2]!   @ \n\t"                       \
            "vst1.32     {d4-d7}, [r2]!   @ \n\t"                       \
            "3:                           @ \n\t"                       \
            "lsrs        r5, %3, #3       @ \n\t"                       \
            "ands        r5, #1           @ \n\t"                       \
            "beq         4f               @ \n\t"                       \
            "vld1.32     {d16-d19}, [r1]! @ \n\t"                       \
            "vld1.32     {d0-d3}, [r0]!   @ \n\t"                       \
            "vmla.f32    q0, q8, q14      @ \n\t"                       \
            "vmla.f32    q1, q9, q14      @ \n\t"                       \
            "vst1.32     {d0-d3}, [r2]!   @ \n\t"                       \
            "4:                           @ \n\t"                       \
            "ands        r5, %3, #7       @ \n\t"                       \
            "beq         6f               @ \n\t"                       \
            "5:                           @ \n\t"                       \
            "subs        r5, r5, #1       @ \n\t"                       \
            "vld1.32     {d16[0]}, [r1]!  @ \n\t"                       \
            "vld1.32     {d0[0]}, [r0]!   @ \n\t"                       \
            "vmla.f32    d0, d16, d28     @ \n\t"                       \
            "vst1.32     d0[0], [r2]!     @ \n\t"                       \
            "bne         5b               @ \n\t"                       \
            "6:                           @ "                           \
            :                                                           \
            :"r" (y),"r" (x), "r"(caddr),"r"(n)                         \
            : "cc", "r0", "r1", "r2", "r5", "memory",                   \
              "q0", "q1", "q2", "q3", "q14",                            \
              "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7",           \
              "d16", "d17", "d18", "d19", "d20", "d21", "d22", "d23", "d28", "d29" \
            );                                                          \
    }

#else	

static inline void THFloatVector_fill(float *x, float c, long n)
{
	long i = 0;
	for(; i < n-4; i += 4)
	{
		x[i] = c;
		x[i+1] = c;
		x[i+2] = c;
		x[i+3] = c;
	}

	for(; i < n; i++)
		x[i] = c;
}

static inline void THFloatVector_add(float *y, const float *x, const float c, const long n)
{
	long i = 0;

	for(;i < n-4; i += 4)
	{
		y[i] += c * x[i];
		y[i+1] += c * x[i+1];
		y[i+2] += c * x[i+2];
		y[i+3] += c * x[i+3];
	}

	for(; i < n; i++)
		y[i] += c * x[i];
}

#endif
