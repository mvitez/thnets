/*********************************************************************/
/* Copyright 2009, 2010 The University of Texas at Austin.           */
/* All rights reserved.                                              */
/*                                                                   */
/* Redistribution and use in source and binary forms, with or        */
/* without modification, are permitted provided that the following   */
/* conditions are met:                                               */
/*                                                                   */
/*   1. Redistributions of source code must retain the above         */
/*      copyright notice, this list of conditions and the following  */
/*      disclaimer.                                                  */
/*                                                                   */
/*   2. Redistributions in binary form must reproduce the above      */
/*      copyright notice, this list of conditions and the following  */
/*      disclaimer in the documentation and/or other materials       */
/*      provided with the distribution.                              */
/*                                                                   */
/*    THIS  SOFTWARE IS PROVIDED  BY THE  UNIVERSITY OF  TEXAS AT    */
/*    AUSTIN  ``AS IS''  AND ANY  EXPRESS OR  IMPLIED WARRANTIES,    */
/*    INCLUDING, BUT  NOT LIMITED  TO, THE IMPLIED  WARRANTIES OF    */
/*    MERCHANTABILITY  AND FITNESS FOR  A PARTICULAR  PURPOSE ARE    */
/*    DISCLAIMED.  IN  NO EVENT SHALL THE UNIVERSITY  OF TEXAS AT    */
/*    AUSTIN OR CONTRIBUTORS BE  LIABLE FOR ANY DIRECT, INDIRECT,    */
/*    INCIDENTAL,  SPECIAL, EXEMPLARY,  OR  CONSEQUENTIAL DAMAGES    */
/*    (INCLUDING, BUT  NOT LIMITED TO,  PROCUREMENT OF SUBSTITUTE    */
/*    GOODS  OR  SERVICES; LOSS  OF  USE,  DATA,  OR PROFITS;  OR    */
/*    BUSINESS INTERRUPTION) HOWEVER CAUSED  AND ON ANY THEORY OF    */
/*    LIABILITY, WHETHER  IN CONTRACT, STRICT  LIABILITY, OR TORT    */
/*    (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY WAY OUT    */
/*    OF  THE  USE OF  THIS  SOFTWARE,  EVEN  IF ADVISED  OF  THE    */
/*    POSSIBILITY OF SUCH DAMAGE.                                    */
/*                                                                   */
/* The views and conclusions contained in the software and           */
/* documentation are those of the authors and should not be          */
/* interpreted as representing official policies, either expressed   */
/* or implied, of The University of Texas at Austin.                 */
/*********************************************************************/

/* Modified by Marko Vitez for Purdue University                     */

#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <sched.h>
#include "blas.h"
#include "../sgemm.h"

#define MAX_CPU_NUMBER 8
#define CACHE_LINE_SIZE 8
#define DIVIDE_RATE 2

#ifdef ARM

#define SWITCH_RATIO 2
#define GEMM_P 128
#define GEMM_Q 240
#define GEMM_R 12288
#define GEMM_UNROLL_M 4
#define GEMM_UNROLL_N 4

#define ICOPY_OPERATION(M, N, A, LDA, X, Y, BUFFER) sgemm_otcopy(M, N, (float *)(A) + ((Y) + (X) * (LDA)), LDA, BUFFER)
#define ICOPYT_OPERATION(M, N, A, LDA, X, Y, BUFFER) sgemm_oncopy(M, N, (float *)(A) + ((X) + (Y) * (LDA)), LDA, BUFFER)

#else

#define SWITCH_RATIO 4
#define GEMM_P 1024
#define GEMM_Q 512
#define GEMM_R 15328
#if defined SANDYBRIDGE || defined HASWELL
#define GEMM_UNROLL_M 16
#else
#define GEMM_UNROLL_M 8
#endif
#define GEMM_UNROLL_N 4

#define ICOPY_OPERATION(M, N, A, LDA, X, Y, BUFFER) sgemm_itcopy(M, N, (float *)(A) + ((Y) + (X) * (LDA)), LDA, BUFFER)
#define ICOPYT_OPERATION(M, N, A, LDA, X, Y, BUFFER) sgemm_incopy(M, N, (float *)(A) + ((X) + (Y) * (LDA)), LDA, BUFFER)

#endif // #ifdef ARM

#define BUFFER_SIZE (((GEMM_P + GEMM_R) * GEMM_Q * sizeof(float) + GEMM_ALIGN) & ~GEMM_ALIGN)
#define GEMM_ALIGN 0x03fffUL

#define MIN(a,b) ((a)>(b) ? (b) : (a))

typedef struct {
	volatile long working[MAX_CPU_NUMBER][CACHE_LINE_SIZE * DIVIDE_RATE];
} job_t;

#define OCOPY_OPERATION(M, N, A, LDA, X, Y, BUFFER) sgemm_oncopy(M, N, (float *)(A) + ((X) + (Y) * (LDA)), LDA, BUFFER)
#define OCOPYT_OPERATION(M, N, A, LDA, X, Y, BUFFER) sgemm_otcopy(M, N, (float *)(A) + ((Y) + (X) * (LDA)), LDA, BUFFER)
#define KERNEL_OPERATION(M, N, K, ALPHA, SA, SB, C, LDC, X, Y) sgemm_kernel(M, N, K, ALPHA, SA, SB, (float *)(C) + ((X) + (Y) * LDC), LDC)
#define BETA_OPERATION(M_FROM, M_TO, N_FROM, N_TO, BETA, C, LDC) sgemm_beta((M_TO) - (M_FROM), (N_TO - N_FROM), 0, \
		  BETA, NULL, 0, NULL, 0, (float *)(C) + (M_FROM) + (N_FROM) * (LDC), LDC)

#ifdef TIMING
#define START_RPCC()		rpcc_counter = rpcc()
#define STOP_RPCC(COUNTER)	COUNTER  += rpcc() - rpcc_counter
#else
#define START_RPCC()
#define STOP_RPCC(COUNTER)
#endif

#ifdef ARM
#define WMB  __asm__ __volatile__ ("dmb  ishst" : : : "memory")
#define YIELDING        asm volatile ("nop;nop;nop;nop;nop;nop;nop;nop; \n");
#else
#define WMB
#define YIELDING        sched_yield()
#endif

int threads_num;
float *saa[MAX_CPU_NUMBER], *sba[MAX_CPU_NUMBER];
static struct {
	int x1, y1, plane;
} *xtab;

static struct {
	int x1, y1;
} *ytab;

static void calctabs(struct sgemmargs *args)
{
	int x, y;

	xtab = malloc(sizeof(*xtab) * args->k);
	ytab = malloc(sizeof(*ytab) * args->m);
	for(x = 0; x < args->k; x++)
	{
		xtab[x].plane = x / args->ks0;
		int x1 = x % args->ks0;
		int y1 = x1 / args->ks1 - args->padH;
		x1 = x1 % args->ks1 - args->padW;
		xtab[x].y1 = y1 * args->is1;
		xtab[x].x1 = x1;
	}
	for(y = 0; y < args->m; y++)
	{
		int y1 = y / args->os1 * args->dH;
		int x1 = y % args->os1 * args->dW;
		ytab[y].y1 = y1 * args->is1;
		ytab[y].x1 = x1;
	}
}

static float get_a_pad(struct sgemmargs *args, int x, int y)
{
	int x1 = xtab[x].x1 + ytab[y].x1;
	int y1 = xtab[x].y1 + ytab[y].y1;
	if(y1 < 0 || y1 >= args->is0 || x1 < 0 || x1 >= args->is1)
		return 0;
	return args->a[xtab[x].plane*args->is0 + y1 + x1];
}

static float get_a_nopad(struct sgemmargs *args, int x, int y)
{
	int x1 = xtab[x].x1 + ytab[y].x1;
	int y1 = xtab[x].y1 + ytab[y].y1;
	return args->a[xtab[x].plane*args->is0 + y1 + x1];
}

#ifdef ARM
static void icopy_operation_pad(int m, int n, struct sgemmargs *args, int x, int y, float *b)
{
	int i, i1, j, im;

	//printf("icopy (%d,%d)%d (%d,%d)\n", m, n, lda, x, y);
	for(j = 0; j + 3 < n; j += 4)
	{
		for(i = 0; i < m; i += 4)
		{
			im = m - i > 4 ? 4 : m - i;
			for(i1 = 0; i1 < im; i1++)
			{
				b[i1*4 + 0] = get_a_pad(args, x+i+i1, y+j);
				b[i1*4 + 1] = get_a_pad(args, x+i+i1, y+j+1);
				b[i1*4 + 2] = get_a_pad(args, x+i+i1, y+j+2);
				b[i1*4 + 3] = get_a_pad(args, x+i+i1, y+j+3);
			}
			b += im * 4;
		}
	}
	if(j + 1 < n)
	{
		for(i = 0; i < m; i += 4)
		{
			im = m - i > 4 ? 4 : m - i;
			for(i1 = 0; i1 < im; i1++)
			{
				b[i1*2 + 0] = get_a_pad(args, x+i+i1, y+j);
				b[i1*2 + 1] = get_a_pad(args, x+i+i1, y+j+1);
			}
			b += im * 2;
		}
		j += 2;
	}
	if(j < n)
	{
		for(i = 0; i < m; i += 4)
		{
			im = m - i > 4 ? 4 : m - i;
			for(i1 = 0; i1 < im; i1++)
				b[i1] = get_a_pad(args, x+i+i1, y+j);
			b += im;
		}
	}
}

static void icopy_operation_nopad(int m, int n, struct sgemmargs *args, int x, int y, float *b)
{
	int i, i1, j, im;

	//printf("icopy (%d,%d)%d (%d,%d)\n", m, n, lda, x, y);
	for(j = 0; j + 3 < n; j += 4)
	{
		for(i = 0; i < m; i += 4)
		{
			im = m - i > 4 ? 4 : m - i;
			for(i1 = 0; i1 < im; i1++)
			{
				b[i1*4 + 0] = get_a_nopad(args, x+i+i1, y+j);
				b[i1*4 + 1] = get_a_nopad(args, x+i+i1, y+j+1);
				b[i1*4 + 2] = get_a_nopad(args, x+i+i1, y+j+2);
				b[i1*4 + 3] = get_a_nopad(args, x+i+i1, y+j+3);
			}
			b += im * 4;
		}
	}
	if(j + 1 < n)
	{
		for(i = 0; i < m; i += 4)
		{
			im = m - i > 4 ? 4 : m - i;
			for(i1 = 0; i1 < im; i1++)
			{
				b[i1*2 + 0] = get_a_nopad(args, x+i+i1, y+j);
				b[i1*2 + 1] = get_a_nopad(args, x+i+i1, y+j+1);
			}
			b += im * 2;
		}
		j += 2;
	}
	if(j < n)
	{
		for(i = 0; i < m; i += 4)
		{
			im = m - i > 4 ? 4 : m - i;
			for(i1 = 0; i1 < im; i1++)
				b[i1] = get_a_nopad(args, x+i+i1, y+j);
			b += im;
		}
	}
}

#else

#if defined SANDYBRIDGE || defined HASWELL

static void icopy_operation_pad(int m, int n, struct sgemmargs *args, int x, int y, float *b)
{
	int i, i1, j, im;

	for(j = 0; j + 15 < n; j += 16)
	{
		for(i = 0; i < m; i += 16)
		{
			im = m - i > 16 ? 16 : m - i;
			for(i1 = 0; i1 < im; i1++)
			{
				b[i1*16 + 0] = get_a_pad(args, x+i+i1, y+j);
				b[i1*16 + 1] = get_a_pad(args, x+i+i1, y+j+1);
				b[i1*16 + 2] = get_a_pad(args, x+i+i1, y+j+2);
				b[i1*16 + 3] = get_a_pad(args, x+i+i1, y+j+3);
				b[i1*16 + 4] = get_a_pad(args, x+i+i1, y+j+4);
				b[i1*16 + 5] = get_a_pad(args, x+i+i1, y+j+5);
				b[i1*16 + 6] = get_a_pad(args, x+i+i1, y+j+6);
				b[i1*16 + 7] = get_a_pad(args, x+i+i1, y+j+7);
				b[i1*16 + 8] = get_a_pad(args, x+i+i1, y+j+8);
				b[i1*16 + 9] = get_a_pad(args, x+i+i1, y+j+9);
				b[i1*16 + 10] = get_a_pad(args, x+i+i1, y+j+10);
				b[i1*16 + 11] = get_a_pad(args, x+i+i1, y+j+11);
				b[i1*16 + 12] = get_a_pad(args, x+i+i1, y+j+12);
				b[i1*16 + 13] = get_a_pad(args, x+i+i1, y+j+13);
				b[i1*16 + 14] = get_a_pad(args, x+i+i1, y+j+14);
				b[i1*16 + 15] = get_a_pad(args, x+i+i1, y+j+15);
			}
			b += im * 16;
		}
	}
	if(j + 7 < n)
	{
		for(i = 0; i < m; i += 16)
		{
			im = m - i > 16 ? 16 : m - i;
			for(i1 = 0; i1 < im; i1++)
			{
				b[i1*8 + 0] = get_a_pad(args, x+i+i1, y+j);
				b[i1*8 + 1] = get_a_pad(args, x+i+i1, y+j+1);
				b[i1*8 + 2] = get_a_pad(args, x+i+i1, y+j+2);
				b[i1*8 + 3] = get_a_pad(args, x+i+i1, y+j+3);
				b[i1*8 + 4] = get_a_pad(args, x+i+i1, y+j+4);
				b[i1*8 + 5] = get_a_pad(args, x+i+i1, y+j+5);
				b[i1*8 + 6] = get_a_pad(args, x+i+i1, y+j+6);
				b[i1*8 + 7] = get_a_pad(args, x+i+i1, y+j+7);
			}
			b += im * 8;
		}
		j += 8;
	}
	if(j + 3 < n)
	{
		for(i = 0; i < m; i += 16)
		{
			im = m - i > 16 ? 16 : m - i;
			for(i1 = 0; i1 < im; i1++)
			{
				b[i1*4 + 0] = get_a_pad(args, x+i+i1, y+j);
				b[i1*4 + 1] = get_a_pad(args, x+i+i1, y+j+1);
				b[i1*4 + 2] = get_a_pad(args, x+i+i1, y+j+2);
				b[i1*4 + 3] = get_a_pad(args, x+i+i1, y+j+3);
			}
			b += im * 4;
		}
		j += 4;
	}
	if(j + 1 < n)
	{
		for(i = 0; i < m; i += 16)
		{
			im = m - i > 16 ? 16 : m - i;
			for(i1 = 0; i1 < im; i1++)
			{
				b[i1*2 + 0] = get_a_pad(args, x+i+i1, y+j);
				b[i1*2 + 1] = get_a_pad(args, x+i+i1, y+j+1);
			}
			b += im * 2;
		}
		j += 2;
	}
	if(j < n)
	{
		for(i = 0; i < m; i += 16)
		{
			im = m - i > 16 ? 16 : m - i;
			for(i1 = 0; i1 < im; i1++)
				b[i1] = get_a_pad(args, x+i+i1, y+j);
			b += im;
		}
	}
}

static void icopy_operation_nopad(int m, int n, struct sgemmargs *args, int x, int y, float *b)
{
	int i, i1, j, im;

	for(j = 0; j + 15 < n; j += 16)
	{
		for(i = 0; i < m; i += 16)
		{
			im = m - i > 16 ? 16 : m - i;
			for(i1 = 0; i1 < im; i1++)
			{
				b[i1*16 + 0] = get_a_nopad(args, x+i+i1, y+j);
				b[i1*16 + 1] = get_a_nopad(args, x+i+i1, y+j+1);
				b[i1*16 + 2] = get_a_nopad(args, x+i+i1, y+j+2);
				b[i1*16 + 3] = get_a_nopad(args, x+i+i1, y+j+3);
				b[i1*16 + 4] = get_a_nopad(args, x+i+i1, y+j+4);
				b[i1*16 + 5] = get_a_nopad(args, x+i+i1, y+j+5);
				b[i1*16 + 6] = get_a_nopad(args, x+i+i1, y+j+6);
				b[i1*16 + 7] = get_a_nopad(args, x+i+i1, y+j+7);
				b[i1*16 + 8] = get_a_nopad(args, x+i+i1, y+j+8);
				b[i1*16 + 9] = get_a_nopad(args, x+i+i1, y+j+9);
				b[i1*16 + 10] = get_a_nopad(args, x+i+i1, y+j+10);
				b[i1*16 + 11] = get_a_nopad(args, x+i+i1, y+j+11);
				b[i1*16 + 12] = get_a_nopad(args, x+i+i1, y+j+12);
				b[i1*16 + 13] = get_a_nopad(args, x+i+i1, y+j+13);
				b[i1*16 + 14] = get_a_nopad(args, x+i+i1, y+j+14);
				b[i1*16 + 15] = get_a_nopad(args, x+i+i1, y+j+15);
			}
			b += im * 16;
		}
	}
	if(j + 7 < n)
	{
		for(i = 0; i < m; i += 16)
		{
			im = m - i > 16 ? 16 : m - i;
			for(i1 = 0; i1 < im; i1++)
			{
				b[i1*8 + 0] = get_a_nopad(args, x+i+i1, y+j);
				b[i1*8 + 1] = get_a_nopad(args, x+i+i1, y+j+1);
				b[i1*8 + 2] = get_a_nopad(args, x+i+i1, y+j+2);
				b[i1*8 + 3] = get_a_nopad(args, x+i+i1, y+j+3);
				b[i1*8 + 4] = get_a_nopad(args, x+i+i1, y+j+4);
				b[i1*8 + 5] = get_a_nopad(args, x+i+i1, y+j+5);
				b[i1*8 + 6] = get_a_nopad(args, x+i+i1, y+j+6);
				b[i1*8 + 7] = get_a_nopad(args, x+i+i1, y+j+7);
			}
			b += im * 8;
		}
		j += 8;
	}
	if(j + 3 < n)
	{
		for(i = 0; i < m; i += 16)
		{
			im = m - i > 16 ? 16 : m - i;
			for(i1 = 0; i1 < im; i1++)
			{
				b[i1*4 + 0] = get_a_nopad(args, x+i+i1, y+j);
				b[i1*4 + 1] = get_a_nopad(args, x+i+i1, y+j+1);
				b[i1*4 + 2] = get_a_nopad(args, x+i+i1, y+j+2);
				b[i1*4 + 3] = get_a_nopad(args, x+i+i1, y+j+3);
			}
			b += im * 4;
		}
		j += 4;
	}
	if(j + 1 < n)
	{
		for(i = 0; i < m; i += 16)
		{
			im = m - i > 16 ? 16 : m - i;
			for(i1 = 0; i1 < im; i1++)
			{
				b[i1*2 + 0] = get_a_nopad(args, x+i+i1, y+j);
				b[i1*2 + 1] = get_a_nopad(args, x+i+i1, y+j+1);
			}
			b += im * 2;
		}
		j += 2;
	}
	if(j < n)
	{
		for(i = 0; i < m; i += 16)
		{
			im = m - i > 16 ? 16 : m - i;
			for(i1 = 0; i1 < im; i1++)
				b[i1] = get_a_nopad(args, x+i+i1, y+j);
			b += im;
		}
	}
}

#else
static void icopy_operation_pad(int m, int n, struct sgemmargs *args, int x, int y, float *b)
{
	int i, i1, j, im;

	for(j = 0; j + 7 < n; j += 8)
	{
		for(i = 0; i < m; i += 8)
		{
			im = m - i > 8 ? 8 : m - i;
			for(i1 = 0; i1 < im; i1++)
			{
				b[i1*8 + 0] = get_a_pad(args, x+i+i1, y+j);
				b[i1*8 + 1] = get_a_pad(args, x+i+i1, y+j+1);
				b[i1*8 + 2] = get_a_pad(args, x+i+i1, y+j+2);
				b[i1*8 + 3] = get_a_pad(args, x+i+i1, y+j+3);
				b[i1*8 + 4] = get_a_pad(args, x+i+i1, y+j+4);
				b[i1*8 + 5] = get_a_pad(args, x+i+i1, y+j+5);
				b[i1*8 + 6] = get_a_pad(args, x+i+i1, y+j+6);
				b[i1*8 + 7] = get_a_pad(args, x+i+i1, y+j+7);
			}
			b += im * 8;
		}
	}
	if(j + 3 < n)
	{
		for(i = 0; i < m; i += 8)
		{
			im = m - i > 8 ? 8 : m - i;
			for(i1 = 0; i1 < im; i1++)
			{
				b[i1*4 + 0] = get_a_pad(args, x+i+i1, y+j);
				b[i1*4 + 1] = get_a_pad(args, x+i+i1, y+j+1);
				b[i1*4 + 2] = get_a_pad(args, x+i+i1, y+j+2);
				b[i1*4 + 3] = get_a_pad(args, x+i+i1, y+j+3);
			}
			b += im * 4;
		}
		j += 4;
	}
	if(j + 1 < n)
	{
		for(i = 0; i < m; i += 8)
		{
			im = m - i > 8 ? 8 : m - i;
			for(i1 = 0; i1 < im; i1++)
			{
				b[i1*2 + 0] = get_a_pad(args, x+i+i1, y+j);
				b[i1*2 + 1] = get_a_pad(args, x+i+i1, y+j+1);
			}
			b += im * 2;
		}
		j += 2;
	}
	if(j < n)
	{
		for(i = 0; i < m; i += 8)
		{
			im = m - i > 8 ? 8 : m - i;
			for(i1 = 0; i1 < im; i1++)
				b[i1] = get_a_pad(args, x+i+i1, y+j);
			b += im;
		}
	}
}

static void icopy_operation_nopad(int m, int n, struct sgemmargs *args, int x, int y, float *b)
{
	int i, i1, j, im;

	for(j = 0; j + 7 < n; j += 8)
	{
		for(i = 0; i < m; i += 8)
		{
			im = m - i > 8 ? 8 : m - i;
			for(i1 = 0; i1 < im; i1++)
			{
				b[i1*8 + 0] = get_a_nopad(args, x+i+i1, y+j);
				b[i1*8 + 1] = get_a_nopad(args, x+i+i1, y+j+1);
				b[i1*8 + 2] = get_a_nopad(args, x+i+i1, y+j+2);
				b[i1*8 + 3] = get_a_nopad(args, x+i+i1, y+j+3);
				b[i1*8 + 4] = get_a_nopad(args, x+i+i1, y+j+4);
				b[i1*8 + 5] = get_a_nopad(args, x+i+i1, y+j+5);
				b[i1*8 + 6] = get_a_nopad(args, x+i+i1, y+j+6);
				b[i1*8 + 7] = get_a_nopad(args, x+i+i1, y+j+7);
			}
			b += im * 8;
		}
	}
	if(j + 3 < n)
	{
		for(i = 0; i < m; i += 8)
		{
			im = m - i > 8 ? 8 : m - i;
			for(i1 = 0; i1 < im; i1++)
			{
				b[i1*4 + 0] = get_a_nopad(args, x+i+i1, y+j);
				b[i1*4 + 1] = get_a_nopad(args, x+i+i1, y+j+1);
				b[i1*4 + 2] = get_a_nopad(args, x+i+i1, y+j+2);
				b[i1*4 + 3] = get_a_nopad(args, x+i+i1, y+j+3);
			}
			b += im * 4;
		}
		j += 4;
	}
	if(j + 1 < n)
	{
		for(i = 0; i < m; i += 8)
		{
			im = m - i > 8 ? 8 : m - i;
			for(i1 = 0; i1 < im; i1++)
			{
				b[i1*2 + 0] = get_a_nopad(args, x+i+i1, y+j);
				b[i1*2 + 1] = get_a_nopad(args, x+i+i1, y+j+1);
			}
			b += im * 2;
		}
		j += 2;
	}
	if(j < n)
	{
		for(i = 0; i < m; i += 8)
		{
			im = m - i > 8 ? 8 : m - i;
			for(i1 = 0; i1 < im; i1++)
				b[i1] = get_a_nopad(args, x+i+i1, y+j);
			b += im;
		}
	}
}

#endif
#endif

static void icopy_operation(int m, int n, struct sgemmargs *args, int x, int y, float *b)
{
	if(args->padW || args->padH)
		icopy_operation_pad(m, n, args, x, y, b);
	else icopy_operation_nopad(m, n, args, x, y, b);
}

static job_t job[MAX_CPU_NUMBER];

static int gemm_thread(long mypos, long nthreads, struct sgemmargs *args, long *range_m, long *range_n)
{
	float *buffer[DIVIDE_RATE];
	float *sa, *sb;
	long m_from, m_to, n_from, n_to;
	long xxx, bufferside;
	long ls, min_l, jjs, min_jj;
	long is, min_i, div_n;
	long i, current;
	long l1stride;
	char transa = args->transa;
	char transb = args->transb;
	long m = args->m;
	long n = args->n;
	long k = args->k;
	float alpha = args->alpha;
	float beta = args->beta;
	float *a = args->a;
	float *b = args->b;
	float *c = args->c;
	long lda = args->lda;
	long ldb = args->ldb;
	long ldc = args->ldc;

#ifdef TIMING
	unsigned long rpcc_counter;
	unsigned long copy_A = 0;
	unsigned long copy_B = 0;
	unsigned long kernel = 0;
	unsigned long waiting1 = 0;
	unsigned long waiting2 = 0;
	unsigned long waiting3 = 0;
	unsigned long waiting6[MAX_CPU_NUMBER];
	unsigned long ops    = 0;

	for (i = 0; i < num_threads; i++)
		waiting6[i] = 0;
#endif
	sa = saa[mypos];
	sb = sba[mypos];
	m_from = 0;
	m_to = m;

	if(range_m)
	{
		m_from = range_m[mypos + 0];
		m_to   = range_m[mypos + 1];
	}
	n_from = 0;
	n_to   = n;
	if (range_n)
	{
		n_from = range_n[mypos + 0];
		n_to   = range_n[mypos + 1];
	}
	if(beta != 1)
		BETA_OPERATION(m_from, m_to, 0, n, beta, c, ldc);
	if(k == 0 || alpha == 0)
		return 0;
#if 0
	fprintf(stderr, "Thread[%ld]  m_from : %ld m_to : %ld n_from : %ld n_to : %ld\n",
		mypos, m_from, m_to, n_from, n_to);
	fprintf(stderr, "GEMM: P = %4ld  Q = %4ld  R = %4ld\n", (long)GEMM_P, (long)GEMM_Q, (long)GEMM_R);
#endif
	div_n = (n_to - n_from + DIVIDE_RATE - 1) / DIVIDE_RATE;
	buffer[0] = sb;
	for (i = 1; i < DIVIDE_RATE; i++)
		buffer[i] = buffer[i - 1] + GEMM_Q * ((div_n + GEMM_UNROLL_N - 1) & ~(GEMM_UNROLL_N - 1));
	for(ls = 0; ls < k; ls += min_l)
	{
		min_l = k - ls;
		if (min_l >= GEMM_Q * 2)
			min_l  = GEMM_Q;
		else if (min_l > GEMM_Q)
			min_l = (min_l + 1) / 2;
		l1stride = 1;
		min_i = m_to - m_from;
		if (min_i >= GEMM_P * 2)
			min_i = GEMM_P;
		else if(min_i > GEMM_P)
			min_i = (min_i / 2 + GEMM_UNROLL_M - 1) & ~(GEMM_UNROLL_M - 1);
		else if (nthreads == 1)
			l1stride = 0;
		START_RPCC();
		//printf("icopy%ld (%ld,%ld)%ld (%ld,%ld)\n", mypos, min_l, min_i, lda, ls, m_from);
		if(transa)
			ICOPYT_OPERATION(min_l, min_i, a, lda, ls, m_from, sa);
		else if(args->ks0)
			icopy_operation(min_l, min_i, args, ls, m_from, sa);
		else ICOPY_OPERATION(min_l, min_i, a, lda, ls, m_from, sa);
		STOP_RPCC(copy_A);
		div_n = (n_to - n_from + DIVIDE_RATE - 1) / DIVIDE_RATE;
		for (xxx = n_from, bufferside = 0; xxx < n_to; xxx += div_n, bufferside ++)
		{
			START_RPCC();
			/* Make sure if no one is using buffer */
			for (i = 0; i < nthreads; i++)
				while (job[mypos].working[i][CACHE_LINE_SIZE * bufferside])
					{YIELDING;}
			STOP_RPCC(waiting1);
			for(jjs = xxx; jjs < MIN(n_to, xxx + div_n); jjs += min_jj)
			{
				min_jj = MIN(n_to, xxx + div_n) - jjs;
				if(min_jj >= 3*GEMM_UNROLL_N)
					min_jj = 3*GEMM_UNROLL_N;
				else if (min_jj > GEMM_UNROLL_N)
					min_jj = GEMM_UNROLL_N;
				START_RPCC();
				if(transb)
					OCOPYT_OPERATION(min_l, min_jj, b, ldb, ls, jjs, buffer[bufferside] + min_l * (jjs - xxx) * l1stride);
				else OCOPY_OPERATION(min_l, min_jj, b, ldb, ls, jjs, buffer[bufferside] + min_l * (jjs - xxx) * l1stride);
				STOP_RPCC(copy_B);
				START_RPCC();
				KERNEL_OPERATION(min_i, min_jj, min_l, alpha, sa, buffer[bufferside] + min_l * (jjs - xxx) * l1stride, c, ldc, m_from, jjs);
				STOP_RPCC(kernel);
#ifdef TIMING
				ops += 2 * min_i * min_jj * min_l;
#endif
			}
			for (i = 0; i < nthreads; i++)
				job[mypos].working[i][CACHE_LINE_SIZE * bufferside] = (long)buffer[bufferside];
			WMB;
		}
		current = mypos;
		do {
			current ++;
			if(current >= nthreads)
				current = 0;
			div_n = (range_n[current + 1]  - range_n[current] + DIVIDE_RATE - 1) / DIVIDE_RATE;
			for (xxx = range_n[current], bufferside = 0; xxx < range_n[current + 1]; xxx += div_n, bufferside ++)
			{
				if (current != mypos)
				{
					START_RPCC();
					/* thread has to wait */
					while(job[current].working[mypos][CACHE_LINE_SIZE * bufferside] == 0)
						{YIELDING;}
					STOP_RPCC(waiting2);
					START_RPCC();
					KERNEL_OPERATION(min_i, MIN(range_n[current + 1]  - xxx,  div_n), min_l, alpha, sa,
						(float *)job[current].working[mypos][CACHE_LINE_SIZE * bufferside], c, ldc, m_from, xxx);
					STOP_RPCC(kernel);
	#ifdef TIMING
					ops += 2 * min_i * MIN(range_n[current + 1]  - xxx,  div_n) * min_l;
	#endif
				}
				if (m_to - m_from == min_i)
					job[current].working[mypos][CACHE_LINE_SIZE * bufferside] &= 0;
			}
		} while (current != mypos);
		for(is = m_from + min_i; is < m_to; is += min_i)
		{
			min_i = m_to - is;
			if (min_i >= GEMM_P * 2)
				min_i = GEMM_P;
			else if (min_i > GEMM_P)
				min_i = ((min_i + 1) / 2 + GEMM_UNROLL_M - 1) & ~(GEMM_UNROLL_M - 1);
			START_RPCC();
			//printf("icopya%ld (%ld,%ld)%ld (%ld,%ld)\n", mypos, min_l, min_i, lda, ls, is);
			if(transa)
				ICOPYT_OPERATION(min_l, min_i, a, lda, ls, is, sa);
			else if(args->ks0)
				icopy_operation(min_l, min_i, args, ls, is, sa);
			else ICOPY_OPERATION(min_l, min_i, a, lda, ls, is, sa);
			STOP_RPCC(copy_A);
			current = mypos;
			do {
				div_n = (range_n[current + 1] - range_n[current] + DIVIDE_RATE - 1) / DIVIDE_RATE;
				for (xxx = range_n[current], bufferside = 0; xxx < range_n[current + 1]; xxx += div_n, bufferside ++)
				{
					START_RPCC();
					KERNEL_OPERATION(min_i, MIN(range_n[current + 1] - xxx, div_n), min_l, alpha, sa,
						(float *)job[current].working[mypos][CACHE_LINE_SIZE * bufferside], c, ldc, is, xxx);
					STOP_RPCC(kernel);
#ifdef TIMING
					ops += 2 * min_i * MIN(range_n[current + 1]  - xxx, div_n) * min_l;
#endif
					if(is + min_i >= m_to)
					{
						/* Thread doesn't need this buffer any more */
						job[current].working[mypos][CACHE_LINE_SIZE * bufferside] &= 0;
						WMB;
					}
				}
				current ++;
				if(current >= nthreads)
					current = 0;
			} while (current != mypos);
		}
	}
	START_RPCC();
	for (i = 0; i < nthreads; i++)
		for (xxx = 0; xxx < DIVIDE_RATE; xxx++)
			while (job[mypos].working[i][CACHE_LINE_SIZE * xxx] )
				{YIELDING;}
	STOP_RPCC(waiting3);
#ifdef TIMING
	long waiting = waiting1 + waiting2 + waiting3;
	long total = copy_A + copy_B + kernel + waiting;

	fprintf(stderr, "GEMM   [%2ld] Copy_A : %6.2f  Copy_B : %6.2f  Wait1 : %6.2f Wait2 : %6.2f Wait3 : %6.2f Kernel : %6.2f",
	mypos, (double)copy_A /(double)total * 100., (double)copy_B /(double)total * 100.,
	(double)waiting1 /(double)total * 100.,
	(double)waiting2 /(double)total * 100.,
	(double)waiting3 /(double)total * 100.,
	(double)ops/(double)kernel / 4. * 100.);

#if 0
	fprintf(stderr, "GEMM   [%2ld] Copy_A : %6.2ld  Copy_B : %6.2ld  Wait : %6.2ld\n",
	mypos, copy_A, copy_B, waiting);

	fprintf(stderr, "Waiting[%2ld] %6.2f %6.2f %6.2f\n", mypos,
		(double)waiting1/(double)waiting * 100.,
		(double)waiting2/(double)waiting * 100.,
		(double)waiting3/(double)waiting * 100.);
#endif
	fprintf(stderr, "\n");
#endif
	return 0;
}

static int gemm_single(int mypos, struct sgemmargs *args)
{
	long m_from, m_to, n_from, n_to;

	long ls, is, js;
	long min_l, min_i, min_j;
	long jjs, min_jj;
	float *sa = saa[mypos];
	float *sb = sba[mypos];
	long l1stride, gemm_p, l2size;
	char transa = args->transa;
	char transb = args->transb;
	long m = args->m;
	long n = args->n;
	long k = args->k;
	float alpha = args->alpha;
	float beta = args->beta;
	float *a = args->a;
	float *b = args->b;
	float *c = args->c;
	long lda = args->lda;
	long ldb = args->ldb;
	long ldc = args->ldc;

#ifdef TIMING
	unsigned long long rpcc_counter;
	unsigned long long innercost  = 0;
	unsigned long long outercost  = 0;
	unsigned long long kernelcost = 0;
	double total;
#endif

	m_from = 0;
	m_to   = m;
	n_from = 0;
	n_to   = n;
	if (beta != 1)
		BETA_OPERATION(m_from, m_to, n_from, n_to, beta, c, ldc);

	if((k == 0) || (alpha == 0))
		return 0;
	l2size = GEMM_P * GEMM_Q;
#if 0
	fprintf(stderr, "GEMM(Single): M_from : %ld  M_to : %ld  N_from : %ld  N_to : %ld  k : %ld\n", m_from, m_to, n_from, n_to, k);
	fprintf(stderr, "GEMM(Single):: P = %4ld  Q = %4ld  R = %4ld\n", (long)GEMM_P, (long)GEMM_Q, (long)GEMM_R);
	//  fprintf(stderr, "GEMM: SA .. %p  SB .. %p\n", sa, sb);

	//  fprintf(stderr, "A = %p  B = %p  C = %p\n\tlda = %ld  ldb = %ld ldc = %ld\n", a, b, c, lda, ldb, ldc);
#endif

#ifdef TIMING
	innercost = 0;
	outercost = 0;
	kernelcost = 0;
#endif

	for(js = n_from; js < n_to; js += GEMM_R)
	{
		min_j = n_to - js;
		if (min_j > GEMM_R)
			min_j = GEMM_R;

		for(ls = 0; ls < k; ls += min_l)
		{
			min_l = k - ls;
			if(min_l >= GEMM_Q * 2)
			{
				gemm_p = GEMM_P;
				min_l  = GEMM_Q;
			} else {
				if(min_l > GEMM_Q)
					min_l = (min_l / 2 + GEMM_UNROLL_M - 1) & ~(GEMM_UNROLL_M - 1);
				gemm_p = ((l2size / min_l + GEMM_UNROLL_M - 1) & ~(GEMM_UNROLL_M - 1));
				while (gemm_p * min_l > l2size)
					gemm_p -= GEMM_UNROLL_M;
			}
			/* First, we have to move data A to L2 cache */
			min_i = m_to - m_from;
			l1stride = 1;
			if(min_i >= GEMM_P * 2)
				min_i = GEMM_P;
			else if(min_i > GEMM_P)
				min_i = (min_i / 2 + GEMM_UNROLL_M - 1) & ~(GEMM_UNROLL_M - 1);
			else l1stride = 0;
			START_RPCC();
			if(transa)
				ICOPYT_OPERATION(min_l, min_i, a, lda, ls, m_from, sa);
			else if(args->ks0)
				icopy_operation(min_l, min_i, args, ls, m_from, sa);
			else ICOPY_OPERATION(min_l, min_i, a, lda, ls, m_from, sa);
			STOP_RPCC(innercost);
			for(jjs = js; jjs < js + min_j; jjs += min_jj)
			{
				min_jj = min_j + js - jjs;
				if(min_jj >= 3*GEMM_UNROLL_N)
					min_jj = 3*GEMM_UNROLL_N;
				else if(min_jj > GEMM_UNROLL_N)
					min_jj = GEMM_UNROLL_N;
				START_RPCC();
				if(transb)
					OCOPYT_OPERATION(min_l, min_jj, b, ldb, ls, jjs, sb + min_l * (jjs - js) * l1stride);
				else OCOPY_OPERATION(min_l, min_jj, b, ldb, ls, jjs, sb + min_l * (jjs - js) * l1stride);
				STOP_RPCC(outercost);
				START_RPCC();
				KERNEL_OPERATION(min_i, min_jj, min_l, alpha, sa,
					sb + min_l * (jjs - js) * l1stride, c, ldc, m_from, jjs);
				STOP_RPCC(kernelcost);
			}

			for(is = m_from + min_i; is < m_to; is += min_i)
			{
				min_i = m_to - is;
				if(min_i >= GEMM_P * 2)
					min_i = GEMM_P;
				else if(min_i > GEMM_P)
					min_i = (min_i / 2 + GEMM_UNROLL_M - 1) & ~(GEMM_UNROLL_M - 1);
				START_RPCC();
				if(transa)
					ICOPYT_OPERATION(min_l, min_i, a, lda, ls, is, sa);
				else if(args->ks0)
					icopy_operation(min_l, min_i, args, ls, is, sa);
				else ICOPY_OPERATION(min_l, min_i, a, lda, ls, is, sa);
				STOP_RPCC(innercost);
				START_RPCC();
				KERNEL_OPERATION(min_i, min_j, min_l, alpha, sa, sb, c, ldc, is, js);
				STOP_RPCC(kernelcost);
			} /* end of is */
		} /* end of js */
	} /* end of ls */

#ifdef TIMING
	total = (double)outercost + (double)innercost + (double)kernelcost;
	printf( "Copy A : %5.2f Copy  B: %5.2f  Kernel : %5.2f  kernel Effi. : %5.2f Total Effi. : %5.2f\n",
		innercost / total * 100., outercost / total * 100.,
		kernelcost / total * 100.,
		(double)(m_to - m_from) * (double)(n_to - n_from) * (double)k / (double)kernelcost * 100.  / 2.,
		(double)(m_to - m_from) * (double)(n_to - n_from) * (double)k / total * 100. / 2.);
#endif
	return 0;
}

void sgemmargs(struct sgemmargs *args)
{
	long range_M[MAX_CPU_NUMBER + 1];
	long range_N[MAX_CPU_NUMBER + 1];

	long num_cpu_m, num_cpu_n;

	long width, i, j, k1, js;
	long m1, n1, n_from, n_to;

	range_M[0] = 0;
	m1 = args->m;
	num_cpu_m  = 0;

	if(omp_in_parallel())
	{
		if(args->ks0 && omp_get_thread_num() == 0)
			calctabs(args);
#pragma omp barrier
		gemm_single(omp_get_thread_num(), args);
#pragma omp barrier
		if(args->ks0 && omp_get_thread_num() == 0)
		{
			free(xtab);
			free(ytab);
		}
		return;
	}
	if(args->ks0)
		calctabs(args);
	if((args->m < threads_num * SWITCH_RATIO) || (args->n < threads_num * SWITCH_RATIO))
	{
		gemm_single(0, args);
		if(args->ks0)
		{
			free(xtab);
			free(ytab);
		}
		return;
	}

	while(m1 > 0)
	{
		width = (m1 + threads_num - num_cpu_m - 1) / (threads_num - num_cpu_m);
		m1 -= width;
		if(m1 < 0)
			width = width + m1;
		range_M[num_cpu_m + 1] = range_M[num_cpu_m] + width;
		num_cpu_m++;
	}
	n_from = 0;
	n_to   = args->n;

	for(js = n_from; js < n_to; js += GEMM_R * threads_num)
	{
		n1 = n_to - js;
		if (n1 > GEMM_R * threads_num)
			n1 = GEMM_R * threads_num;
		range_N[0] = js;
		num_cpu_n  = 0;
		while (n1 > 0)
		{
			width = (n1 + threads_num - num_cpu_n - 1) / (threads_num - num_cpu_n);
			n1 -= width;
			if(n1 < 0)
				width = width + n1;
			range_N[num_cpu_n + 1] = range_N[num_cpu_n] + width;
			num_cpu_n++;
		}

		for (j = 0; j < num_cpu_m; j++)
			for (i = 0; i < num_cpu_m; i++)
				for (k1 = 0; k1 < DIVIDE_RATE; k1++)
					job[j].working[i][CACHE_LINE_SIZE * k1] = 0;
#pragma omp parallel for
		for(i = 0; i < threads_num; i++)
			gemm_thread(i, threads_num, args, range_M, range_N);
	}
	if(args->ks0)
	{
		free(xtab);
		free(ytab);
	}
}

void sgemm(char transa, char transb, long m, long n, long k, float alpha, float *a, long lda, float *b, long ldb, float beta, float *c, long ldc)
{
	struct sgemmargs args;

	args.transa = transa == 't';
	args.transb = transb == 't';
	args.m = m;
	args.n = n;
	args.k = k;
	args.alpha = alpha;
	args.a = a;
	args.lda = lda;
	args.b = b;
	args.ldb = ldb;
	args.beta = beta;
	args.c = c;
	args.ldc = ldc;
	args.ks0 = 0;
	sgemmargs(&args);
}

void blas_init()
{
	int i;

	if(!threads_num)
		threads_num = omp_get_max_threads();
	for(i = 0; i < threads_num; i++)
	{
		saa[i] = malloc(BUFFER_SIZE);
		sba[i] = (float *)((char *)saa[i] + ((GEMM_P * GEMM_Q * sizeof(float) + GEMM_ALIGN) & ~GEMM_ALIGN));
	}
}
