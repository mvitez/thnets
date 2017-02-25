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
#include <stdio.h>
#include <stdlib.h>
#include "blas.h"

#define GEMM_MULTITHREAD_THRESHOLD 4

void sgemv(char trans, int m, int n, float alpha, float *a, int lda, float *x, int incx, float beta, float *y, int incy)
{
	long range[MAX_CPU_NUMBER+1];
	int nthreads;
	long width;	
	double MNK;
	int lenx, leny;
	int i, num_cpu;

	lenx = m;
	leny = n;
	if(trans != 't')
	{
		fprintf(stderr, "sgemv not supported for untransposed matrix\n");
		exit(0);
	}
	if(beta != 1)
	{
		fprintf(stderr, "sgemv not supported for beta != 1\n");
		exit(0);
	}
	if(alpha == 0)
		return;
	if(incx < 0)
		x -= (lenx - 1) * incx;
	if(incy < 0)
		y -= (leny - 1) * incy;
	
	if(omp_in_parallel())
	{
		sgemv_t(m, n, 0, alpha, a, lda, x, incx, y, incy, 0);
		return;
	}
	MNK = (double) m * (double) n;
	if(MNK <= 24.0 * 24.0 * GEMM_MULTITHREAD_THRESHOLD*GEMM_MULTITHREAD_THRESHOLD)
	{
		sgemv_t(m, n, 0, alpha, a, lda, x, incx, y, incy, 0);
		return;
	}
	range[0] = 0;
	i = n;
	nthreads = threads_num;
	num_cpu = 0;
	while(i > 0)
	{
		width  = (i + nthreads - num_cpu - 1) / (nthreads - num_cpu);
		if(width < 4)
		width = 4;
		if(i < width)
			width = i;
		range[num_cpu + 1] = range[num_cpu] + width;
		num_cpu ++;
		i -= width;
	}
#pragma omp parallel for
	for(i = 0; i < nthreads; i++)
		sgemv_t(m, range[i+1] - range[i], 0, alpha, a + range[i] * lda, lda, x, incx, y + range[i] * incy, incy, 0);
}
