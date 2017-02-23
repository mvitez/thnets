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

#ifndef COMMON_X86
#define COMMON_X86

#ifdef __APPLE__
#define OS_DARWIN
#endif

#define SIZE 4
#define BASE_SHIFT 2
#define BLASLONG long
#define BLASULONG unsigned long
#define FLOAT float

#define ALIGNED_ACCESS
#define SHUFPD_0	shufpd	$0,
#define SHUFPD_1	shufpd	$1,
#define SHUFPD_2	shufpd	$2,
#define SHUFPD_3	shufpd	$3,
#define SHUFPS_39	shufps	$0x39,

#ifndef PAGESIZE
#define PAGESIZE	( 4 << 10)
#endif
#define HUGE_PAGESIZE	( 2 << 20)

#define BUFFER_SIZE	(32 << 20)

#define SEEK_ADDRESS

#ifdef F_INTERFACE_G77
#define RETURN_BY_STACK
#define NEED_F2CCONV
#endif

#ifdef F_INTERFACE_G95
#define RETURN_BY_PACKED
#endif

#ifdef F_INTERFACE_GFORT
#ifdef OS_WINDOWS
#ifndef DOUBLE
#define RETURN_BY_REGS
#else
#define RETURN_BY_STACK
#endif
#else
#define RETURN_BY_PACKED
#endif
#endif

#ifdef F_INTERFACE_INTEL
#define RETURN_BY_STACK
#endif

#ifdef F_INTERFACE_FUJITSU
#define RETURN_BY_STACK
#endif

#ifdef F_INTERFACE_PGI
#define RETURN_BY_STACK
#endif

#ifdef F_INTERFACE_PATHSCALE
#define RETURN_BY_PACKED
#endif

#ifdef F_INTERFACE_SUN
#define RETURN_BY_PACKED
#endif

#ifdef ASSEMBLER

#if defined(PILEDRIVER) || defined(BULLDOZER) || defined(STEAMROLLER) || defined(EXCAVATOR)
//Enable some optimazation for barcelona.
#define BARCELONA_OPTIMIZATION
#endif

#if defined(HAVE_3DNOW)
#define EMMS	femms
#elif defined(HAVE_MMX)
#define EMMS	emms
#endif

#ifndef EMMS
#define EMMS
#endif

#define BRANCH		.byte 0x3e
#define NOBRANCH	.byte 0x2e
#define PADDING		.byte 0x66

#ifdef OS_WINDOWS
#define ARG1	%rcx
#define ARG2	%rdx
#define ARG3	%r8
#define ARG4	%r9
#else
#define ARG1	%rdi
#define ARG2	%rsi
#define ARG3	%rdx
#define ARG4	%rcx
#define ARG5	%r8
#define ARG6	%r9
#endif

#ifndef COMPLEX
#ifdef XDOUBLE
#define LOCAL_BUFFER_SIZE  QLOCAL_BUFFER_SIZE
#elif defined DOUBLE
#define LOCAL_BUFFER_SIZE  DLOCAL_BUFFER_SIZE
#else
#define LOCAL_BUFFER_SIZE  SLOCAL_BUFFER_SIZE
#endif
#else
#ifdef XDOUBLE
#define LOCAL_BUFFER_SIZE  XLOCAL_BUFFER_SIZE
#elif defined DOUBLE
#define LOCAL_BUFFER_SIZE  ZLOCAL_BUFFER_SIZE
#else
#define LOCAL_BUFFER_SIZE  CLOCAL_BUFFER_SIZE
#endif
#endif

#if defined(OS_WINDOWS)
#if   LOCAL_BUFFER_SIZE > 16384
#define STACK_TOUCHING \
	movl	$0,  4096 * 4(%rsp);\
	movl	$0,  4096 * 3(%rsp);\
	movl	$0,  4096 * 2(%rsp);\
	movl	$0,  4096 * 1(%rsp);
#elif LOCAL_BUFFER_SIZE > 12288
#define STACK_TOUCHING \
	movl	$0,  4096 * 3(%rsp);\
	movl	$0,  4096 * 2(%rsp);\
	movl	$0,  4096 * 1(%rsp);
#elif LOCAL_BUFFER_SIZE > 8192
#define STACK_TOUCHING \
	movl	$0,  4096 * 2(%rsp);\
	movl	$0,  4096 * 1(%rsp);
#elif LOCAL_BUFFER_SIZE > 4096
#define STACK_TOUCHING \
	movl	$0,  4096 * 1(%rsp);
#else
#define STACK_TOUCHING
#endif
#else
#define STACK_TOUCHING
#endif

#if defined(CORE2)
#define movapd	movaps
#define andpd	andps
#define movlpd	movlps
#define movhpd	movhps
#endif

#ifdef OS_DARWIN
#define BUILDNAME(a) _ ## a
#else
#define BUILDNAME(a) a
#endif

#define REALNAME ASMNAME

#ifdef OS_DARWIN
#define PROLOGUE .text;.align 5; .globl REALNAME; REALNAME:
#define EPILOGUE	.subsections_via_symbols
#define PROFCODE
#endif

#ifdef OS_WINDOWS
#define SAVEREGISTERS \
	subq	$256, %rsp;\
	movups	%xmm6,    0(%rsp);\
	movups	%xmm7,   16(%rsp);\
	movups	%xmm8,   32(%rsp);\
	movups	%xmm9,   48(%rsp);\
	movups	%xmm10,  64(%rsp);\
	movups	%xmm11,  80(%rsp);\
	movups	%xmm12,  96(%rsp);\
	movups	%xmm13, 112(%rsp);\
	movups	%xmm14, 128(%rsp);\
	movups	%xmm15, 144(%rsp)

#define RESTOREREGISTERS \
	movups	   0(%rsp), %xmm6;\
	movups	  16(%rsp), %xmm7;\
	movups	  32(%rsp), %xmm8;\
	movups	  48(%rsp), %xmm9;\
	movups	  64(%rsp), %xmm10;\
	movups	  80(%rsp), %xmm11;\
	movups	  96(%rsp), %xmm12;\
	movups	 112(%rsp), %xmm13;\
	movups	 128(%rsp), %xmm14;\
	movups	 144(%rsp), %xmm15;\
	addq	$256, %rsp
#else
#define SAVEREGISTERS
#define RESTOREREGISTERS
#endif

#if defined(OS_WINDOWS) && !defined(C_PGI)
#define PROLOGUE \
	.text; \
	.align 16; \
	.globl REALNAME ;\
	.def REALNAME;.scl	2;.type	32;.endef; \
REALNAME:

#define PROFCODE

#define EPILOGUE .end	 REALNAME
#endif

#if defined(OS_LINUX) || defined(OS_FREEBSD) || defined(OS_NETBSD) || defined(__ELF__) || defined(C_PGI)
#define PROLOGUE \
	.text; \
	.align 512; \
	.globl REALNAME ;\
       .type REALNAME, @function; \
REALNAME:

#ifdef PROFILE
#define PROFCODE call *mcount@GOTPCREL(%rip)
#else
#define PROFCODE
#endif

#define EPILOGUE \
        .size	 REALNAME, .-REALNAME; \
        .section .note.GNU-stack,"",@progbits


#endif

#endif

#ifdef XDOUBLE
#define FLD	fldt
#define FST	fstpt
#define MOVQ	movq
#elif defined(DOUBLE)
#define FLD	fldl
#define FST	fstpl
#define FSTU	fstl
#define FMUL	fmull
#define FADD	faddl
#define MOVSD	movsd
#define MULSD	mulsd
#define MULPD	mulpd
#define CMPEQPD	cmpeqpd
#define COMISD	comisd
#define PSRLQ	psrlq
#define ANDPD	andpd
#define ADDPD	addpd
#define ADDSD	addsd
#define SUBPD	subpd
#define SUBSD	subsd
#define MOVQ	movq
#define MOVUPD	movupd
#define XORPD	xorpd
#else
#define FLD	flds
#define FST	fstps
#define FSTU	fsts
#define FMUL	fmuls
#define FADD	fadds
#define MOVSD	movss
#define MULSD	mulss
#define MULPD	mulps
#define CMPEQPD	cmpeqps
#define COMISD	comiss
#define PSRLQ	psrld
#define ANDPD	andps
#define ADDPD	addps
#define ADDSD	addss
#define SUBPD	subps
#define SUBSD	subss
#define MOVQ	movd
#define MOVUPD	movups
#define XORPD	xorps
#endif

#define HALT	hlt

#ifdef OS_DARWIN
#define ALIGN_2 .align 2
#define ALIGN_3 .align 3
#define ALIGN_4 .align 4
#define ALIGN_5 .align 5
#define ffreep	fstp
#endif

#ifndef ALIGN_2
#define ALIGN_2 .align 4
#endif

#ifndef ALIGN_3
#define ALIGN_3 .align 8
#endif

#ifndef ALIGN_4
#define ALIGN_4 .align 16
#endif

#ifndef ALIGN_5
#define ALIGN_5 .align 32
#endif

#ifndef ALIGN_6
#define ALIGN_6 .align 64
#endif

// ffreep %st(0).
// Because Clang didn't support ffreep, we directly use the opcode.
// Please check out http://www.sandpile.org/x86/opc_fpu.htm
#ifndef ffreep
#define ffreep .byte 0xdf, 0xc0 #
#endif
#endif
