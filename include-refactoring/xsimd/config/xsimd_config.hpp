#ifndef XSIMD_CONFIG_HPP
#define XSIMD_CONFIG_HPP

#ifdef __SSE__
#define XSIMD_WITH_SSE 1
#else
#define XSIMD_WITH_SSE 0
#endif

#ifdef __SSE2__
#define XSIMD_WITH_SSE2 1
#else
#define XSIMD_WITH_SSE2 0
#endif

#ifdef __SSE3__
#define XSIMD_WITH_SSE3 1
#else
#define XSIMD_WITH_SSE3 0
#endif

#ifdef __SSSE3__
#define XSIMD_WITH_SSSE3 1
#else
#define XSIMD_WITH_SSSE3 0
#endif

#ifdef __SSE4_1__
#define XSIMD_WITH_SSE4_1 1
#else
#define XSIMD_WITH_SSE4_1 0
#endif

#ifdef __SSE4_2__
#define XSIMD_WITH_SSE4_2 1
#else
#define XSIMD_WITH_SSE4_2 0
#endif

#ifdef __AVX__
#define XSIMD_WITH_AVX 1
#else
#define XSIMD_WITH_AVX 0
#endif

#ifdef __AVX2__
#define XSIMD_WITH_AVX2 1
#else
#define XSIMD_WITH_AVX2 0
#endif

#ifdef __FMA__

#if defined(__SSE__) && ! defined(__AVX__)
#define XSIMD_WITH_FMA3 1
#define XSIMD_WITH_FMA5 0
#endif

#if defined(__AVX__)
#define XSIMD_WITH_FMA3 0
#define XSIMD_WITH_FMA5 1
#endif

#if !defined(__SSE__) && ! defined(__AVX__)
#define XSIMD_WITH_FMA3 0
#define XSIMD_WITH_FMA5 0
#endif

#else

#define XSIMD_WITH_FMA3 0
#define XSIMD_WITH_FMA5 0
#endif

#ifdef __AVX512F__
// AVX512 instructions are supported starting with gcc 6
// see https://www.gnu.org/software/gcc/gcc-6/changes.html
#if defined(__GNUC__) && __GNUC__ < 6
#define XSIMD_WITH_AVX512F 0
#else
#define XSIMD_WITH_AVX512F 1
#endif
#else
#define XSIMD_WITH_AVX512F 0
#endif

#endif
