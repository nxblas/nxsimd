#ifndef XSIMD_ISA_HPP
#define XSIMD_ISA_HPP

#include "../config/xsimd_arch.hpp"

#include "./xsimd_generic_fwd.hpp"

#if XSIMD_WITH_SSE
#include "./xsimd_sse.hpp"
#endif

#if XSIMD_WITH_SSE2
#include "./xsimd_sse2.hpp"
#endif

#if XSIMD_WITH_SSE3
#include "./xsimd_sse3.hpp"
#endif

#if XSIMD_WITH_SSSE3
#include "./xsimd_ssse3.hpp"
#endif

#if XSIMD_WITH_SSE4_1
#include "./xsimd_sse4_1.hpp"
#endif

#if XSIMD_WITH_SSE4_2
#include "./xsimd_sse4_2.hpp"
#endif

#if XSIMD_WITH_FMA3
#include "./xsimd_fma3.hpp"
#endif

#if XSIMD_WITH_AVX
#include "./xsimd_avx.hpp"
#endif

#if XSIMD_WITH_AVX2
#include "./xsimd_avx2.hpp"
#endif

#if XSIMD_WITH_FMA5
#include "./xsimd_fma5.hpp"
#endif

#if XSIMD_WITH_AVX512F
#include "./xsimd_avx512f.hpp"
#endif

// Must come last to have access to all conversion specializations.
#include "./xsimd_generic.hpp"


#endif

