#ifndef XSIMD_SSE4_2_REGISTER_HPP
#define XSIMD_SSE4_2_REGISTER_HPP

#include "./xsimd_sse4_1_register.hpp"

#if XSIMD_WITH_SSE4_2
#include <nmmintrin.h>
#endif

namespace xsimd
{
    struct sse4_2 : sse4_1
    {
        static constexpr bool supported() { return XSIMD_WITH_SSE4_2; }
        static constexpr bool available() { return true; }
        static constexpr unsigned version() { return generic::version(1, 4, 2); }
    };

#if XSIMD_WITH_SSE4_2
    namespace types
    {
        XSIMD_DECLARE_SIMD_REGISTER_ALIAS(sse4_2, sse4_1);
    }
#endif
}

#endif

