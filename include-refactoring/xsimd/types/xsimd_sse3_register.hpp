#ifndef XSIMD_SSE3_REGISTER_HPP
#define XSIMD_SSE3_REGISTER_HPP

#include "./xsimd_sse2_register.hpp"

#if XSIMD_WITH_SSE3
#include <pmmintrin.h>
#endif

namespace xsimd
{
    struct sse3 : sse2
    {
        static constexpr bool supported() { return XSIMD_WITH_SSE3; }
        static constexpr bool available() { return true; }
        static constexpr unsigned version() { return generic::version(1, 3, 0); }
    };

#if XSIMD_WITH_SSE3
    namespace types
    {

        XSIMD_DECLARE_SIMD_REGISTER_ALIAS(sse3, sse2);
    }
#endif
}

#endif

