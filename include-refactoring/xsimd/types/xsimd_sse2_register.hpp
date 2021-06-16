#ifndef XSIMD_SSE2_REGISTER_HPP
#define XSIMD_SSE2_REGISTER_HPP

#include "./xsimd_sse_register.hpp"

#if XSIMD_WITH_SSE2
#include <emmintrin.h>
#endif

namespace xsimd
{
    struct sse2 : sse
    {
        static constexpr bool supported() { return XSIMD_WITH_SSE2; }
        static constexpr bool available() { return true; }
        static constexpr unsigned version() { return generic::version(1, 2, 0); }
        static constexpr std::size_t alignment() { return 16; }
    };

#if XSIMD_WITH_SSE2
    namespace types
    {
        XSIMD_DECLARE_SIMD_REGISTER_ALIAS(sse2, sse);
    }
#endif
}

#endif

