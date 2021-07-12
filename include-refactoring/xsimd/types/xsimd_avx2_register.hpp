#ifndef XSIMD_AVX2_REGISTER_HPP
#define XSIMD_AVX2_REGISTER_HPP

#include "./xsimd_avx_register.hpp"

namespace xsimd
{
    struct avx2 : avx
    {
        static constexpr bool supported() { return XSIMD_WITH_AVX2; }
        static constexpr bool available() { return true; }
        static constexpr unsigned version() { return generic::version(2, 2, 0); }
    };

#if XSIMD_WITH_AVX2
    namespace types
    {
        XSIMD_DECLARE_SIMD_REGISTER_ALIAS(avx2, avx);
    }
#endif
}

#endif

