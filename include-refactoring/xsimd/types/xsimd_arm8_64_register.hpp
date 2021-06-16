#ifndef XSIMD_ARM8_64_REGISTER_HPP
#define XSIMD_ARM8_64_REGISTER_HPP

#include "xsimd_arm8_32_register.hpp"

namespace xsimd
{
    struct arm8_64 : arm8_32
    {
        static constexpr bool supported() { return XSIMD_WITH_ARM8_64; }
        static constexpr bool available() { return true; }
        static constexpr bool requires_alignment() { return true; }
        static constexpr std::size_t alignment() { return 16; }
        static constexpr unsigned version() { return generic::version(8, 1, 0); }
    };

#if XSIMD_WITH_ARM8_64

    namespace types
    {
        XSIMD_DECLARE_SIMD_REGISTER_ALIAS(arm8_64, arm8_32);
        XSIMD_DECLARE_SIMD_REGISTER(double, arm8_64, float64x2_t);
    }

#endif

}

#endif


