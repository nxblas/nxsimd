#ifndef XSIMD_ARM8_32_REGISTER_HPP
#define XSIMD_ARM8_32_REGISTER_HPP

#include "xsimd_arm7_register.hpp"

namespace xsimd
{
    struct arm8_32 : arm7
    {
        static constexpr bool supported() { return XSIMD_WITH_ARM8_32; }
        static constexpr bool available() { return true; }
        static constexpr bool requires_alignment() { return true; }
        static constexpr std::size_t alignment() { return 16; }
        static constexpr unsigned version() { return generic::version(8, 0, 0); }
    };

#if XSIMD_WITH_ARM8_32

    namespace types
    {
        XSIMD_DECLARE_SIMD_REGISTER_ALIAS(arm8_32, arm7);
    }

#endif

}

#endif

