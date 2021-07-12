#ifndef XSIMD_SSSE3_REGISTER_HPP
#define XSIMD_SSSE3_REGISTER_HPP

#include "./xsimd_sse3_register.hpp"

#if XSIMD_WITH_SSSE3
#include <tmmintrin.h>
#endif

namespace xsimd
{
    struct ssse3 : sse3
    {
        static constexpr bool supported() { return XSIMD_WITH_SSSE3; }
        static constexpr bool available() { return true; }
        static constexpr unsigned version() { return generic::version(1, 3, 1); }
    };

#if XSIMD_WITH_SSSE3
  namespace types
  {
      XSIMD_DECLARE_SIMD_REGISTER_ALIAS(ssse3, sse3);
  }
#endif
}

#endif

