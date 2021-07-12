#ifndef XSIMD_SSE_REGISTER_HPP
#define XSIMD_SSE_REGISTER_HPP

#include "./xsimd_register.hpp"
#include "./xsimd_generic_arch.hpp"

#if XSIMD_WITH_SSE
#include <xmmintrin.h>
#endif

namespace xsimd
{
    struct sse : generic
    {
        static constexpr bool supported() { return XSIMD_WITH_SSE; }
        static constexpr bool available() { return true; }
        static constexpr bool requires_alignment() { return true; }
        static constexpr std::size_t alignment() { return 8; }
        static constexpr unsigned version() { return generic::version(1, 1, 0); }
    };

#if XSIMD_WITH_SSE
    namespace types
    {

        XSIMD_DECLARE_SIMD_REGISTER(bool, sse, __m128i);
        XSIMD_DECLARE_SIMD_REGISTER(signed char, sse, __m128i);
        XSIMD_DECLARE_SIMD_REGISTER(unsigned char, sse, __m128i);
        XSIMD_DECLARE_SIMD_REGISTER(char, sse, __m128i);
        XSIMD_DECLARE_SIMD_REGISTER(unsigned short, sse, __m128i);
        XSIMD_DECLARE_SIMD_REGISTER(short, sse, __m128i);
        XSIMD_DECLARE_SIMD_REGISTER(unsigned int, sse, __m128i);
        XSIMD_DECLARE_SIMD_REGISTER(int, sse, __m128i);
        XSIMD_DECLARE_SIMD_REGISTER(unsigned long int, sse, __m128i);
        XSIMD_DECLARE_SIMD_REGISTER(long int, sse, __m128i);
        XSIMD_DECLARE_SIMD_REGISTER(unsigned long long int, sse, __m128i);
        XSIMD_DECLARE_SIMD_REGISTER(long long int, sse, __m128i);
        XSIMD_DECLARE_SIMD_REGISTER(float, sse, __m128);
        XSIMD_DECLARE_SIMD_REGISTER(double, sse, __m128d);

    }
#endif
}

#endif

