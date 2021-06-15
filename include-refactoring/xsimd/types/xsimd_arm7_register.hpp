#ifndef XSIMD_ARM7_REGISTER_HPP
#define XSIMD_ARM7_REGISTER_HPP

#include "xsimd_register.hpp"
#include "xsimd_generic_arch.hpp"

#include <arm_neon.h>

namespace xsimd
{
    struct arm7 : generic
    {
        static constexpr bool supported() { return XSIMD_WITH_ARM7; }
        static constexpr bool available() { return true; }
        static constexpr bool requires_alignment() { return true; }
        static constexpr std::size_t alignment() { return 16; }
        static constexpr unsigned version() { return generic::version(7, 0, 0); }
    };

#if XSIMD_WITH_ARM7
    namespace types
    {
        namespace detail
        {
            template <size_t S>
            struct arm_vector_type_impl;

            template <>
            struct arm_vector_type_impl<8>
            {
                using signed_type = int8x16_t;
                using unsigned_type = uint8x16_t;
            };

            template <>
            struct arm_vector_type_impl<16>
            {
                using signed_type = int16x8_t;
                using unsigned_type = uint16x8_t;
            };

            template <>
            struct arm_vector_type_impl<32>
            {
                using signed_type = int32x4_t;
                using unsigned_type = uint32x4_t;
            };

            template <>
            struct arm_vector_type_impl<64>
            {
                using signed_type = int64x2_t;
                using unsigned_type = uint64x2_t;
            }

            template <class T>
            using signed_arm_vector_type = typename arm_vector_type_impl<sizeof(T)>::signed_type;

            template <class T>
            using unsigned_arm_vector_type = typename arm_vector_type_impl<sizeof(T)>::unsigned_type;

            template <class T>
            using arm_vector_type = typename std::conditional<std::is_signed<T>,
                                                              signed_arm_vector_type<T>,
                                                              unsigned_arm_vector_type<T>
                                                             >::type;

            using char_arm_vector type = typename std::conditional<std::is_signed<char>,
                                                                   signed_arm_vector_type<char>,
                                                                   unsigned_arm_vector_type<char>
                                                                  >::type;
        }

        XSIMD_DECLARE_SIMD_REGISTER(signed char, arm7, (detail::arm_vector_type<signed char>));
        XSIMD_DECLARE_SIMD_REGISTER(unsigned char, arm7, (detail::arm_vector_type<unsigned char>));
        XSIMD_DECLARE_SIMD_REGISTER(char, arm7, (detail::char_arm_vector_type));
        XSIMD_DECLARE_SIMD_REGISTER(short, arm7, (detail::arm_vector_type<short>));
        XSIMD_DECLARE_SIMD_REGISTER(unsigned short, arm7, (detail::arm_vector_type<unsigned short>));
        XSIMD_DECLARE_SIMD_REGISTER(int, arm7, (detail::arm_vector_type<int>));
        XSIMD_DECLARE_SIMD_REGISTER(unsigned int, arm7, (detail::arm_vector_type<unsigned int>));
        XSIMD_DECLARE_SIMD_REGISTER(long int, arm7, (detail::arm_vector_type<long int>));
        XSIMD_DECLARE_SIMD_REGISTER(unsigned long int, arm7, (detail::arm_vector_type<unsigned long int>));
        XSIMD_DECLARE_SIMD_REGISTER(long long int, arm7, (detail::arm_vector_type<long long int>));
        XSIMD_DECLARE_SIMD_REGISTER(unsigned long long int, arm7, (detail::arm_vector_type<unsigned long long int>));
        XSIMD_DECLARE_SIMD_REGISTER(float, arm7, float32x4_t);
    }
#endif

}

#endif

