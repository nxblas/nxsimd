/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay, Wolf Vollprecht and         *
* Martin Renou                                                             *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_AVX512_CONVERSION_HPP
#define XSIMD_AVX512_CONVERSION_HPP

#include "xsimd_avx512_double.hpp"
#include "xsimd_avx512_float.hpp"
#include "xsimd_avx512_int8.hpp"
#include "xsimd_avx512_int16.hpp"
#include "xsimd_avx512_int32.hpp"
#include "xsimd_avx512_int64.hpp"

namespace xsimd
{

    /************************
     * conversion functions *
     ************************/

    batch<int32_t, 16> to_int(const batch<float, 16>& x);
    batch<int64_t, 8> to_int(const batch<double, 16>& x);

    batch<float, 16> to_float(const batch<int32_t, 16>& x);
    batch<double, 8> to_float(const batch<int64_t, 8>& x);

    batch<uint16_t, 32> u8_to_u16(const batch<uint8_t, 64>& x);
    batch<uint8_t, 64> u16_to_u8(const batch<uint16_t, 32>& x);
    batch<uint32_t, 16> u8_to_u32(const batch<uint8_t, 64>& x);
    batch<uint8_t, 64> u32_to_u8(const batch<uint32_t, 16>& x);
    batch<uint64_t, 8> u8_to_u64(const batch<uint8_t, 64>& x);
    batch<uint8_t, 64> u64_to_u8(const batch<uint64_t, 8>& x);

    /**************************
     * boolean cast functions *
     **************************/

    batch_bool<int32_t, 16> bool_cast(const batch_bool<float, 16>& x);
    batch_bool<int64_t, 8> bool_cast(const batch_bool<double, 8>& x);
    batch_bool<float, 16> bool_cast(const batch_bool<int32_t, 16>& x);
    batch_bool<double, 8> bool_cast(const batch_bool<int64_t, 8>& x);

    /*******************************
     * bitwise_cast implementation *
     *******************************/

    XSIMD_DEFINE_BITWISE_CAST_ALL(8)

    /***************************************
     * conversion functions implementation *
     ***************************************/

    inline batch<int32_t, 16> to_int(const batch<float, 16>& x)
    {
        return _mm512_cvttps_epi32(x);
    }

    inline batch<int64_t, 8> to_int(const batch<double, 8>& x)
    {
#if defined(XSIMD_AVX512DQ_AVAILABLE)
        return _mm512_cvttpd_epi64(x);
#else
        return batch<int64_t, 8>(static_cast<int64_t>(x[0]),
                                 static_cast<int64_t>(x[1]),
                                 static_cast<int64_t>(x[2]),
                                 static_cast<int64_t>(x[3]),
                                 static_cast<int64_t>(x[4]),
                                 static_cast<int64_t>(x[5]),
                                 static_cast<int64_t>(x[6]),
                                 static_cast<int64_t>(x[7]));
#endif
    }

    inline batch<float, 16> to_float(const batch<int32_t, 16>& x)
    {
        return _mm512_cvtepi32_ps(x);
    }

    inline batch<double, 8> to_float(const batch<int64_t, 8>& x)
    {
#if defined(XSIMD_AVX512DQ_AVAILABLE)
        return _mm512_cvtepi64_pd(x);
#else
        return batch<double, 8>(static_cast<double>(x[0]),
                                static_cast<double>(x[1]),
                                static_cast<double>(x[2]),
                                static_cast<double>(x[3]),
                                static_cast<double>(x[4]),
                                static_cast<double>(x[5]),
                                static_cast<double>(x[6]),
                                static_cast<double>(x[7]));
#endif
    }

    /*****************************************
     * batch cast functions implementation *
     *****************************************/

    XSIMD_BATCH_CAST_IMPLICIT(int8_t, uint8_t, 64)
    XSIMD_BATCH_CAST_IMPLICIT(uint8_t, int8_t, 64)
    XSIMD_BATCH_CAST_IMPLICIT(int16_t, uint16_t, 32)
    XSIMD_BATCH_CAST_INTRINSIC(int16_t, int32_t, 16, _mm512_cvtepi16_epi32)
    XSIMD_BATCH_CAST_INTRINSIC(int16_t, uint32_t, 16, _mm512_cvtepi16_epi32)
    XSIMD_BATCH_CAST_INTRINSIC(int16_t, int64_t, 8, _mm512_cvtepi16_epi64)
    XSIMD_BATCH_CAST_INTRINSIC(int16_t, uint64_t, 8, _mm512_cvtepi16_epi64)
    XSIMD_BATCH_CAST_INTRINSIC2(int16_t, float, 16, _mm512_cvtepi16_epi32, _mm512_cvtepi32_ps)
    XSIMD_BATCH_CAST_IMPLICIT(uint16_t, int16_t, 32)
    XSIMD_BATCH_CAST_INTRINSIC(uint16_t, int32_t, 16, _mm512_cvtepu16_epi32)
    XSIMD_BATCH_CAST_INTRINSIC(uint16_t, uint32_t, 16, _mm512_cvtepu16_epi32)
    XSIMD_BATCH_CAST_INTRINSIC(uint16_t, int64_t, 8, _mm512_cvtepu16_epi64)
    XSIMD_BATCH_CAST_INTRINSIC(uint16_t, uint64_t, 8, _mm512_cvtepu16_epi64)
    XSIMD_BATCH_CAST_INTRINSIC2(uint16_t, float, 16, _mm512_cvtepu16_epi32, _mm512_cvtepi32_ps)
    XSIMD_BATCH_CAST_INTRINSIC(int32_t, int8_t, 16, _mm512_cvtepi32_epi8)
    XSIMD_BATCH_CAST_INTRINSIC(int32_t, uint8_t, 16, _mm512_cvtepi32_epi8)
    XSIMD_BATCH_CAST_INTRINSIC(int32_t, int16_t, 16, _mm512_cvtepi32_epi16)
    XSIMD_BATCH_CAST_INTRINSIC(int32_t, uint16_t, 16, _mm512_cvtepi32_epi16)
    XSIMD_BATCH_CAST_IMPLICIT(int32_t, uint32_t, 16)
    XSIMD_BATCH_CAST_INTRINSIC(int32_t, int64_t, 8, _mm512_cvtepi32_epi64)
    XSIMD_BATCH_CAST_INTRINSIC(int32_t, uint64_t, 8, _mm512_cvtepi32_epi64)
    XSIMD_BATCH_CAST_INTRINSIC(int32_t, float, 16, _mm512_cvtepi32_ps)
    XSIMD_BATCH_CAST_INTRINSIC(int32_t, double, 8, _mm512_cvtepi32_pd)
    XSIMD_BATCH_CAST_INTRINSIC(uint32_t, int8_t, 16, _mm512_cvtepi32_epi8)
    XSIMD_BATCH_CAST_INTRINSIC(uint32_t, uint8_t, 16, _mm512_cvtepi32_epi8)
    XSIMD_BATCH_CAST_INTRINSIC(uint32_t, int16_t, 16, _mm512_cvtepi32_epi16)
    XSIMD_BATCH_CAST_INTRINSIC(uint32_t, uint16_t, 16, _mm512_cvtepi32_epi16)
    XSIMD_BATCH_CAST_IMPLICIT(uint32_t, int32_t, 16)
    XSIMD_BATCH_CAST_INTRINSIC(uint32_t, int64_t, 8, _mm512_cvtepu32_epi64)
    XSIMD_BATCH_CAST_INTRINSIC(uint32_t, uint64_t, 8, _mm512_cvtepu32_epi64)
    XSIMD_BATCH_CAST_INTRINSIC(uint32_t, float, 16, _mm512_cvtepu32_ps)
    XSIMD_BATCH_CAST_INTRINSIC(uint32_t, double, 8, _mm512_cvtepu32_pd)
    XSIMD_BATCH_CAST_INTRINSIC(int64_t, int16_t, 8, _mm512_cvtepi64_epi16)
    XSIMD_BATCH_CAST_INTRINSIC(int64_t, uint16_t, 8, _mm512_cvtepi64_epi16)
    XSIMD_BATCH_CAST_INTRINSIC(int64_t, int32_t, 8, _mm512_cvtepi64_epi32)
    XSIMD_BATCH_CAST_INTRINSIC(int64_t, uint32_t, 8, _mm512_cvtepi64_epi32)
    XSIMD_BATCH_CAST_IMPLICIT(int64_t, uint64_t, 8)
    XSIMD_BATCH_CAST_INTRINSIC(uint64_t, int16_t, 8, _mm512_cvtepi64_epi16)
    XSIMD_BATCH_CAST_INTRINSIC(uint64_t, uint16_t, 8, _mm512_cvtepi64_epi16)
    XSIMD_BATCH_CAST_INTRINSIC(uint64_t, int32_t, 8, _mm512_cvtepi64_epi32)
    XSIMD_BATCH_CAST_INTRINSIC(uint64_t, uint32_t, 8, _mm512_cvtepi64_epi32)
    XSIMD_BATCH_CAST_IMPLICIT(uint64_t, int64_t, 8)
    XSIMD_BATCH_CAST_INTRINSIC2(float, int8_t, 16, _mm512_cvttps_epi32, _mm512_cvtepi32_epi8)
    XSIMD_BATCH_CAST_INTRINSIC2(float, uint8_t, 16, _mm512_cvttps_epi32, _mm512_cvtepi32_epi8)
    XSIMD_BATCH_CAST_INTRINSIC2(float, int16_t, 16, _mm512_cvttps_epi32, _mm512_cvtepi32_epi16)
    XSIMD_BATCH_CAST_INTRINSIC2(float, uint16_t, 16, _mm512_cvttps_epi32, _mm512_cvtepi32_epi16)
    XSIMD_BATCH_CAST_INTRINSIC(float, int32_t, 16, _mm512_cvttps_epi32)
    XSIMD_BATCH_CAST_INTRINSIC(float, uint32_t, 16, _mm512_cvttps_epu32)
    XSIMD_BATCH_CAST_INTRINSIC(float, double, 8, _mm512_cvtps_pd)
    XSIMD_BATCH_CAST_INTRINSIC(double, int32_t, 8, _mm512_cvttpd_epi32)
    XSIMD_BATCH_CAST_INTRINSIC(double, uint32_t, 8, _mm512_cvttpd_epu32)
    XSIMD_BATCH_CAST_INTRINSIC(double, float, 8, _mm512_cvtpd_ps)
#if defined(XSIMD_AVX512BW_AVAILABLE)
    XSIMD_BATCH_CAST_INTRINSIC(int8_t, int16_t, 32, _mm512_cvtepi8_epi16)
    XSIMD_BATCH_CAST_INTRINSIC(int8_t, uint16_t, 32, _mm512_cvtepi8_epi16)
    XSIMD_BATCH_CAST_INTRINSIC(int8_t, int32_t, 16, _mm512_cvtepi8_epi32)
    XSIMD_BATCH_CAST_INTRINSIC(int8_t, uint32_t, 16, _mm512_cvtepi8_epi32)
    XSIMD_BATCH_CAST_INTRINSIC2(int8_t, float, 16, _mm512_cvtepi8_epi32, _mm512_cvtepi32_ps)
    XSIMD_BATCH_CAST_INTRINSIC(uint8_t, int16_t, 32, _mm512_cvtepu8_epi16)
    XSIMD_BATCH_CAST_INTRINSIC(uint8_t, uint16_t, 32, _mm512_cvtepu8_epi16)
    XSIMD_BATCH_CAST_INTRINSIC(uint8_t, int32_t, 16, _mm512_cvtepu8_epi32)
    XSIMD_BATCH_CAST_INTRINSIC(uint8_t, uint32_t, 16, _mm512_cvtepu8_epi32)
    XSIMD_BATCH_CAST_INTRINSIC2(uint8_t, float, 16, _mm512_cvtepu8_epi32, _mm512_cvtepi32_ps)
    XSIMD_BATCH_CAST_INTRINSIC(int16_t, int8_t, 32, _mm512_cvtepi16_epi8)
    XSIMD_BATCH_CAST_INTRINSIC(int16_t, uint8_t, 32, _mm512_cvtepi16_epi8)
    XSIMD_BATCH_CAST_INTRINSIC(uint16_t, int8_t, 32, _mm512_cvtepi16_epi8)
    XSIMD_BATCH_CAST_INTRINSIC(uint16_t, uint8_t, 32, _mm512_cvtepi16_epi8)
#endif
#if defined(XSIMD_AVX512DQ_AVAILABLE)
    XSIMD_BATCH_CAST_INTRINSIC2(int16_t, double, 8, _mm512_cvtepi16_epi64, _mm512_cvtepi64_pd)
    XSIMD_BATCH_CAST_INTRINSIC2(uint16_t, double, 8, _mm512_cvtepu16_epi64, _mm512_cvtepi64_pd)
    XSIMD_BATCH_CAST_INTRINSIC(int64_t, float, 8, _mm512_cvtepi64_ps)
    XSIMD_BATCH_CAST_INTRINSIC(int64_t, double, 8, _mm512_cvtepi64_pd)
    XSIMD_BATCH_CAST_INTRINSIC(uint64_t, float, 8, _mm512_cvtepu64_ps)
    XSIMD_BATCH_CAST_INTRINSIC(uint64_t, double, 8, _mm512_cvtepu64_pd)
    XSIMD_BATCH_CAST_INTRINSIC(float, int64_t, 8, _mm512_cvttps_epi64)
    XSIMD_BATCH_CAST_INTRINSIC(float, uint64_t, 8, _mm512_cvttps_epu64)
    XSIMD_BATCH_CAST_INTRINSIC2(double, int16_t, 8, _mm512_cvttpd_epi64, _mm512_cvtepi64_epi16)
    XSIMD_BATCH_CAST_INTRINSIC2(double, uint16_t, 8, _mm512_cvttpd_epi64, _mm512_cvtepi64_epi16)
    XSIMD_BATCH_CAST_INTRINSIC(double, int64_t, 8, _mm512_cvttpd_epi64)
    XSIMD_BATCH_CAST_INTRINSIC(double, uint64_t, 8, _mm512_cvttpd_epu64)
#endif

    inline batch<uint16_t, 32> u8_to_u16(const batch<uint8_t, 64>& x)
    {
        return static_cast<batch<uint16_t, 32>>(x);
    }
    inline batch<uint8_t, 64> u16_to_u8(const batch<uint16_t, 32>& x)
    {
        return static_cast<batch<uint8_t, 64>>(x);
    }

    inline batch<uint32_t, 16> u8_to_u32(const batch<uint8_t, 64>& x)
    {
        return static_cast<batch<uint32_t, 16>>(x);
    }
    inline batch<uint8_t, 64> u32_to_u8(const batch<uint32_t, 16>& x)
    {
        return static_cast<batch<uint8_t, 64>>(x);
    }

    inline batch<uint64_t, 8> u8_to_u64(const batch<uint8_t, 64>& x)
    {
        return static_cast<batch<uint64_t, 8>>(x);
    }

    inline batch<uint8_t, 64> u64_to_u8(const batch<uint64_t, 8>& x)
    {
        return static_cast<batch<uint8_t, 64>>(x);
    }

    /**************************
     * boolean cast functions *
     **************************/

    inline batch_bool<int32_t, 16> bool_cast(const batch_bool<float, 16>& x)
    {
        return __mmask16(x);
    }

    inline batch_bool<int64_t, 8> bool_cast(const batch_bool<double, 8>& x)
    {
        return __mmask8(x);
    }

    inline batch_bool<float, 16> bool_cast(const batch_bool<int32_t, 16>& x)
    {
        return __mmask16(x);
    }

    inline batch_bool<double, 8> bool_cast(const batch_bool<int64_t, 8>& x)
    {
        return __mmask8(x);
    }

    /*****************************************
     * bitwise cast functions implementation *
     *****************************************/

    XSIMD_BITWISE_CAST_INTRINSIC(float, 16,
                                 double, 8,
                                 _mm512_castps_pd)

    XSIMD_BITWISE_CAST_INTRINSIC(float, 16,
                                 int32_t, 16,
                                 _mm512_castps_si512)

    XSIMD_BITWISE_CAST_INTRINSIC(float, 16,
                                 int64_t, 8,
                                 _mm512_castps_si512)

    XSIMD_BITWISE_CAST_INTRINSIC(double, 8,
                                 float, 16,
                                 _mm512_castpd_ps)

    XSIMD_BITWISE_CAST_INTRINSIC(double, 8,
                                 int32_t, 16,
                                 _mm512_castpd_si512)

    XSIMD_BITWISE_CAST_INTRINSIC(double, 8,
                                 int64_t, 8,
                                 _mm512_castpd_si512)

    XSIMD_BITWISE_CAST_INTRINSIC(int32_t, 16,
                                 float, 16,
                                 _mm512_castsi512_ps)

    XSIMD_BITWISE_CAST_INTRINSIC(int32_t, 16,
                                 double, 8,
                                 _mm512_castsi512_pd)

    XSIMD_BITWISE_CAST_INTRINSIC(int64_t, 8,
                                 float, 16,
                                 _mm512_castsi512_ps)

    XSIMD_BITWISE_CAST_INTRINSIC(int64_t, 8,
                                 double, 8,
                                 _mm512_castsi512_pd)

#if defined(__clang__) || (!defined(__GNUC__) || __GNUC__ >= 9)
    /******************************************
     *  Convert Bytes, Shorts, Words, Doubles *
     *  to batch functions                    *
     ******************************************/

    void bytes_to_batch_vector(batch<uint8_t, 64>& vec,
                         int8_t b63, int8_t b62, int8_t b61, int8_t b60,
                         int8_t b59, int8_t b58, int8_t b57, int8_t b56,
                         int8_t b55, int8_t b54, int8_t b53, int8_t b52,
                         int8_t b51, int8_t b50, int8_t b49, int8_t b48,
                         int8_t b47, int8_t b46, int8_t b45, int8_t b44,
                         int8_t b43, int8_t b42, int8_t b41, int8_t b40,
                         int8_t b39, int8_t b38, int8_t b37, int8_t b36,
                         int8_t b35, int8_t b34, int8_t b33, int8_t b32,
                         int8_t b31, int8_t b30, int8_t b29, int8_t b28,
                         int8_t b27, int8_t b26, int8_t b25, int8_t b24,
                         int8_t b23, int8_t b22, int8_t b21, int8_t b20,
                         int8_t b19, int8_t b18, int8_t b17, int8_t b16,
                         int8_t b15, int8_t b14, int8_t b13, int8_t b12,
                         int8_t b11, int8_t b10, int8_t b9, int8_t b8,
                         int8_t b7, int8_t b6, int8_t b5, int8_t b4,
                         int8_t b3, int8_t b2, int8_t b1, int8_t b0);

    void shorts_to_batch_vector(batch<uint8_t, 64>& vec,
                          int16_t s31, int16_t s30, int16_t s29, int16_t s28,
                          int16_t s27, int16_t s26, int16_t s25, int16_t s24,
                          int16_t s23, int16_t s22, int16_t s21, int16_t s20,
                          int16_t s19, int16_t s18, int16_t s17, int16_t s16,
                          int16_t s15, int16_t s14, int16_t s13, int16_t s12,
                          int16_t s11, int16_t s10, int16_t s9, int16_t s8,
                          int16_t s7, int16_t s6, int16_t s5, int16_t s4,
                          int16_t s3, int16_t s2, int16_t s1, int16_t s0);

    void words_to_batch_vector(batch<uint8_t, 64>& vec,
                         int32_t i15, int32_t i14, int32_t i13, int32_t i12,
                         int32_t i11, int32_t i10, int32_t i9, int32_t i8,
                         int32_t i7, int32_t i6, int32_t i5, int32_t i4,
                         int32_t i3, int32_t i2, int32_t i1, int32_t i0);

    void longs_to_batch_vector(batch<uint8_t, 64>& vec,
                           int64_t d7, int64_t d6, int64_t d5, int64_t d4,
                           int64_t d3, int64_t d2, int64_t d1, int64_t d0);


    /*****************************************
     * vector cast functions implementation *
     *****************************************/
    inline void bytes_to_batch_vector(batch<uint8_t, 64>& vec,
                         int8_t b63, int8_t b62, int8_t b61, int8_t b60,
                         int8_t b59, int8_t b58, int8_t b57, int8_t b56,
                         int8_t b55, int8_t b54, int8_t b53, int8_t b52,
                         int8_t b51, int8_t b50, int8_t b49, int8_t b48,
                         int8_t b47, int8_t b46, int8_t b45, int8_t b44,
                         int8_t b43, int8_t b42, int8_t b41, int8_t b40,
                         int8_t b39, int8_t b38, int8_t b37, int8_t b36,
                         int8_t b35, int8_t b34, int8_t b33, int8_t b32,
                         int8_t b31, int8_t b30, int8_t b29, int8_t b28,
                         int8_t b27, int8_t b26, int8_t b25, int8_t b24,
                         int8_t b23, int8_t b22, int8_t b21, int8_t b20,
                         int8_t b19, int8_t b18, int8_t b17, int8_t b16,
                         int8_t b15, int8_t b14, int8_t b13, int8_t b12,
                         int8_t b11, int8_t b10, int8_t b9, int8_t b8,
                         int8_t b7, int8_t b6, int8_t b5, int8_t b4,
                         int8_t b3, int8_t b2, int8_t b1, int8_t b0)
    {
        vec = _mm512_set_epi8(
            b63, b62, b61, b60, b59, b58, b57, b56, b55, b54, b53, b52, b51, b50, b49, b48,
            b47, b46, b45, b44, b43, b42, b41, b40, b39, b38, b37, b36, b35, b34, b33, b32,
            b31, b30, b29, b28, b27, b26, b25, b24, b23, b22, b21, b20, b19, b18, b17, b16,
            b15, b14, b13, b12, b11, b10, b9, b8, b7, b6, b5, b4, b3, b2, b1, b0);
    }

    inline void shorts_to_batch_vector(batch<uint8_t, 64>& vec,
                          int16_t s31, int16_t s30, int16_t s29, int16_t s28,
                          int16_t s27, int16_t s26, int16_t s25, int16_t s24,
                          int16_t s23, int16_t s22, int16_t s21, int16_t s20,
                          int16_t s19, int16_t s18, int16_t s17, int16_t s16,
                          int16_t s15, int16_t s14, int16_t s13, int16_t s12,
                          int16_t s11, int16_t s10, int16_t s9, int16_t s8,
                          int16_t s7, int16_t s6, int16_t s5, int16_t s4,
                          int16_t s3, int16_t s2, int16_t s1, int16_t s0)
    {
        vec = _mm512_set_epi16(
            s31, s30, s29, s28, s27, s26, s25, s24, s23, s22, s21, s20, s19, s18, s17, s16,
            s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0);
    }

    inline void words_to_batch_vector(batch<uint8_t, 64>& vec,
                         int32_t i15, int32_t i14, int32_t i13, int32_t i12,
                         int32_t i11, int32_t i10, int32_t i9, int32_t i8,
                         int32_t i7, int32_t i6, int32_t i5, int32_t i4,
                         int32_t i3, int32_t i2, int32_t i1, int32_t i0)
    {
        vec = _mm512_set_epi32(
            i15, i14, i13, i12, i11, i10, i9, i8, i7, i6, i5, i4, i3, i2, i1, i0);
    }

    inline void longs_to_batch_vector(batch<uint8_t, 64>& vec,
                           int64_t d7, int64_t d6, int64_t d5, int64_t d4,
                           int64_t d3, int64_t d2, int64_t d1, int64_t d0)
    {
        vec = _mm512_set_epi64(d7, d6, d5, d4, d3, d2, d1, d0);
    }
#endif  // defined(__clang__) || (!defined(__GNUC__) || __GNUC__ >= 9)

}

#endif
