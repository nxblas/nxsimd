/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_AVX_CONVERSION_HPP
#define XSIMD_AVX_CONVERSION_HPP

#include "xsimd_avx_double.hpp"
#include "xsimd_avx_float.hpp"
#include "xsimd_avx_int32.hpp"
#include "xsimd_avx_int64.hpp"

namespace xsimd
{

    /************************
     * conversion functions *
     ************************/

    batch<int32_t, 8> to_int(const batch<float, 8>& x);
    batch<int64_t, 4> to_int(const batch<double, 4>& x);

    batch<float, 8> to_float(const batch<int32_t, 8>& x);
    batch<double, 4> to_float(const batch<int64_t, 4>& x);

    /**************************
     * boolean cast functions *
     **************************/

    batch_bool<int32_t, 8> bool_cast(const batch_bool<float, 8>& x);
    batch_bool<int64_t, 4> bool_cast(const batch_bool<double, 4>& x);
    batch_bool<float, 8> bool_cast(const batch_bool<int32_t, 8>& x);
    batch_bool<double, 4> bool_cast(const batch_bool<int64_t, 4>& x);

    /**************************
     * bitwise cast functions *
     **************************/

    template <class B>
    B bitwise_cast(const batch<float, 8>& x);

    template <class B>
    B bitwise_cast(const batch<double, 4>& x);

    template <class B>
    B bitwise_cast(const batch<int32_t, 8>& x);

    template <class B>
    B bitwise_cast(const batch<int64_t, 4>& x);

    /***************************************
     * conversion functions implementation *
     ***************************************/

    inline batch<int32_t, 8> to_int(const batch<float, 8>& x)
    {
        return _mm256_cvttps_epi32(x);
    }

    inline batch<int64_t, 4> to_int(const batch<double, 4>& x)
    {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX2_VERSION
        return _mm256_cvtepi32_epi64(_mm256_cvttpd_epi32(x));
#else
        using batch_int = batch<int32_t, 4>;
        __m128i tmp = _mm256_cvttpd_epi32(x);
        __m128i res_low = _mm_unpacklo_epi32(tmp, batch_int(tmp) < batch_int(0));
        __m128i res_high = _mm_unpackhi_epi32(tmp, batch_int(tmp) < batch_int(0));
        __m256i result = _mm256_castsi128_si256(res_low);
        return _mm256_insertf128_si256(result, res_high, 1);
#endif
    }

    inline batch<float, 8> to_float(const batch<int32_t, 8>& x)
    {
        return _mm256_cvtepi32_ps(x);
    }

    inline batch<double, 4> to_float(const batch<int64_t, 4>& x)
    {
        return batch<double, 4>(static_cast<double>(x[0]),
                                static_cast<double>(x[1]),
                                static_cast<double>(x[2]),
                                static_cast<double>(x[3]));
    }

    /**************************
     * boolean cast functions *
     **************************/

    inline batch_bool<int32_t, 8> bool_cast(const batch_bool<float, 8>& x)
    {
        return _mm256_castps_si256(x);
    }

    inline batch_bool<int64_t, 4> bool_cast(const batch_bool<double, 4>& x)
    {
        return _mm256_castpd_si256(x);
    }

    inline batch_bool<float, 8> bool_cast(const batch_bool<int32_t, 8>& x)
    {
        return _mm256_castsi256_ps(x);
    }

    inline batch_bool<double, 4> bool_cast(const batch_bool<int64_t, 4>& x)
    {
        return _mm256_castsi256_pd(x);
    }

    /*****************************************
     * bitwise cast functions implementation *
     *****************************************/

    template <>
    inline batch<double, 4> bitwise_cast(const batch<float, 8>& x)
    {
        return _mm256_castps_pd(x);
    }

    template <>
    inline batch<int32_t, 8> bitwise_cast(const batch<float, 8>& x)
    {
        return _mm256_castps_si256(x);
    }

    template <>
    inline batch<int64_t, 4> bitwise_cast(const batch<float, 8>& x)
    {
        return _mm256_castps_si256(x);
    }

    template <>
    inline batch<int64_t, 4> bitwise_cast(const batch<int32_t, 8>& x)
    {
        return __m256i(x);
    }

    template <>
    inline batch<int32_t, 8> bitwise_cast(const batch<int64_t, 4>& x)
    {
        return __m256i(x);
    }

    template <>
    inline batch<float, 8> bitwise_cast(const batch<double, 4>& x)
    {
        return _mm256_castpd_ps(x);
    }

    template <>
    inline batch<int32_t, 8> bitwise_cast(const batch<double, 4>& x)
    {
        return _mm256_castpd_si256(x);
    }

    template <>
    inline batch<int64_t, 4> bitwise_cast(const batch<double, 4>& x)
    {
        return _mm256_castpd_si256(x);
    }

    template <>
    inline batch<double, 4> bitwise_cast(const batch<double, 4>& x)
    {
        return x;
    }

    template <>
    inline batch<float, 8> bitwise_cast(const batch<int32_t, 8>& x)
    {
        return _mm256_castsi256_ps(x);
    }

    template <>
    inline batch<double, 4> bitwise_cast(const batch<int32_t, 8>& x)
    {
        return _mm256_castsi256_pd(x);
    }

    template <>
    inline batch<float, 8> bitwise_cast(const batch<int64_t, 4>& x)
    {
        return _mm256_castsi256_ps(x);
    }

    template <>
    inline batch<int64_t, 4> bitwise_cast(const batch<int64_t, 4>& x)
    {
        return x;
    }

    template <>
    inline batch<double, 4> bitwise_cast(const batch<int64_t, 4>& x)
    {
        return _mm256_castsi256_pd(x);
    }
}

#endif
