/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_AVX_SHUFFLING_HPP
#define XSIMD_AVX_SHUFFLING_HPP

#include "xsimd/types/xsimd_avx_conversion.hpp"

namespace xsimd 
{
    // Version of select(a, b) with compile time mask
    template <class B>
    struct selector_immediate;

    template <class T, std::size_t N>
    struct selector_immediate<batch<T, N>>
    {
        using type = int32_t;
    };

    template <typename selector_immediate<batch<float, 8>>::type P, class T>
    batch<T, 8> select(const batch<T, 8>& a, const batch<T, 8>& b);

    template <typename selector_immediate<batch<double, 4>>::type P, class T>
    batch<T, 4> select(const batch<T, 4>& a, const batch<T, 4>& b);

    constexpr static int AaBbCcDd = 1;
    constexpr static int AaCc = 2;
    constexpr static int BbDd = 3;
    constexpr static int AaBbEeFf = 4;
    constexpr static int ABabEFef = 5;
    constexpr static int CcDdGgHh = 6;
    constexpr static int CDcdGHgh = 7;

    template <uint64_t P, int B = -1, class T, size_t N>
    auto interleave(const batch<T, N>& a, const batch<T, N>& b);

    namespace detail
    {
        template <uint64_t P, int B, class T, size_t N>
        struct interleave_impl;

        template <class T, size_t N>
        struct interleave_impl<AaCc, 64, T, N>
        {
            batch<T, N> operator()(const batch<T, N>& a, const batch<T, N>& b)
            {
                // {a, b, c, d} Δ {A, B, C, D} → {a, A, c, C}
                batch<int64_t, 4> tmp = _mm256_unpacklo_epi64(bitwise_cast<batch<int64_t, 4>>(a),
                                                              bitwise_cast<batch<int64_t, 4>>(b));
                return bitwise_cast<batch<T, N>>(tmp);
            }
        };

        template <class T, size_t N>
        struct interleave_impl<BbDd, 64, T, N>
        {
            batch<T, N> operator()(const batch<T, N>& a, const batch<T, N>& b)
            {
                // {a, b, c, d} Δ {A, B, C, D} → {b, B, d, D}
                batch<int64_t, 4> tmp = _mm256_unpackhi_epi64(bitwise_cast<batch<int64_t, 4>>(a),
                                                              bitwise_cast<batch<int64_t, 4>>(b));
                return bitwise_cast<batch<T, N>>(tmp);
            }
        };
            // Needs other name?
        template <class T, size_t N>
        struct interleave_impl<AaBbCcDd, 64, T, N>
        {
            std::pair<batch<T, N>, batch<T, N>> operator()(const batch<T, N>& a, const batch<T, N>& b)
            {
                // {a, b, c, d} Δ {A, B, C, D} → std::pair{{a, A, b, B}, {c, C, d, D}}
                auto lo = interleave_impl<AaCc, 64, int64_t, 4>{}(bitwise_cast<batch<int64_t, 4>>(a), bitwise_cast<batch<int64_t, 4>>(b));
                auto hi = interleave_impl<BbDd, 64, int64_t, 4>{}(bitwise_cast<batch<int64_t, 4>>(a), bitwise_cast<batch<int64_t, 4>>(b));
                batch<int64_t, 4> tmp_up = _mm256_permute2f128_si256(lo, hi, _MM_SHUFFLE(0, 2, 0, 0));
                batch<int64_t, 4> tmp_bt = _mm256_permute2f128_si256(lo, hi, _MM_SHUFFLE(0, 3, 0, 1));
                return {
                    bitwise_cast<batch<T, N>>(tmp_up),
                    bitwise_cast<batch<T, N>>(tmp_bt)
                };
            }
        };

        template <class T, size_t N>
        struct interleave_impl<AaBbEeFf, 32, T, N>
        {
            batch<T, N> operator()(const batch<T, N>& a, const batch<T, N>& b)
            {
                return _mm256_unpacklo_epi32(a, b);
            }
        };

        template <class T, size_t N>
        struct interleave_impl<CcDdGgHh, 32, T, N>
        {
            batch<T, N> operator()(const batch<T, N>& a, const batch<T, N>& b)
            {
                return _mm256_unpackhi_epi32(a, b);
            }
        };

        template <class T, size_t N>
        struct interleave_impl<CDcdGHgh, 32, T, N>
            : interleave_impl<BbDd, 64, T, N>
        {
            using interleave_impl<BbDd, 64, T, N>::operator();
        };

        template <class T, size_t N>
        struct interleave_impl<ABabEFef, 32, T, N>
            : interleave_impl<AaCc, 64, T, N>
        {
            using interleave_impl<AaCc, 64, T, N>::operator();
        };
    }

    template <uint64_t P, int B = -1, class T, size_t N>
    auto interleave(const batch<T, N>& a, const batch<T, N>& b)
    {
        constexpr int BC = B == -1 ? sizeof(T) * 8 : B;
        return detail::interleave_impl<P, BC, T, N>{}(a, b);
    }

    template <typename selector_immediate<batch<float, 8>>::type P, class T>
    batch<T, 8> select(batch<T, 8>& a, batch<T, 8>& b)
    {
        auto res = _mm256_blend_ps(bitwise_cast<batch<float, 8>>(a), 
                                   bitwise_cast<batch<float, 8>>(b),
                                   P);
        return bitwise_cast<batch<T, 8>>(res);
    }

    template <typename selector_immediate<batch<double, 4>>::type P, class T>
    batch<T, 4> select(batch<T, 4>& a, batch<T, 4>& b)
    {
        auto res = _mm256_blend_pd(bitwise_cast<batch<double, 4>>(a), 
                                   bitwise_cast<batch<double, 4>>(b),
                                   P);
        return bitwise_cast<batch<T, 4>>(res);
    }
}

#endif