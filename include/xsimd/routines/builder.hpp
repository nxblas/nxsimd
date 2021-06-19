/***************************************************************************
* Copyright (c) 2016, Wolf Vollprecht, Johan Mabille and Sylvain Corlay    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_BUILDER_HPP
#define XSIMD_BUILDER_HPP

#include "xsimd/xsimd.hpp"

namespace xsimd
{
    template <class T, std::size_t N>
    inline void arange(batch<T, N>& batch, T start = 0, T step = 1)
    {
        T val = start;
        alignas(XSIMD_DEFAULT_ALIGNMENT) T xrange[N];
        for (auto it = std::begin(xrange); it != std::end(xrange); ++it)
        {
            *it = val;
            val += step;
        }
        xsimd::load_aligned(xrange, batch);
    }

    template <class I, class T>
    inline void arange(I begin, I end, T start, T step)
    {
        using value_type = std::decay_t<decltype(*begin)>;
        using traits = simd_traits<value_type>;
        using batch_type = typename traits::type;

        std::size_t size = static_cast<std::size_t>(std::distance(begin, end));
        std::size_t simd_size = traits::size;

        value_type* ptr_begin = &(*begin);
        value_type* ptr_end = &(*end);

        std::size_t align_begin = xsimd::get_alignment_offset(ptr_begin, size, simd_size);
        std::size_t align_end = align_begin + ((size - align_begin) & ~(simd_size - 1));

        for (std::size_t i = 0; i < align_begin; ++i)
        {
            *ptr_begin = start;
            ++ptr_begin;
            start += step;
        }

        batch_type brange, bstep(step * traits::size);
        arange(brange, start, step);

        for (std::size_t i = align_begin; i < align_end; i += simd_size)
        {
            xsimd::store_aligned(ptr_begin, brange);
            brange += bstep;
            ptr_begin += simd_size;
        }

        start = *(ptr_begin - 1) + step;
        for (std::size_t i = align_end; i < size; ++i)
        {
            *ptr_begin = start;
            ++ptr_begin;
            start += step;
        }
    }

    template <class I>
    inline void arange(I begin, I end)
    {
        arange(begin, end, 0, 1);
    }
}
#endif