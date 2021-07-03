/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay, Wolf Vollprecht and         *
* Martin Renou                                                             *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include <numeric>
#include "test_utils.hpp"
#include <xsimd/xsimd.hpp>
#include <xsimd/types/xsimd_fallback.hpp>
#include <xsimd/types/xsimd_traits.hpp>
#include <xsimd/types/xsimd_base.hpp>

struct binary_functor
{
    template <class T>
    T operator()(const T& a, const T& b) const
    {
        return a + b;
    }
};

struct unary_functor
{
    template <class T>
    T operator()(const T& a) const
    {
        return -a;
    }
};

template <class T>
using test_allocator_type = xsimd::aligned_allocator<T>;

template <class C>
struct types {
    using value_type = typename std::decay<decltype(*C().begin())>::type;
    using traits = xsimd::simd_traits<value_type>;
    using batch_type = typename traits::type;
};

#if XSIMD_X86_INSTR_SET > XSIMD_VERSION_NUMBER_NOT_AVAILABLE || XSIMD_ARM_INSTR_SET > XSIMD_VERSION_NUMBER_NOT_AVAILABLE
TEST(algorithms, unary_transform_batch)
{
    using vector_type = std::vector<int, test_allocator_type<int>>;
    using batch_type = types<vector_type>::batch_type;
    auto flip_flop = vector_type(42, 0);
    std::iota(flip_flop.begin(), flip_flop.end(), 1);
    auto square_pair = [](int x) {
        return !(x % 2) ? x : x*x;
    };
    auto flip_flop_axpected = flip_flop;
    std::transform(flip_flop_axpected.begin(), flip_flop_axpected.end(), flip_flop_axpected.begin(), square_pair);

    xsimd::transform(flip_flop.begin(), flip_flop.end(), flip_flop.begin(),
    // NOTE: since c++14 a simple `[](auto x)` reduce code complexity
    [](int x) {
        return !(x % 2) ? x : x*x;
    },
    // NOTE: since c++14 a simple `[](auto b)` reduce code complexity
    [](batch_type b) {
        return xsimd::select(!(b % 2), b, b*b);
    });
    EXPECT_TRUE(std::equal(flip_flop_axpected.begin(), flip_flop_axpected.end(), flip_flop.begin()) && flip_flop_axpected.size() == flip_flop.size());
}

TEST(algorithms, binary_transform_batch)
{
    using vector_type = std::vector<int, test_allocator_type<int>>;
    using batch_type = types<vector_type>::batch_type;
    auto flip_flop_a = vector_type(42, 0);
    auto flip_flop_b = vector_type(42, 0);
    std::iota(flip_flop_a.begin(), flip_flop_a.end(), 1);
    std::iota(flip_flop_b.begin(), flip_flop_b.end(), 3);
    auto square_pair = [](int x, int y) {
        return !((x + y) % 2) ? x + y : x*x + y*y;
    };
    auto flip_flop_axpected = flip_flop_a;
    std::transform(flip_flop_a.begin(), flip_flop_a.end(), flip_flop_b.begin(), flip_flop_axpected.begin(), square_pair);

    auto flip_flop_result = vector_type(flip_flop_axpected.size());
    xsimd::transform(flip_flop_a.begin(), flip_flop_a.end(), flip_flop_b.begin(), flip_flop_result.begin(),
    [](int x, int y) {
        return !((x +y) % 2) ? x + y : x*x + y*y;
    },
    [](batch_type bx, batch_type by) {
        return xsimd::select(!((bx + by) % 2), bx + by, bx*bx + by+by);
    });
    EXPECT_TRUE(std::equal(flip_flop_axpected.begin(), flip_flop_axpected.end(), flip_flop_result.begin()) && flip_flop_axpected.size() == flip_flop_result.size());
}
#endif

TEST(algorithms, binary_transform)
{
    std::vector<double> expected(93);

    std::vector<double> a(93, 123), b(93, 123), c(93);
    std::vector<double, test_allocator_type<double>> aa(93, 123), ba(93, 123), ca(93);

    std::transform(a.begin(), a.end(), b.begin(), expected.begin(),
                    binary_functor{});

    xsimd::transform(a.begin(), a.end(), b.begin(), c.begin(),
                     binary_functor{});
    EXPECT_TRUE(std::equal(expected.begin(), expected.end(), c.begin()) && expected.size() == c.size());
    std::fill(c.begin(), c.end(), -1); // erase

    xsimd::transform(aa.begin(), aa.end(), ba.begin(), c.begin(),
                     binary_functor{});
    EXPECT_TRUE(std::equal(expected.begin(), expected.end(), c.begin()) && expected.size() == c.size());
    std::fill(c.begin(), c.end(), -1); // erase

    xsimd::transform(aa.begin(), aa.end(), b.begin(), c.begin(),
                     binary_functor{});
    EXPECT_TRUE(std::equal(expected.begin(), expected.end(), c.begin()) && expected.size() == c.size());
    std::fill(c.begin(), c.end(), -1); // erase

    xsimd::transform(a.begin(), a.end(), ba.begin(), c.begin(),
                     binary_functor{});
    EXPECT_TRUE(std::equal(expected.begin(), expected.end(), c.begin()) && expected.size() == c.size());
    std::fill(c.begin(), c.end(), -1); // erase

    xsimd::transform(aa.begin(), aa.end(), ba.begin(), ca.begin(),
                     binary_functor{});
    EXPECT_TRUE(std::equal(expected.begin(), expected.end(), ca.begin()) && expected.size() == ca.size());
    std::fill(ca.begin(), ca.end(), -1); // erase

    xsimd::transform(aa.begin(), aa.end(), b.begin(), ca.begin(),
                     binary_functor{});
    EXPECT_TRUE(std::equal(expected.begin(), expected.end(), ca.begin()) && expected.size() == ca.size());
    std::fill(ca.begin(), ca.end(), -1); // erase

    xsimd::transform(a.begin(), a.end(), ba.begin(), ca.begin(),
                     binary_functor{});
    EXPECT_TRUE(std::equal(expected.begin(), expected.end(), ca.begin()) && expected.size() == ca.size());
    std::fill(ca.begin(), ca.end(), -1); // erase
}

TEST(algorithms, unary_transform)
{
    std::vector<double> expected(93);
    std::vector<double> a(93, 123), c(93);
    std::vector<double, test_allocator_type<double>> aa(93, 123), ca(93);

    std::transform(a.begin(), a.end(), expected.begin(),
                   unary_functor{});

    xsimd::transform(a.begin(), a.end(), c.begin(),
                     unary_functor{});
    EXPECT_TRUE(std::equal(expected.begin(), expected.end(), c.begin()) && expected.size() == c.size());
    std::fill(c.begin(), c.end(), -1); // erase

    xsimd::transform(aa.begin(), aa.end(), c.begin(),
                     unary_functor{});
    EXPECT_TRUE(std::equal(expected.begin(), expected.end(), c.begin()) && expected.size() == c.size());
    std::fill(c.begin(), c.end(), -1); // erase

    xsimd::transform(a.begin(), a.end(), ca.begin(),
                     unary_functor{});
    EXPECT_TRUE(std::equal(expected.begin(), expected.end(), ca.begin()) && expected.size() == ca.size());
    std::fill(ca.begin(), ca.end(), -1); // erase

    xsimd::transform(aa.begin(), aa.end(), ca.begin(),
                     unary_functor{});
    EXPECT_TRUE(std::equal(expected.begin(), expected.end(), ca.begin()) && expected.size() == ca.size());
    std::fill(ca.begin(), ca.end(), -1); // erase
}

class xsimd_reduce : public ::testing::Test
{
public:
    using aligned_vec_t = std::vector<double, test_allocator_type<double>>;

    static constexpr std::size_t num_elements = 4 * xsimd::simd_traits<double>::size;
    static constexpr std::size_t small_num = xsimd::simd_traits<double>::size - 1;

    aligned_vec_t vec = aligned_vec_t(num_elements, 123.);
    aligned_vec_t small_vec = aligned_vec_t(small_num, 42.); 
    double        init = 1337.;

    struct multiply
    {
        template <class T>
        T operator()(const T& a, const T& b) const
        {
            return a * b;
        }
    };
};

TEST_F(xsimd_reduce, unaligned_begin_unaligned_end)
{
    auto const begin = std::next(vec.begin());
    auto const end = std::prev(vec.end());

    EXPECT_EQ(std::accumulate(begin, end, init), xsimd::reduce(begin, end, init));

    if(small_vec.size() > 1)
    {
        auto const sbegin = std::next(small_vec.begin());
        auto const send = std::prev(small_vec.end());

        EXPECT_EQ(std::accumulate(sbegin, send, init), xsimd::reduce(sbegin, send, init));
    }
}

TEST_F(xsimd_reduce, unaligned_begin_aligned_end)
{
    auto const begin = std::next(vec.begin());
    auto const end = vec.end();

    EXPECT_EQ(std::accumulate(begin, end, init), xsimd::reduce(begin, end, init));

    if(small_vec.size() > 1)
    {
        auto const sbegin = std::next(small_vec.begin());
        auto const send = small_vec.end();

        EXPECT_EQ(std::accumulate(sbegin, send, init), xsimd::reduce(sbegin, send, init));
    }
}

TEST_F(xsimd_reduce, aligned_begin_unaligned_end)
{
    auto const begin = vec.begin();
    auto const end = std::prev(vec.end());

    EXPECT_EQ(std::accumulate(begin, end, init), xsimd::reduce(begin, end, init));

    if(small_vec.size() > 1)
    {
        auto const sbegin = small_vec.begin();
        auto const send = std::prev(small_vec.end());

        EXPECT_EQ(std::accumulate(sbegin, send, init), xsimd::reduce(sbegin, send, init));
    }
}

TEST_F(xsimd_reduce, aligned_begin_aligned_end)
{
    auto const begin = vec.begin();
    auto const end = vec.end();

    EXPECT_EQ(std::accumulate(begin, end, init), xsimd::reduce(begin, end, init));

    if(small_vec.size() > 1)
    {
        auto const sbegin = small_vec.begin();
        auto const send = small_vec.end();

        EXPECT_EQ(std::accumulate(sbegin, send, init), xsimd::reduce(sbegin, send, init));
    }
}

TEST_F(xsimd_reduce, using_custom_binary_function)
{
    auto const begin = vec.begin();
    auto const end = vec.end();

    EXPECT_DOUBLE_EQ(std::accumulate(begin, end, init, multiply{}), xsimd::reduce(begin, end, init, multiply{}));

    if(small_vec.size() > 1)
    {
        auto const sbegin = small_vec.begin();
        auto const send = small_vec.end();

        EXPECT_DOUBLE_EQ(std::accumulate(sbegin, send, init, multiply{}), xsimd::reduce(sbegin, send, init, multiply{}));
    }
}

TEST(algorithms, reduce_batch)
{
    const float nan = std::numeric_limits<float>::quiet_NaN();
    using vector_type = std::vector<float, test_allocator_type<float>>;
    using batch_type = types<vector_type>::batch_type;
    auto vector_with_nan = vector_type(1000, 0);
    std::iota(vector_with_nan.begin(), vector_with_nan.end(), 3.5);
    auto i = 0;
    auto add_nan = [&i, &nan](const float x) {
        return i % 2 ? nan : x;
    };
    std::transform(vector_with_nan.begin(), vector_with_nan.end(), vector_with_nan.begin(), add_nan);

    auto nansum_expected = std::accumulate(vector_with_nan.begin(), vector_with_nan.end(), 0.0,
    [](float x, float y) {
        return (std::isnan(x) ? 0.0 : x) + (std::isnan(y) ? 0.0 : y);
    });

    auto nansum = xsimd::reduce(vector_with_nan.begin(), vector_with_nan.end(), 0.0, 
    [](float x, float y) {
        return (std::isnan(x) ? 0.0 : x) + (std::isnan(y) ? 0.0 : y);
    },
    [](batch_type x, batch_type y) {
        static batch_type zero(0.0);
        auto xnan = xsimd::isnan(x);
        auto ynan = xsimd::isnan(y);
        auto xs = xsimd::select(xnan, zero, x);
        auto ys = xsimd::select(ynan, zero, y);
        return xs + ys;
    });

    EXPECT_NEAR(nansum_expected, nansum, 1e-6);

    auto count_nan_expected = std::count_if(vector_with_nan.begin(), vector_with_nan.end(),
    [](float x){
        return static_cast<std::size_t>(std::isnan(x));
    });

    auto count_nan = xsimd::count_if(vector_with_nan.begin(), vector_with_nan.end(),
    [](float x){
        return static_cast<std::size_t>(std::isnan(x));
    },
    [](batch_type b) {
        static decltype(b) zero(0.0);
        static decltype(b) one(1.0);
        auto bnan = xsimd::isnan(b);
        return static_cast<std::size_t>(xsimd::hadd(xsimd::select(bnan, one, zero)));
    });

    EXPECT_EQ(count_nan_expected, count_nan);

    auto count_not_nan_expected = vector_with_nan.size() - count_nan_expected;
    auto count_not_nan = vector_with_nan.size() - count_nan;

    auto nanmean_expected = count_not_nan_expected ? nansum_expected / (float)count_not_nan_expected : 0;
    auto nanmean = count_not_nan ? nansum / (float)count_not_nan : 0;

    EXPECT_NEAR(nanmean_expected, nanmean, 1e-6);
}

TEST(algorithms, count)
{
    using vector_type = std::vector<float, test_allocator_type<float>>;
    auto v = vector_type(100, 0);
    std::iota(v.begin(), v.end(), 3.14);
    v[12] = 123.321;
    v[42] = 123.321;
    v[93] = 123.321;

    EXPECT_EQ(3, xsimd::count(v.begin(), v.end(), 123.321));
}

TEST(algorithms, count_if)
{
    using vector_type = std::vector<int, test_allocator_type<int>>;
    using batch_type = types<vector_type>::batch_type;
    auto v = vector_type(100, 0);
    std::iota(v.begin(), v.end(), 1);

    auto count_expected = std::count_if(v.begin(), v.end(), 
    [](int x) {
        return x >= 50 && x <= 70 ? 1 : 0;
    });

    auto count = xsimd::count_if(v.begin(), v.end(), 
    [](int x) {
        return x >= 50 && x <= 70 ? 1 : 0;
    },
    [](batch_type b) {
        static batch_type zero(0);
        static batch_type one(1);
        return xsimd::hadd(xsimd::select(b >= 50 && b <= 70, one, zero));
    });
    EXPECT_EQ(count_expected, count);
}

#if XSIMD_X86_INSTR_SET > XSIMD_VERSION_NUMBER_NOT_AVAILABLE || XSIMD_ARM_INSTR_SET > XSIMD_VERSION_NUMBER_NOT_AVAILABLE
TEST(algorithms, iterator)
{
    std::vector<float, test_allocator_type<float>> a(10 * 16, 0.2), b(1000, 2.), c(1000, 3.);

    std::iota(a.begin(), a.end(), 0.f);
    std::vector<float> a_cpy(a.begin(), a.end());

    using batch_type = typename xsimd::simd_traits<float>::type;
    auto begin = xsimd::aligned_iterator<batch_type>(&a[0]);
    auto end = xsimd::aligned_iterator<batch_type>(&a[0] + a.size());
 
    for (; begin != end; ++begin)
    {
        *begin = *begin / 2.f;
    }

    for (auto& el : a_cpy)
    {
        el /= 2.f;
    }

    EXPECT_TRUE(a.size() == a_cpy.size() && std::equal(a.begin(), a.end(), a_cpy.begin()));

    begin = xsimd::aligned_iterator<batch_type>(&a[0]);
    *begin = sin(*begin);

    for (std::size_t i = 0; i < batch_type::size; ++i)
    {
        EXPECT_NEAR(a[i], sinf(a_cpy[i]), 1e-6);
    }

#ifdef XSIMD_BATCH_DOUBLE_SIZE
    std::vector<std::complex<double>, test_allocator_type<std::complex<double>>> ca(10 * 16, std::complex<double>(0.2));
    using cbatch_type = typename xsimd::simd_traits<std::complex<double>>::type;
    auto cbegin = xsimd::aligned_iterator<cbatch_type>(&ca[0]);
    auto cend = xsimd::aligned_iterator<cbatch_type>(&ca[0] + a.size());

    for (; cbegin != cend; ++cbegin)
    {
        *cbegin = (*cbegin + std::complex<double>(0, .3)) / 2.;
    }

    cbegin = xsimd::aligned_iterator<cbatch_type>(&ca[0]);
    *cbegin = sin(*cbegin);
    *cbegin = sqrt(*cbegin);
    auto real_part = abs(*(cbegin));
    (void)real_part;
#endif

}
#endif
