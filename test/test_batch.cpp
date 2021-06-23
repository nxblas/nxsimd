/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay, Wolf Vollprecht and         *
* Martin Renou                                                             *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include <cmath>
#include <functional>
#include <numeric>

#include "test_utils.hpp"

using namespace std::placeholders;

template <class B>
class batch_test : public testing::Test
{
public:

    using batch_type = B;
    using value_type = typename B::value_type;
    static constexpr size_t size = B::size;
    using array_type = std::array<value_type, size>;
    using bool_array_type = std::array<bool, size>;

    array_type lhs;
    array_type rhs;
    value_type scalar;

    batch_test()
    {
        init_operands();
    }

    void test_load_store() const
    {
        array_type res;
        batch_type b;
        b.load_unaligned(lhs.data());
        b.store_unaligned(res.data());
        {
            INFO( print_function_name("load_unaligned / store_unaligned"));
            EXPECT_EQ(res, lhs);
        }

        alignas(xsimd::arch::default_::alignment) array_type arhs(this->rhs);
        alignas(xsimd::arch::default_::alignment) array_type ares;
        b.load_aligned(arhs.data());
        b.store_aligned(ares.data());
        {
            INFO( print_function_name("load_aligned / store_aligned"));
            EXPECT_EQ(ares, rhs);
        }
    }

    void test_constructors() const
    {
        array_type tmp;
        std::fill(tmp.begin(), tmp.end(), value_type(2));
        batch_type b0(2);

        {
            INFO( print_function_name("batch(value_type)"));
            EXPECT_EQ(b0, tmp);
        }

        batch_type b1(lhs.data());
        {
            INFO( print_function_name("batch(value_type*)"));
            EXPECT_EQ(b1, lhs);
        }
    }

    void test_static_builders() const
    {
        {
            array_type expected;
            std::fill(expected.begin(), expected.end(), value_type(2));

            auto res = batch_type::broadcast(value_type(2));
            {
                INFO( print_function_name("batch::broadcast"));
                EXPECT_EQ(res, expected);
            }
        }
        {
            array_type res;
            auto b = batch_type::from_unaligned(lhs.data());
            b.store_unaligned(res.data());
            {
                INFO( print_function_name("batch::from_unaligned"));
                EXPECT_EQ(res, lhs);
            }
        }
        {
            alignas(xsimd::arch::default_::alignment) array_type arhs(this->rhs);
            alignas(xsimd::arch::default_::alignment) array_type ares;
            auto b = batch_type::from_aligned(arhs.data());
            b.store_aligned(ares.data());
            {
                INFO( print_function_name("batch::from_aligned"));
                EXPECT_EQ(ares, rhs);
            }
        }
    }

    void test_access_operator() const
    {
        batch_type res = batch_lhs();
        for (size_t i = 0; i < size; ++i)
        {
            {
                INFO(print_function_name("operator[]("),i, ")");
                EXPECT_EQ(res[i], lhs[i]);
            }
        }
    }

    void test_arithmetic() const
    {
        // +batch
        {
            array_type expected = lhs;
            batch_type res = +batch_lhs();
            {
                INFO(print_function_name("+batch"));
                EXPECT_BATCH_EQ(res, expected);
            }
        }
        // -batch
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(), std::negate<value_type>());
            batch_type res = -batch_lhs();
            {
                INFO(print_function_name("-batch"));
                EXPECT_BATCH_EQ(res, expected);
            }
        }
        // batch + batch
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), expected.begin(), std::plus<value_type>());
            batch_type res = batch_lhs() + batch_rhs();
            {
                INFO(print_function_name("batch + batch"));
                EXPECT_BATCH_EQ(res, expected);
            }
        }
        // batch + scalar
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(), std::bind(std::plus<value_type>(), _1, scalar));
            batch_type lres = batch_lhs() + scalar;
            {
                INFO(print_function_name("batch + scalar"));
                EXPECT_BATCH_EQ(lres, expected);
            }
            batch_type rres = scalar + batch_lhs();
            {
                INFO(print_function_name("scalar + batch"));
                EXPECT_BATCH_EQ(rres, expected);
            }
        }
        // batch - batch
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), expected.begin(), std::minus<value_type>());
            batch_type res = batch_lhs() - batch_rhs();
            {
                INFO(print_function_name("batch - batch"));
                EXPECT_BATCH_EQ(res, expected);
            }
        }
        // batch - scalar
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(), std::bind(std::minus<value_type>(), _1, scalar));
            batch_type lres = batch_lhs() - scalar;
            {
                INFO(print_function_name("batch - scalar"));
                EXPECT_BATCH_EQ(lres, expected);
            }
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(), std::bind(std::minus<value_type>(), scalar, _1));
            batch_type rres = scalar - batch_lhs();
            {
                INFO(print_function_name("scalar - batch"));
                EXPECT_BATCH_EQ(rres, expected);
            }
        }
        // batch * batch
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), expected.begin(), std::multiplies<value_type>());
            batch_type res = batch_lhs() * batch_rhs();
            {
                INFO(print_function_name("batch * batch"));
                EXPECT_BATCH_EQ(res, expected);
            }
        }
        // batch * scalar
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(), std::bind(std::multiplies<value_type>(), _1, scalar));
            batch_type lres = batch_lhs() * scalar;
            {
                INFO(print_function_name("batch * scalar"));
                EXPECT_BATCH_EQ(lres, expected);
            }
            batch_type rres = scalar * batch_lhs();
            {
                INFO(print_function_name("scalar * batch"));
                EXPECT_BATCH_EQ(rres, expected);
            }
        }
        // batch / batch
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), expected.begin(), std::divides<value_type>());
            batch_type res = batch_lhs() / batch_rhs();
            {
                INFO(print_function_name("batch / batch"));
                EXPECT_BATCH_EQ(res, expected);
            }
        }
        // batch / scalar
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(), std::bind(std::divides<value_type>(), _1, scalar));
            batch_type lres = batch_lhs() / scalar;
            {
                INFO(print_function_name("batch / scalar"));
                EXPECT_BATCH_EQ(lres, expected);
            }
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(), std::bind(std::divides<value_type>(), scalar, _1));
            batch_type rres = scalar / batch_lhs();
            {
                INFO(print_function_name("scalar / batch"));
                EXPECT_BATCH_EQ(rres, expected);
            }
        }
    }

    void test_saturated_arithmetic() const
    {
        // batch + batch
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), expected.begin(), xsimd::sadd<value_type>);
            batch_type res = xsimd::sadd(batch_lhs(), batch_rhs());
            {
                INFO(print_function_name("sadd(batch, batch)"));
                EXPECT_BATCH_EQ(res, expected);
            }
        }
        // batch + scalar
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(), std::bind(xsimd::sadd<value_type>, _1, scalar));
            batch_type lres = xsimd::sadd(batch_lhs(), scalar);
            {
                INFO(print_function_name("sadd(batch, scalar)"));
                EXPECT_BATCH_EQ(lres, expected);
            }
            batch_type rres = xsimd::sadd(scalar, batch_lhs());
            {
                INFO(print_function_name("sadd(scalar, batch)"));
                EXPECT_BATCH_EQ(rres, expected);
            }
        }
        // batch - batch
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), expected.begin(), xsimd::ssub<value_type>);
            batch_type res = xsimd::ssub(batch_lhs(), batch_rhs());
            {
                INFO(print_function_name("ssub(batch, batch)"));
                EXPECT_BATCH_EQ(res, expected);
            }
        }
        // batch - scalar
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(), std::bind(xsimd::ssub<value_type>, _1, scalar));
            batch_type lres = xsimd::ssub(batch_lhs(), scalar);
            {
                INFO(print_function_name("ssub(batch, scalar)"));
                EXPECT_BATCH_EQ(lres, expected);
            }
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(), std::bind(xsimd::ssub<value_type>, scalar, _1));
            batch_type rres = xsimd::ssub(scalar, batch_lhs());
            {
                INFO(print_function_name("ssub(scalar, batch)"));
                EXPECT_BATCH_EQ(rres, expected);
            }
        }
    }

    void test_computed_assignment() const
    {
        // batch += batch
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), expected.begin(), std::plus<value_type>());
            batch_type res = batch_lhs();
            res += batch_rhs();
            {
                INFO(print_function_name("batch += batch"));
                EXPECT_BATCH_EQ(res, expected);
            }
        }
        // batch += scalar
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(), std::bind(std::plus<value_type>(), _1, scalar));
            batch_type res = batch_lhs(); 
            res += scalar;
            {
                INFO(print_function_name("batch += scalar"));
                EXPECT_BATCH_EQ(res, expected);
            }
        }
        // batch -= batch
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), expected.begin(), std::minus<value_type>());
            batch_type res = batch_lhs();
            res -= batch_rhs();
            {
                INFO(print_function_name("batch -= batch"));
                EXPECT_BATCH_EQ(res, expected);
            }
        }
        // batch -= scalar
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(), std::bind(std::minus<value_type>(), _1, scalar));
            batch_type res = batch_lhs();
            res -= scalar;
            {
                INFO(print_function_name("batch -= scalar"));
                EXPECT_BATCH_EQ(res, expected);
            }
        }
        // batch *= batch
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), expected.begin(), std::multiplies<value_type>());
            batch_type res = batch_lhs();
            res *= batch_rhs();
            {
                INFO(print_function_name("batch *= batch"));
                EXPECT_BATCH_EQ(res, expected);
            }
        }
        // batch *= scalar
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(), std::bind(std::multiplies<value_type>(), _1, scalar));
            batch_type res = batch_lhs();
            res *= scalar;
            {
                INFO(print_function_name("batch *= scalar"));
                EXPECT_BATCH_EQ(res, expected);
            }
        }
        // batch /= batch
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), expected.begin(), std::divides<value_type>());
            batch_type res = batch_lhs();
            res /= batch_rhs();
            {
                INFO(print_function_name("batch /= batch"));
                EXPECT_BATCH_EQ(res, expected);
            }
        }
        // batch /= scalar
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(), std::bind(std::divides<value_type>(), _1, scalar));
            batch_type res = batch_lhs();
            res /= scalar;
            {
                INFO(print_function_name("batch /= scalar"));
                EXPECT_BATCH_EQ(res, expected);
            }
        }
    }

    void test_comparison() const
    {
        // batch == batch
        {
            bool_array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), expected.begin(),
                            [](const value_type& l, const value_type& r) { return l == r; });
            auto res = batch_lhs() == batch_rhs();
            {
                INFO(print_function_name("batch == batch"));
                EXPECT_BATCH_EQ(res, expected);
            }
        }
        // batch == scalar
        {
            bool_array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(),
                            [this](const value_type& l) { return l == scalar; });
            auto res = batch_lhs() == scalar;
            {
                INFO(print_function_name("batch == scalar"));
                EXPECT_BATCH_EQ(res, expected);
            }
        }
        // batch != batch
        {
            bool_array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), expected.begin(),
                            [](const value_type& l, const value_type& r) { return l != r; });
            auto res = batch_lhs() != batch_rhs();
            {
                INFO(print_function_name("batch != batch"));
                EXPECT_BATCH_EQ(res, expected);
            }
        }
        // batch != scalar
        {
            bool_array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(),
                            [this](const value_type& l) { return l != scalar; });
            auto res = batch_lhs() != scalar;
            {
                INFO(print_function_name("batch != scalar"));
                EXPECT_BATCH_EQ(res, expected);
            }
        }
        // batch < batch
        {
            bool_array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), expected.begin(),
                            [](const value_type& l, const value_type& r) { return l < r; });
            auto res = batch_lhs() < batch_rhs();
            {
                INFO(print_function_name("batch < batch"));
                EXPECT_BATCH_EQ(res, expected);
            }
        }
        // batch < scalar
        {
            bool_array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(),
                            [this](const value_type& l) { return l < scalar; });
            auto res = batch_lhs() < scalar;
            {
                INFO(print_function_name("batch < scalar"));
                EXPECT_BATCH_EQ(res, expected);
            }
        }
        // batch <= batch
        {
            bool_array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), expected.begin(),
                            [](const value_type& l, const value_type& r) { return l <= r; });
            auto res = batch_lhs() <= batch_rhs();
            {
                INFO(print_function_name("batch <= batch"));
                EXPECT_BATCH_EQ(res, expected);
            }
        }
        // batch <= scalar
        {
            bool_array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(),
                            [this](const value_type& l) { return l <= scalar; });
            auto res = batch_lhs() <= scalar;
            {
                INFO(print_function_name("batch <= scalar"));
                EXPECT_BATCH_EQ(res, expected);
            }
        }
        // batch > batch
        {
            bool_array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), expected.begin(),
                            [](const value_type& l, const value_type& r) { return l > r; });
            auto res = batch_lhs() > batch_rhs();
            {
                INFO(print_function_name("batch > batch"));
                EXPECT_BATCH_EQ(res, expected);
            }
        }
        // batch > scalar
        {
            bool_array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(),
                            [this](const value_type& l) { return l > scalar; });
            auto res = batch_lhs() > scalar;
            {
                INFO(print_function_name("batch > scalar"));
                EXPECT_BATCH_EQ(res, expected);
            }
        }
        // batch >= batch
        {
            bool_array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), expected.begin(),
                            [](const value_type& l, const value_type& r) { return l >= r; });
            auto res = batch_lhs() >= batch_rhs();
            {
                INFO(print_function_name("batch >= batch"));
                EXPECT_BATCH_EQ(res, expected);
            }
        }
        // batch >= scalar
        {
            bool_array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(),
                            [this](const value_type& l) { return l >= scalar; });
            auto res = batch_lhs() >= scalar;
            {
                INFO(print_function_name("batch >= scalar"));
                EXPECT_BATCH_EQ(res, expected);
            }
        }
    }

    void test_min_max() const
    {
        // min
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), expected.begin(),
                           [](const value_type& l, const value_type& r) { return std::min(l, r); });
            batch_type res = min(batch_lhs(), batch_rhs());
            {
                INFO(print_function_name("min"));
                EXPECT_BATCH_EQ(res, expected);
            }
        }
        // min limit case
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), expected.begin(),
                           [](const value_type& , const value_type& r) { return std::min(std::numeric_limits<value_type>::min(), r); });
            batch_type res = xsimd::min(batch_type(std::numeric_limits<value_type>::min()), batch_rhs());
            {
                INFO(print_function_name("min limit"));
                EXPECT_BATCH_EQ(res, expected);
            }
        }
        // fmin
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), expected.begin(),
                           [](const value_type& l, const value_type& r) { return std::fmin(l, r); });
            batch_type res = min(batch_lhs(), batch_rhs());
            {
                INFO(print_function_name("fmin"));
                EXPECT_BATCH_EQ(res, expected);
            }
        }
        // max
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), expected.begin(),
                           [](const value_type& l, const value_type& r) { return std::max(l, r); });
            batch_type res = max(batch_lhs(), batch_rhs());
            {
                INFO(print_function_name("max"));
                EXPECT_BATCH_EQ(res, expected);
            }
        }
        // max limit case
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), expected.begin(),
                           [](const value_type& , const value_type& r) { return std::max(std::numeric_limits<value_type>::max(), r); });
            batch_type res = xsimd::max(batch_type(std::numeric_limits<value_type>::max()), batch_rhs());
            {
                INFO(print_function_name("max limit"));
                EXPECT_BATCH_EQ(res, expected);
            }
        }
        // fmax
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), expected.begin(),
                           [](const value_type& l, const value_type& r) { return std::fmax(l, r); });
            batch_type res = fmax(batch_lhs(), batch_rhs());
            {
                INFO(print_function_name("fmax"));
                EXPECT_BATCH_EQ(res, expected);
            }
        }
    }

    void test_fused_operations() const
    {
        // fma
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.begin(), expected.begin(),
                            [](const value_type& l, const value_type& r) { return l * r + r; });
            // Warning: ADL seems to not work correctly on Windows, thus the full qualified call
            batch_type res = xsimd::fma(batch_lhs(), batch_rhs(), batch_rhs());
            {
                INFO(print_function_name("fma"));
                EXPECT_BATCH_EQ(res, expected);
            }
        }
        // fms
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.begin(), expected.begin(),
                            [](const value_type& l, const value_type& r) { return l * r - r; });
            batch_type res = fms(batch_lhs(), batch_rhs(), batch_rhs());
            {
                INFO(print_function_name("fms"));
                EXPECT_BATCH_EQ(res, expected);
            }
        }
        // fnma
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.begin(), expected.begin(),
                            [](const value_type& l, const value_type& r) { return -l * r + r; });
            batch_type res = fnma(batch_lhs(), batch_rhs(), batch_rhs());
            {
                INFO(print_function_name("fnma"));
                EXPECT_BATCH_EQ(res, expected);
            }
        }
        // fnms
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.begin(), expected.begin(),
                            [](const value_type& l, const value_type& r) { return -l * r - r; });
            batch_type res = fnms(batch_lhs(), batch_rhs(), batch_rhs());
            {
                INFO(print_function_name("fnms"));
                EXPECT_BATCH_EQ(res, expected);
            }
        }
    }

    void test_abs() const
    {
        // abs
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(),
                            [](const value_type& l) { return ::detail::uabs(l); });
            batch_type res = abs(batch_lhs());
            {
                INFO(print_function_name("abs"));
                EXPECT_BATCH_EQ(res, expected);
            }
        }
        // fabs
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(),
                            [](const value_type& l) { return std::fabs(l); });
            batch_type res = fabs(batch_lhs());
            {
                INFO(print_function_name("fabs"));
                EXPECT_BATCH_EQ(res, expected);
            }
        }
    }

    void test_horizontal_operations() const
    {
        // hadd
        {
            value_type expected = std::accumulate(lhs.cbegin(), lhs.cend(), value_type(0));
            value_type res = hadd(batch_lhs());
            INFO(print_function_name("hadd"));
            EXPECT_SCALAR_EQ(res, expected);
        }
    }

    void test_boolean_conversions() const
    {
        using batch_bool_type = typename batch_type::batch_bool_type;
        // batch = true
        {
            batch_bool_type tbt(true);
            batch_type expected = batch_type(value_type(1));
            batch_type res = tbt;
            {
                INFO(print_function_name("batch = true"));
                EXPECT_BATCH_EQ(res, expected);
            }
        }
        // batch = false
        {
            batch_bool_type fbt(false);
            batch_type expected = batch_type(value_type(0));
            batch_type res = fbt;
            {
                INFO(print_function_name("batch = false"));
                EXPECT_BATCH_EQ(res, expected);
            }
        }
        // !batch
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(),
                            [](const value_type& l) { return !l; });
            batch_type res = !batch_lhs();
            {
                INFO(print_function_name("!batch"));
                EXPECT_BATCH_EQ(res, expected);
            }
        }
        // bitwise_cast
        {
            batch_bool_type fbt(false);
            batch_type expected = batch_type(value_type(0));
            batch_type res = bitwise_cast(fbt);
            {
                INFO(print_function_name("bitwise_cast"));
                EXPECT_BATCH_EQ(res, expected);
            }
        }
        // bitwise not
        {
            batch_bool_type fbt(true);
            batch_type expected = batch_type(value_type(0));
            batch_type res = ~bitwise_cast(fbt);
            {
                INFO(print_function_name("~batch"));
                EXPECT_BATCH_EQ(res, expected);
            }
        }
    }

    void test_iterator() const
    {
        array_type expected = lhs;
        batch_type v = batch_lhs();
        array_type res;
        // iterator
        {
            std::copy(v.begin(), v.end(), res.begin());
            {
                INFO( print_function_name("iterator"));
                EXPECT_EQ(res, expected);
            }
        }
        // constant iterator
        {
            std::copy(v.cbegin(), v.cend(), res.begin());
            {
                INFO( print_function_name("const iterator"));
                EXPECT_EQ(res, expected);
            }
        }
        // reverse iterator
        {
            std::copy(v.rbegin(), v.rend(), res.rbegin());
            {
                INFO( print_function_name("reverse iterator"));
                EXPECT_EQ(res, expected);
            }
        }
        // constant reverse iterator
        {
            std::copy(v.crbegin(), v.crend(), res.rbegin());
            {
                INFO( print_function_name("const reverse iterator"));
                EXPECT_EQ(res, expected);
            }
        }
    }

private:

    batch_type batch_lhs() const
    {
        return batch_type(lhs.data());
    }

    batch_type batch_rhs() const
    {
        return batch_type(rhs.data());
    }

    template <class T = value_type>
    xsimd::enable_integral_t<T, void> init_operands()
    {
        for (size_t i = 0; i < size; ++i)
        {
            bool negative_lhs = std::is_signed<T>::value && (i % 2 == 1);
            lhs[i] = value_type(i) * (negative_lhs ? -10 : 10);
            if (lhs[i] == value_type(0))
            {
                lhs[i] += value_type(1);
            }
            rhs[i] = value_type(i) + value_type(4);
        }
        scalar = value_type(3);
    }

    template <class T = value_type>
    xsimd::enable_floating_point_t<T, void> init_operands()
    {
        for (size_t i = 0; i < size; ++i)
        {
            lhs[i] = value_type(i) / 4 + value_type(1.2) * std::sqrt(value_type(i + 0.25));
            if (lhs[i] == value_type(0))
            {
                lhs[i] += value_type(0.1);
            }
            rhs[i] = value_type(10.2) / (i + 2) + value_type(0.25);
        }
        scalar = value_type(1.2);
    }
};

TEST_SUITE("batch_test")
{

    TEST_CASE_TEMPLATE_DEFINE("load_store", TypeParam, batch_test_load_store)
    {   
        batch_test<TypeParam> tester;
        tester.test_load_store();
    }
    TEST_CASE_TEMPLATE_APPLY(batch_test_load_store, batch_types);

    TEST_CASE_TEMPLATE_DEFINE("constructors", TypeParam, batch_test_constructors)
    {   
        batch_test<TypeParam> tester;
        tester.test_constructors();
    }
    TEST_CASE_TEMPLATE_APPLY(batch_test_constructors, batch_types);

    TEST_CASE_TEMPLATE_DEFINE("static_builders", TypeParam, batch_test_static_builders)
    {   
        batch_test<TypeParam> tester;
        tester.test_static_builders();
    }
    TEST_CASE_TEMPLATE_APPLY(batch_test_static_builders, batch_types);

    TEST_CASE_TEMPLATE_DEFINE("access_operator", TypeParam, batch_test_access_operator)
    {   
        batch_test<TypeParam> tester;
        tester.test_access_operator();
    }
    TEST_CASE_TEMPLATE_APPLY(batch_test_access_operator, batch_types);

    TEST_CASE_TEMPLATE_DEFINE("arithmetic", TypeParam, batch_test_arithmetic)
    {   
        batch_test<TypeParam> tester;
        tester.test_arithmetic();
    }
    TEST_CASE_TEMPLATE_APPLY(batch_test_arithmetic, batch_types);

    TEST_CASE_TEMPLATE_DEFINE("saturated_arithmetic", TypeParam, batch_test_saturated_arithmetic)
    {   
        batch_test<TypeParam> tester;
        tester.test_saturated_arithmetic();
    }
    TEST_CASE_TEMPLATE_APPLY(batch_test_saturated_arithmetic, batch_types);

    TEST_CASE_TEMPLATE_DEFINE("computed_assignment", TypeParam, batch_test_computed_assignment)
    {   
        batch_test<TypeParam> tester;
        tester.test_computed_assignment();
    }
    TEST_CASE_TEMPLATE_APPLY(batch_test_computed_assignment, batch_types);

    TEST_CASE_TEMPLATE_DEFINE("comparison", TypeParam, batch_test_comparison)
    {   
        batch_test<TypeParam> tester;
        tester.test_comparison();
    }
    TEST_CASE_TEMPLATE_APPLY(batch_test_comparison, batch_types);

    TEST_CASE_TEMPLATE_DEFINE("min_max", TypeParam, batch_test_min_max)
    {   
        batch_test<TypeParam> tester;
        tester.test_min_max();
    }
    TEST_CASE_TEMPLATE_APPLY(batch_test_min_max, batch_types);

    TEST_CASE_TEMPLATE_DEFINE("fused_operations", TypeParam, batch_test_fused_operations)
    {   
        batch_test<TypeParam> tester;
        tester.test_fused_operations();
    }
    TEST_CASE_TEMPLATE_APPLY(batch_test_fused_operations, batch_types);

    TEST_CASE_TEMPLATE_DEFINE("abs", TypeParam, batch_test_abs)
    {   
        batch_test<TypeParam> tester;
        tester.test_abs();
    }
    TEST_CASE_TEMPLATE_APPLY(batch_test_abs, batch_types);

    TEST_CASE_TEMPLATE_DEFINE("horizontal_operations", TypeParam, batch_test_horizontal_operations)
    {   
        batch_test<TypeParam> tester;
        tester.test_horizontal_operations();
    }
    TEST_CASE_TEMPLATE_APPLY(batch_test_horizontal_operations, batch_types);

    TEST_CASE_TEMPLATE_DEFINE("boolean_conversions", TypeParam, batch_test_boolean_conversions)
    {   
        batch_test<TypeParam> tester;
        tester.test_boolean_conversions();
    }
    TEST_CASE_TEMPLATE_APPLY(batch_test_boolean_conversions, batch_types);

    TEST_CASE_TEMPLATE_DEFINE("iterator", TypeParam, batch_test_iterator)
    {   
        batch_test<TypeParam> tester;
        tester.test_iterator();
    }
    TEST_CASE_TEMPLATE_APPLY(batch_test_iterator, batch_types);
}