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
class batch_complex_test : public testing::Test
{
public:

    using batch_type = B;
    using real_batch_type = typename B::real_batch;
    using value_type = typename B::value_type;
    using real_value_type = typename value_type::value_type;
    static constexpr size_t size = B::size;
    using array_type = std::array<value_type, size>;
    using bool_array_type = std::array<bool, size>;
    using real_array_type = std::array<real_value_type, size>;

    array_type lhs;
    array_type rhs;
    value_type scalar;
    real_value_type real_scalar;

    batch_complex_test()
    {
        scalar = value_type(real_value_type(1.4), real_value_type(2.3));
        real_scalar = scalar.real();
        for (size_t i = 0; i < size; ++i)
        {
            lhs[i] = value_type(real_value_type(i) / real_value_type(4) + real_value_type(1.2) * std::sqrt(real_value_type(i + 0.25)),
                                real_value_type(i) / real_value_type(5));
            rhs[i] = value_type(real_value_type(10.2) / real_value_type(i + 2) + real_value_type(0.25), real_value_type(i) / real_value_type(3.2));
        }
    }

    void test_load_store() const
    {
        {
            array_type res;
            batch_type b;
            b.load_unaligned(lhs.data());
            b.store_unaligned(res.data());
            {
                INFO(print_function_name("load_unaligned / store_unaligned complex*"));
                EXPECT_EQ(res, lhs);
            }

            alignas(xsimd::arch::default_::alignment) array_type arhs(this->rhs);
            alignas(xsimd::arch::default_::alignment) array_type ares;
            b.load_aligned(arhs.data());
            b.store_aligned(ares.data());
            {
                INFO(print_function_name("load_aligned / store_aligned complex*"));
                EXPECT_EQ(ares, rhs);
            }
        }
        {
            real_array_type real, imag, res_real, res_imag;
            for (size_t i = 0; i < size; ++i)
            {
                real[i] = lhs[i].real();
                imag[i] = lhs[i].imag();
            }
            batch_type b;
            b.load_unaligned(real.data(), imag.data());
            b.store_unaligned(res_real.data(), res_imag.data());
            {
                INFO(print_function_name("load_unaligned / store_unaligned (real*, real*)"));
                EXPECT_EQ(res_real, real);
            }

            alignas(xsimd::arch::default_::alignment) real_array_type areal, aimag, ares_real, ares_imag;
            for (size_t i = 0; i < size; ++i)
            {
                areal[i] = lhs[i].real();
                aimag[i] = lhs[i].imag();
            }
            b.load_aligned(areal.data(), aimag.data());
            b.store_aligned(ares_real.data(), ares_imag.data());
            {
                INFO(print_function_name("load_aligned / store_aligned (real*, real*)"));
                EXPECT_EQ(ares_real, areal);
            }
        }
        {
            real_array_type real, res_real;
            for (size_t i = 0; i < size; ++i)
            {
                real[i] = lhs[i].real();
            }
            batch_type b;
            b.load_unaligned(real.data());
            b.store_unaligned(res_real.data());
            {
                INFO(print_function_name("load_unaligned / store_unaligned (real*)"));
                EXPECT_EQ(res_real, real);
            }

            alignas(xsimd::arch::default_::alignment) real_array_type areal, ares_real;
            for (size_t i = 0; i < size; ++i)
            {
                areal[i] = lhs[i].real();
            }
            b.load_aligned(areal.data());
            b.store_aligned(ares_real.data());
            {
                INFO(print_function_name("load_aligned / store_aligned (real*)"));
                EXPECT_EQ(ares_real, areal);
            }
        }
    }

    void test_constructors() const
    {
        array_type tmp;
        std::fill(tmp.begin(), tmp.end(), value_type(2, 3));
        batch_type b0(value_type(2, 3));
        {
            INFO(print_function_name("batch(value_type)"));
            EXPECT_EQ(b0, tmp);
        }

        std::fill(tmp.begin(), tmp.end(), value_type(real_scalar));
        batch_type b1(real_scalar);
        {
            INFO(print_function_name("batch(real_value_type)"));
            EXPECT_EQ(b1, tmp);
        }

        real_array_type real, imag;
        for (size_t i = 0; i < size; ++i)
        {
            real[i] = lhs[i].real();
            imag[i] = lhs[i].imag();
            tmp[i] = value_type(real[i]);
        }

        batch_type b2(real.data());
        {
            INFO(print_function_name("batch(real_batch)"));
            EXPECT_EQ(b2, tmp);
        }

        batch_type b3(real.data(), imag.data());
        {
            INFO(print_function_name("batch(real_batch, real_batch)"));
            EXPECT_EQ(b3, lhs);
        }

        batch_type b4(real_batch_type(real.data()));
        {
            INFO(print_function_name("batch(real_ptr)"));
            EXPECT_EQ(b4, tmp);
        }

        batch_type b5(real_batch_type(real.data()), real_batch_type(imag.data()));
        {
            INFO(print_function_name("batch(real_ptr, real_ptr)"));
            EXPECT_EQ(b5, lhs);
        }
    }

    void test_access_operator() const
    {
        batch_type res = batch_lhs();
        for (size_t i = 0; i < size; ++i)
        {
            {
                INFO(print_function_name("operator[](") << i << ")");
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
        // batch + real_batch
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), expected.begin(),
                            [](const value_type& l, const value_type& r) { return l + r.real(); });
            batch_type lres = batch_lhs() + batch_rhs().real();
            {
                INFO(print_function_name("batch + real_batch"));
                EXPECT_BATCH_EQ(lres, expected);
            }
            batch_type rres = batch_rhs().real() + batch_lhs();
            {
                INFO(print_function_name("real_batch + batch"));
                EXPECT_BATCH_EQ(rres, expected);
            }
        }
        // batch + real_scalar
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(), std::bind(std::plus<value_type>(), _1, real_scalar));
            batch_type lres = batch_lhs() + real_scalar;
            {
                INFO(print_function_name("batch + real_scalar"));
                EXPECT_BATCH_EQ(lres, expected);
            }
            batch_type rres = real_scalar + batch_lhs();
            {
                INFO(print_function_name("real_scalar + batch"));
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
        // batch - real_batch
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), expected.begin(),
                            [](const value_type& l, const value_type& r) { return l - r.real(); });
            batch_type lres = batch_lhs() - batch_rhs().real();
            {
                INFO(print_function_name("batch - real_batch"));
                EXPECT_BATCH_EQ(lres, expected);
            }
            std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), expected.begin(),
                            [](const value_type& l, const value_type& r) { return r.real() - l; });
            batch_type rres = batch_rhs().real() - batch_lhs();
            {
                INFO(print_function_name("real_batch - batch"));
                EXPECT_BATCH_EQ(rres, expected);
            }
        }
        // batch - real_scalar
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(), std::bind(std::minus<value_type>(), _1, real_scalar));
            batch_type lres = batch_lhs() - real_scalar;
            {
                INFO(print_function_name("batch - real_scalar"));
                EXPECT_BATCH_EQ(lres, expected);
            }
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(), std::bind(std::minus<value_type>(), real_scalar, _1));
            batch_type rres = real_scalar - batch_lhs();
            {
                INFO(print_function_name("real_scalar - batch"));
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
        // batch * real_batch
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), expected.begin(),
                            [](const value_type& l, const value_type& r) { return l * r.real(); });
            batch_type lres = batch_lhs() * batch_rhs().real();
            {
                INFO(print_function_name("batch * real_batch"));
                EXPECT_BATCH_EQ(lres, expected);
            }
            batch_type rres = batch_rhs().real() * batch_lhs();
            {
                INFO(print_function_name("real_batch * batch"));
                EXPECT_BATCH_EQ(rres, expected);
            }
        }
        // batch * real_scalar
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(), std::bind(std::multiplies<value_type>(), _1, real_scalar));
            batch_type lres = batch_lhs() * real_scalar;
            {
                INFO(print_function_name("batch * real_scalar"));
                EXPECT_BATCH_EQ(lres, expected);
            }
            batch_type rres = real_scalar * batch_lhs();
            {
                INFO(print_function_name("real_scalar * batch"));
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
        // batch / real_batch
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), expected.begin(),
                            [](const value_type& l, const value_type& r) { return l / r.real(); });
            batch_type lres = batch_lhs() / batch_rhs().real();
            {
                INFO(print_function_name("batch / real_batch"));
                EXPECT_BATCH_EQ(lres, expected);
            }
            std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), expected.begin(),
                            [](const value_type& l, const value_type& r) { return r.real() / l; });
            batch_type rres = batch_rhs().real() / batch_lhs();
            {
                INFO(print_function_name("real_batch / batch"));
                EXPECT_BATCH_EQ(rres, expected);
            }
        }
        // batch - real_scalar
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(), std::bind(std::divides<value_type>(), _1, real_scalar));
            batch_type lres = batch_lhs() / real_scalar;
            {
                INFO(print_function_name("batch / real_scalar"));
                EXPECT_BATCH_EQ(lres, expected);
            }
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(), std::bind(std::divides<value_type>(), real_scalar, _1));
            batch_type rres = real_scalar / batch_lhs();
            {
                INFO(print_function_name("real_scalar / batch"));
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
        // batch += real_batch
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), expected.begin(),
                            [](const value_type& l, const value_type& r) { return l + r.real(); });
            batch_type res = batch_lhs();
            res += batch_rhs().real();
            {
                INFO(print_function_name("batch += real_batch"));
                EXPECT_BATCH_EQ(res, expected);
            }
        }
        // batch += real_scalar
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(), std::bind(std::plus<value_type>(), _1, real_scalar));
            batch_type res = batch_lhs(); 
            res += real_scalar;
            {
                INFO(print_function_name("batch += real_scalar"));
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
        // batch -= real_batch
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), expected.begin(),
                            [](const value_type& l, const value_type& r) { return l - r.real(); });
            batch_type res = batch_lhs();
            res -= batch_rhs().real();
            {
                INFO(print_function_name("batch -= real_batch"));
                EXPECT_BATCH_EQ(res, expected);
            }
        }
        // batch -= real_scalar
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(), std::bind(std::minus<value_type>(), _1, real_scalar));
            batch_type res = batch_lhs(); 
            res -= real_scalar;
            {
                INFO(print_function_name("batch -= real_scalar"));
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
        // batch *= real_batch
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), expected.begin(),
                            [](const value_type& l, const value_type& r) { return l * r.real(); });
            batch_type res = batch_lhs();
            res *= batch_rhs().real();
            {
                INFO(print_function_name("batch *= real_batch"));
                EXPECT_BATCH_EQ(res, expected);
            }
        }
        // batch *= real_scalar
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(), std::bind(std::multiplies<value_type>(), _1, real_scalar));
            batch_type res = batch_lhs(); 
            res *= real_scalar;
            {
                INFO(print_function_name("batch *= real_scalar"));
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
        // batch /= real_batch
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), expected.begin(),
                            [](const value_type& l, const value_type& r) { return l / r.real(); });
            batch_type res = batch_lhs();
            res /= batch_rhs().real();
            {
                INFO(print_function_name("batch /= real_batch"));
                EXPECT_BATCH_EQ(res, expected);
            }
        }
        // batch /= real_scalar
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(), std::bind(std::divides<value_type>(), _1, real_scalar));
            batch_type res = batch_lhs(); 
            res /= real_scalar;
            {
                INFO(print_function_name("batch /= real_scalar"));
                EXPECT_BATCH_EQ(res, expected);
            }
        }
    }

    void test_conj_norm_proj() const
    {
        // conj
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(),
                            [](const value_type& v) { using std::conj; return conj(v); });
            batch_type res = conj(batch_lhs());
            {
                INFO(print_function_name("conj"));
                EXPECT_BATCH_EQ(res, expected);
            }
        }
        // norm
        {
            real_array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(),
                            [](const value_type& v) { using std::norm; return norm(v); });
            real_batch_type res = norm(batch_lhs());
            {
                INFO(print_function_name("norm"));
                EXPECT_BATCH_EQ(res, expected);
            }
        }
        // proj
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(),
                            [](const value_type& v) { using std::proj; return proj(v); });
            batch_type res = proj(batch_lhs());
            {
                INFO(print_function_name("proj"));
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
            {
                INFO(print_function_name("hadd"));
                EXPECT_SCALAR_EQ(res, expected);
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

private:

    batch_type batch_lhs() const
    {
        batch_type res;
        res.load_unaligned(lhs.data());
        return res;
    }

    batch_type batch_rhs() const
    {
        batch_type res;
        res.load_unaligned(rhs.data());
        return res;
    }
};

TEST_SUITE("batch_complex_test")
{

    TEST_CASE_TEMPLATE_DEFINE("load_store", TypeParam,batch_complex_test_load_store)
    {
        batch_complex_test<TypeParam> tester;       
        tester.test_load_store();
    }
    TEST_CASE_TEMPLATE_APPLY(batch_complex_test_load_store, batch_complex_types);

    TEST_CASE_TEMPLATE_DEFINE("constructors", TypeParam,batch_complex_test_constructors)
    {
        batch_complex_test<TypeParam> tester;
        tester.test_constructors();
    }
    TEST_CASE_TEMPLATE_APPLY(batch_complex_test_constructors, batch_complex_types);

    TEST_CASE_TEMPLATE_DEFINE("access_operator", TypeParam,batch_complex_test_access_operator)
    {
        batch_complex_test<TypeParam> tester;       
        tester.test_access_operator();
    }
    TEST_CASE_TEMPLATE_APPLY(batch_complex_test_access_operator, batch_complex_types);

    TEST_CASE_TEMPLATE_DEFINE("arithmetic", TypeParam,batch_complex_test_arithmetic)
    {
        batch_complex_test<TypeParam> tester;       
        tester.test_arithmetic();
    }
    TEST_CASE_TEMPLATE_APPLY(batch_complex_test_arithmetic, batch_complex_types);

    TEST_CASE_TEMPLATE_DEFINE("computed_assignment", TypeParam,batch_complex_test_computed_assignment)
    {
        batch_complex_test<TypeParam> tester;       
        tester.test_computed_assignment();
    }
    TEST_CASE_TEMPLATE_APPLY(batch_complex_test_computed_assignment, batch_complex_types);

    TEST_CASE_TEMPLATE_DEFINE("conj_norm_proj", TypeParam,batch_complex_test_conj_norm_proj)
    {
        batch_complex_test<TypeParam> tester;       
        tester.test_conj_norm_proj();
    }
    TEST_CASE_TEMPLATE_APPLY(batch_complex_test_conj_norm_proj, batch_complex_types);

    TEST_CASE_TEMPLATE_DEFINE("horizontal_operations", TypeParam,batch_complex_test_horizontal_operations)
    {
        batch_complex_test<TypeParam> tester;       
        tester.test_horizontal_operations();
    }
    TEST_CASE_TEMPLATE_APPLY(batch_complex_test_horizontal_operations, batch_complex_types);

    TEST_CASE_TEMPLATE_DEFINE("fused_operations", TypeParam,batch_complex_test_fused_operations)
    {
        batch_complex_test<TypeParam> tester;       
        tester.test_fused_operations();
    }
    TEST_CASE_TEMPLATE_APPLY(batch_complex_test_fused_operations, batch_complex_types);
}