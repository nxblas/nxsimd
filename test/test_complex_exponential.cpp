/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay, Wolf Vollprecht and         *
* Martin Renou                                                             *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "xsimd/math/xsimd_math_complex.hpp"
#include "test_utils.hpp"

template <class B>
class complex_exponential_test : public testing::Test
{
public:

    using batch_type = B;
    using real_batch_type = typename B::real_batch;
    using value_type = typename B::value_type;
    using real_value_type = typename value_type::value_type;
    static constexpr size_t size = B::size;
    using vector_type = std::vector<value_type>;

    size_t nb_input;
    vector_type exp_input;
    vector_type huge_exp_input;
    vector_type log_input;
    vector_type expected;
    vector_type res;

    complex_exponential_test()
    {
        nb_input = 10000 * size;
        exp_input.resize(nb_input);
        huge_exp_input.resize(nb_input);
        log_input.resize(nb_input);
        for (size_t i = 0; i < nb_input; ++i)
        {
            exp_input[i] = value_type(real_value_type(-1.5) + i * real_value_type(3) / nb_input,
                                      real_value_type(-1.3) + i * real_value_type(2) / nb_input);
            huge_exp_input[i] = value_type(real_value_type(0), real_value_type(102.12) + i * real_value_type(100.) / nb_input);
            log_input[i] = value_type(real_value_type(0.001 + i * 100 / nb_input),
                                      real_value_type(0.002 + i * 110 / nb_input));
        }
        expected.resize(nb_input);
        res.resize(nb_input);
    }

    void test_exp()
    {
        std::transform(exp_input.cbegin(), exp_input.cend(), expected.begin(),
                    [](const value_type& v) { using std::exp; return exp(v); });
        batch_type in, out;
        for (size_t i = 0; i < nb_input; i += size)
        {
            detail::load_batch(in, exp_input, i);
            out = exp(in);
            detail::store_batch(out, res, i);
        }
        size_t diff = detail::get_nb_diff(res, expected);
        {
            INFO(print_function_name("exp"));
            EXPECT_EQ(diff, 0);
        }
    }

    void test_expm1()
    {
        std::transform(exp_input.cbegin(), exp_input.cend(), expected.begin(),
                    [](const value_type& v) { using xsimd::expm1; return expm1(v); });

        batch_type in, out;
        for (size_t i = 0; i < nb_input; i += size)
        {
            detail::load_batch(in, exp_input, i);
            out = expm1(in);
            detail::store_batch(out, res, i);
        }
        size_t diff = detail::get_nb_diff(res, expected);
        {
            INFO(print_function_name("expm1"));
            EXPECT_EQ(diff, 0);
        }
    }

    void test_huge_exp()
    {
        std::transform(huge_exp_input.cbegin(), huge_exp_input.cend(), expected.begin(),
                    [](const value_type& v) { using std::exp; return exp(v); });
        batch_type in, out;
        for (size_t i = 0; i < nb_input; i += size)
        {
            detail::load_batch(in, huge_exp_input, i);
            out = exp(in);
            detail::store_batch(out, res, i);
        }
        size_t diff = detail::get_nb_diff(res, expected);
        {
            INFO(print_function_name("huge exp"));
            EXPECT_EQ(diff, 0);
        }
    }

    void test_log()
    {
        std::transform(log_input.cbegin(), log_input.cend(), expected.begin(),
                    [](const value_type& v) { using std::log; return log(v); });
        batch_type in, out;
        for (size_t i = 0; i < nb_input; i += size)
        {
            detail::load_batch(in, log_input, i);
            out = log(in);
            detail::store_batch(out, res, i);
        }
        size_t diff = detail::get_nb_diff(res, expected);
        {
            INFO(print_function_name("log"));
            EXPECT_EQ(diff, 0);
        }
    }

    void test_log2()
    {
        std::transform(log_input.cbegin(), log_input.cend(), expected.begin(),
                    [](const value_type& v) { using xsimd::log2; return log2(v); });
        batch_type in, out;
        for (size_t i = 0; i < nb_input; i += size)
        {
            detail::load_batch(in, log_input, i);
            out = log2(in);
            detail::store_batch(out, res, i);
        }
        size_t diff = detail::get_nb_diff(res, expected);
        {
            INFO(print_function_name("log2"));
            EXPECT_EQ(diff, 0);
        }
    }

    void test_log10()
    {
        std::transform(log_input.cbegin(), log_input.cend(), expected.begin(),
                    [](const value_type& v) { using std::log10; return log10(v); });
        batch_type in, out;
        for (size_t i = 0; i < nb_input; i += size)
        {
            detail::load_batch(in, log_input, i);
            out = log10(in);
            detail::store_batch(out, res, i);
        }
        size_t diff = detail::get_nb_diff(res, expected);
        {
            INFO(print_function_name("log10"));
            EXPECT_EQ(diff, 0);
        }
    }

    void test_log1p()
    {
        std::transform(log_input.cbegin(), log_input.cend(), expected.begin(),
                    [](const value_type& v) { using xsimd::log1p; return log1p(v); });
        batch_type in, out;
        for (size_t i = 0; i < nb_input; i += size)
        {
            detail::load_batch(in, log_input, i);
            out = log1p(in);
            detail::store_batch(out, res, i);
        }
        size_t diff = detail::get_nb_diff(res, expected);
        {
            INFO(print_function_name("log1p"));
            EXPECT_EQ(diff, 0);
        }
    }

    void test_sign()
    {
        std::transform(log_input.cbegin(), log_input.cend(), expected.begin(),
                    [](const value_type& v) { using xsimd::sign; return sign(v); });
        batch_type in, out;
        for (size_t i = 0; i < nb_input; i += size)
        {
            detail::load_batch(in, log_input, i);
            out = sign(in);
            detail::store_batch(out, res, i);
        }
        size_t diff = detail::get_nb_diff(res, expected);
        {
            INFO(print_function_name("sign"));
            EXPECT_EQ(diff, 0);
        }
    }
};


TEST_CASE_TEMPLATE_DEFINE("exp", TypeParam, complex_exponential_test_exp)
{
    complex_exponential_test<TypeParam> tester;
    tester.test_exp();
}

TEST_CASE_TEMPLATE_DEFINE("expm1", TypeParam, complex_exponential_test_expm1)
{
    complex_exponential_test<TypeParam> tester;
    tester.test_expm1();
}

TEST_CASE_TEMPLATE_DEFINE("huge_exp", TypeParam, complex_exponential_test_huge_exp)
{
    complex_exponential_test<TypeParam> tester;
    tester.test_huge_exp();
}

TEST_CASE_TEMPLATE_DEFINE("log", TypeParam, complex_exponential_test_log)
{
    complex_exponential_test<TypeParam> tester;
    tester.test_log();
}

TEST_CASE_TEMPLATE_DEFINE("log2", TypeParam, complex_exponential_test_log2)
{
    complex_exponential_test<TypeParam> tester;
    tester.test_log2();
}

TEST_CASE_TEMPLATE_DEFINE("log10", TypeParam, complex_exponential_test_log10)
{
    complex_exponential_test<TypeParam> tester;
    tester.test_log10();
}

TEST_CASE_TEMPLATE_DEFINE("log1p", TypeParam, complex_exponential_test_log1p)
{
    complex_exponential_test<TypeParam> tester;
    tester.test_log1p();
}

TEST_CASE_TEMPLATE_DEFINE("sign", TypeParam, complex_exponential_test_sign)
{
    complex_exponential_test<TypeParam> tester;
    tester.test_sign();
}

TEST_CASE_TEMPLATE_APPLY(complex_exponential_test_exp, batch_complex_types);
TEST_CASE_TEMPLATE_APPLY(complex_exponential_test_expm1, batch_complex_types);
TEST_CASE_TEMPLATE_APPLY(complex_exponential_test_huge_exp, batch_complex_types);
TEST_CASE_TEMPLATE_APPLY(complex_exponential_test_log, batch_complex_types);
TEST_CASE_TEMPLATE_APPLY(complex_exponential_test_log2, batch_complex_types);
TEST_CASE_TEMPLATE_APPLY(complex_exponential_test_log10, batch_complex_types);
TEST_CASE_TEMPLATE_APPLY(complex_exponential_test_log1p, batch_complex_types);
TEST_CASE_TEMPLATE_APPLY(complex_exponential_test_sign, batch_complex_types);
