/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay, Wolf Vollprecht and         *
* Martin Renou                                                             *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "test_utils.hpp"

template <class B>
class exponential_test : public testing::Test
{
public:

    using batch_type = B;
    using value_type = typename B::value_type;
    static constexpr size_t size = B::size;
    using vector_type = std::vector<value_type>;

    size_t nb_input;
    vector_type exp_input;
    vector_type log_input;
    vector_type expected;
    vector_type res;

    exponential_test()
    {
        nb_input = size * 10000;
        exp_input.resize(nb_input);
        log_input.resize(nb_input);
        for (size_t i = 0; i < nb_input; ++i)
        {
            exp_input[i] = value_type(-1.5) + i * value_type(3) / nb_input;
            log_input[i] = value_type(0.001 + i * 100 / nb_input);
        }
        expected.resize(nb_input);
        res.resize(nb_input);
    }

    void test_exponential_functions()
    {
        // exp
        {
            std::transform(exp_input.cbegin(), exp_input.cend(), expected.begin(),
                        [](const value_type& v) { return std::exp(v); });
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
        // exp2
        {
            std::transform(exp_input.cbegin(), exp_input.cend(), expected.begin(),
                        [](const value_type& v) { return std::exp2(v); });
            batch_type in, out;
            for (size_t i = 0; i < nb_input; i += size)
            {
                detail::load_batch(in, exp_input, i);
                out = exp2(in);
                detail::store_batch(out, res, i);
            }
            size_t diff = detail::get_nb_diff(res, expected);
            {
                INFO(print_function_name("exp2"));
                EXPECT_EQ(diff, 0);
            }
        }
        // expm1
        {
            std::transform(exp_input.cbegin(), exp_input.cend(), expected.begin(),
                        [](const value_type& v) { return std::expm1(v); });
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
    }

    void test_log_functions()
    {
        // log
        {
            std::transform(log_input.cbegin(), log_input.cend(), expected.begin(),
                        [](const value_type& v) { return std::log(v); });
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
        // log2
        {
            std::transform(log_input.cbegin(), log_input.cend(), expected.begin(),
                        [](const value_type& v) { return std::log2(v); });
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
        // log10
        {
            std::transform(log_input.cbegin(), log_input.cend(), expected.begin(),
                        [](const value_type& v) { return std::log10(v); });
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
        // log1p
        {
            std::transform(log_input.cbegin(), log_input.cend(), expected.begin(),
                        [](const value_type& v) { return std::log1p(v); });
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
    }
};


TEST_CASE_TEMPLATE_DEFINE("exp", TypeParam, exponential_test_exp)
{
    exponential_test<TypeParam> tester;
    tester.test_exponential_functions();
}

TEST_CASE_TEMPLATE_DEFINE("log", TypeParam, exponential_test_log)
{
    exponential_test<TypeParam> tester;
    tester.test_log_functions();
}
TEST_CASE_TEMPLATE_APPLY(exponential_test_exp, batch_float_types);
TEST_CASE_TEMPLATE_APPLY(exponential_test_log, batch_float_types);
