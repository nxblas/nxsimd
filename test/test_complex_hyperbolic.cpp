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
class complex_hyperbolic_test : public testing::Test
{
public:

    using batch_type = B;
    using real_batch_type = typename B::real_batch;
    using value_type = typename B::value_type;
    using real_value_type = typename value_type::value_type;
    static constexpr size_t size = B::size;
    using vector_type = std::vector<value_type>;

    size_t nb_input;
    vector_type input;
    vector_type acosh_input;
    vector_type atanh_input;
    vector_type expected;
    vector_type res;

    complex_hyperbolic_test()
    {
        nb_input = 10000 * size;
        input.resize(nb_input);
        acosh_input.resize(nb_input);
        atanh_input.resize(nb_input);
        for (size_t i = 0; i < nb_input; ++i)
        {
            input[i] = value_type(real_value_type(-1.5) + i * real_value_type(3) / nb_input,
                                  real_value_type(-1.3) + i * real_value_type(2.5) / nb_input);
            acosh_input[i] = value_type(real_value_type(1.) + i * real_value_type(3) / nb_input,
                                        real_value_type(1.2) + i * real_value_type(2.7) / nb_input);
            atanh_input[i] = value_type(real_value_type(-0.95) + i * real_value_type(1.9) / nb_input,
                                        real_value_type(-0.94) + i * real_value_type(1.8) / nb_input);
        }
        expected.resize(nb_input);
        res.resize(nb_input);
    }

    void test_sinh()
    {
        std::transform(input.cbegin(), input.cend(), expected.begin(),
                    [](const value_type& v) { using std::sinh; return sinh(v); });
        batch_type in, out;
        for (size_t i = 0; i < nb_input; i += size)
        {
            detail::load_batch(in, input, i);
            out = sinh(in);
            detail::store_batch(out, res, i);
        }
        size_t diff = detail::get_nb_diff(res, expected);
        {
            INFO(print_function_name("sinh"));
            EXPECT_EQ(diff, 0);
        }
    }

    void test_cosh()
    {
        std::transform(input.cbegin(), input.cend(), expected.begin(),
                    [](const value_type& v) { using std::cosh; return cosh(v); });
        batch_type in, out;
        for (size_t i = 0; i < nb_input; i += size)
        {
            detail::load_batch(in, input, i);
            out = cosh(in);
            detail::store_batch(out, res, i);
        }
        size_t diff = detail::get_nb_diff(res, expected);
        {
            INFO(print_function_name("cosh"));
            EXPECT_EQ(diff, 0);
        }
    }

    void test_tanh()
    {
        std::transform(input.cbegin(), input.cend(), expected.begin(),
                    [](const value_type& v) { using std::tanh; return tanh(v); });
        batch_type in, out;
        for (size_t i = 0; i < nb_input; i += size)
        {
            detail::load_batch(in, input, i);
            out = tanh(in);
            detail::store_batch(out, res, i);
        }
        size_t diff = detail::get_nb_diff(res, expected);
        {
            INFO(print_function_name("tanh"));
            EXPECT_EQ(diff, 0);
        }
    }

    void test_asinh()
    {
        std::transform(input.cbegin(), input.cend(), expected.begin(),
                    [](const value_type& v) { using std::asinh; return asinh(v); });
        batch_type in, out;
        for (size_t i = 0; i < nb_input; i += size)
        {
            detail::load_batch(in, input, i);
            out = asinh(in);
            detail::store_batch(out, res, i);
        }
        size_t diff = detail::get_nb_diff(res, expected);
        {
            INFO(print_function_name("asinh"));
            EXPECT_EQ(diff, 0);
        }
    }

    void test_acosh()
    {
        std::transform(acosh_input.cbegin(), acosh_input.cend(), expected.begin(),
                    [](const value_type& v) { using std::acosh; return acosh(v); });
        batch_type in, out;
        for (size_t i = 0; i < nb_input; i += size)
        {
            detail::load_batch(in, acosh_input, i);
            out = acosh(in);
            detail::store_batch(out, res, i);
        }
        size_t diff = detail::get_nb_diff(res, expected);
        {
            INFO(print_function_name("acosh"));
            EXPECT_EQ(diff, 0);
        }
    }

    void test_atanh()
    {
        std::transform(atanh_input.cbegin(), atanh_input.cend(), expected.begin(),
                    [](const value_type& v) { using std::atanh; return atanh(v); });
        batch_type in, out;
        for (size_t i = 0; i < nb_input; i += size)
        {
            detail::load_batch(in, atanh_input, i);
            out = atanh(in);
            detail::store_batch(out, res, i);
        }
        size_t diff = detail::get_nb_diff(res, expected);
        {
            INFO(print_function_name("atanh"));
            EXPECT_EQ(diff, 0);
        }
    }
};


TEST_CASE_TEMPLATE_DEFINE("sinh", TypeParam, complex_hyperbolic_test_sinh)
{
    complex_hyperbolic_test<TypeParam> tester;
    tester.test_sinh();
}

TEST_CASE_TEMPLATE_DEFINE("cosh", TypeParam, complex_hyperbolic_test_cosh)
{
    complex_hyperbolic_test<TypeParam> tester;
    tester.test_cosh();
}

TEST_CASE_TEMPLATE_DEFINE("tanh", TypeParam, complex_hyperbolic_test_tanh)
{
    complex_hyperbolic_test<TypeParam> tester;
    tester.test_tanh();
}

TEST_CASE_TEMPLATE_DEFINE("asinh", TypeParam, complex_hyperbolic_test_asinh)
{
    complex_hyperbolic_test<TypeParam> tester;
    tester.test_asinh();
}

TEST_CASE_TEMPLATE_DEFINE("acosh", TypeParam, complex_hyperbolic_test_acosh)
{
    complex_hyperbolic_test<TypeParam> tester;
    tester.test_acosh();
}

TEST_CASE_TEMPLATE_DEFINE("atanh", TypeParam, complex_hyperbolic_test_atanh)
{
    complex_hyperbolic_test<TypeParam> tester;
    tester.test_atanh();
}

TEST_CASE_TEMPLATE_APPLY(complex_hyperbolic_test_sinh, batch_complex_types);
TEST_CASE_TEMPLATE_APPLY(complex_hyperbolic_test_cosh, batch_complex_types);
TEST_CASE_TEMPLATE_APPLY(complex_hyperbolic_test_tanh, batch_complex_types);
TEST_CASE_TEMPLATE_APPLY(complex_hyperbolic_test_asinh, batch_complex_types);
TEST_CASE_TEMPLATE_APPLY(complex_hyperbolic_test_acosh, batch_complex_types);
TEST_CASE_TEMPLATE_APPLY(complex_hyperbolic_test_atanh, batch_complex_types);
