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
class hyperbolic_test : public testing::Test
{
public:

    using batch_type = B;
    using value_type = typename B::value_type;
    static constexpr size_t size = B::size;
    using vector_type = std::vector<value_type>;

    size_t nb_input;
    vector_type input;
    vector_type acosh_input;
    vector_type atanh_input;
    vector_type expected;
    vector_type res;

    hyperbolic_test()
    {
        nb_input = size * 10000;
        input.resize(nb_input);
        acosh_input.resize(nb_input);
        atanh_input.resize(nb_input);
        for (size_t i = 0; i < nb_input; ++i)
        {
            input[i] = value_type(-1.5) + i * value_type(3) / nb_input;
            acosh_input[i] = value_type(1.) + i * value_type(3) / nb_input;
            atanh_input[i] = value_type(-0.95) + i * value_type(1.9) / nb_input;
        }
        expected.resize(nb_input);
        res.resize(nb_input);
    }

    void test_hyperbolic_functions()
    {
        // sinh
        {
            std::transform(input.cbegin(), input.cend(), expected.begin(),
                        [](const value_type& v) { return std::sinh(v); });
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
        // cosh
        {
            std::transform(input.cbegin(), input.cend(), expected.begin(),
                        [](const value_type& v) { return std::cosh(v); });
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
        // tanh
        {
            std::transform(input.cbegin(), input.cend(), expected.begin(),
                        [](const value_type& v) { return std::tanh(v); });
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
    }

    void test_reciprocal_functions()
    {
        // asinh
        {
            std::transform(input.cbegin(), input.cend(), expected.begin(),
                        [](const value_type& v) { return std::asinh(v); });
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
        // acosh
        {
            std::transform(acosh_input.cbegin(), acosh_input.cend(), expected.begin(),
                        [](const value_type& v) { return std::acosh(v); });
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
        // atanh
        {
            std::transform(atanh_input.cbegin(), atanh_input.cend(), expected.begin(),
                        [](const value_type& v) { return std::atanh(v); });
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
    }
};


TEST_CASE_TEMPLATE_DEFINE("hyperbolic", TypeParam, hyperbolic_test_hyperbolic)
{
    hyperbolic_test<TypeParam> tester;
    tester.test_hyperbolic_functions();
}

TEST_CASE_TEMPLATE_DEFINE("reciprocal", TypeParam, hyperbolic_test_reciprocal)
{
    hyperbolic_test<TypeParam> tester;
    tester.test_reciprocal_functions();
}
TEST_CASE_TEMPLATE_APPLY(hyperbolic_test_hyperbolic, batch_float_types);
TEST_CASE_TEMPLATE_APPLY(hyperbolic_test_reciprocal, batch_float_types);
