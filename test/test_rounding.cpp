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
class rounding_test : public testing::Test
{
public:

    using batch_type = B;
    using value_type = typename B::value_type;
    static constexpr size_t size = B::size;
    static constexpr size_t nb_input = 8;
    static constexpr size_t nb_batches = nb_input / size;

    std::array<value_type, nb_input> input;
    std::array<value_type, nb_input> expected;
    std::array<value_type, nb_input> res;

    rounding_test()
    {
        input[0] = value_type(-3.5);
        input[1] = value_type(-2.7);
        input[2] = value_type(-2.5);
        input[3] = value_type(-2.3);
        input[4] = value_type(2.3);
        input[5] = value_type(2.5);
        input[6] = value_type(2.7);
        input[7] = value_type(3.5);
    }

    void test_rounding_functions()
    {
        // ceil
        {
            std::transform(input.cbegin(), input.cend(), expected.begin(),
                        [](const value_type& v) { return std::ceil(v); });
            batch_type in, out;
            for (size_t i = 0; i < nb_batches; i += size)
            {
                detail::load_batch(in, input, i);
                out = ceil(in);
                detail::store_batch(out, res, i);
            }
            for (size_t i = nb_batches; i < nb_input; ++i)
            {
                res[i] = std::ceil(input[i]);
            }
            size_t diff = detail::get_nb_diff(res, expected);
            {
                INFO(print_function_name("ceil"));
                EXPECT_EQ(diff, 0);
            }
        }
        // floor
        {
            std::transform(input.cbegin(), input.cend(), expected.begin(),
                        [](const value_type& v) { return std::floor(v); });
            batch_type in, out;
            for (size_t i = 0; i < nb_batches; i += size)
            {
                detail::load_batch(in, input, i);
                out = floor(in);
                detail::store_batch(out, res, i);
            }
            for (size_t i = nb_batches; i < nb_input; ++i)
            {
                res[i] = std::floor(input[i]);
            }
            size_t diff = detail::get_nb_diff(res, expected);
            {
                INFO(print_function_name("floor"));
                EXPECT_EQ(diff, 0);
            }
        }
        // trunc
        {
            std::transform(input.cbegin(), input.cend(), expected.begin(),
                        [](const value_type& v) { return std::trunc(v); });
            batch_type in, out;
            for (size_t i = 0; i < nb_batches; i += size)
            {
                detail::load_batch(in, input, i);
                out = trunc(in);
                detail::store_batch(out, res, i);
            }
            for (size_t i = nb_batches; i < nb_input; ++i)
            {
                res[i] = std::trunc(input[i]);
            }
            size_t diff = detail::get_nb_diff(res, expected);
            {
                INFO(print_function_name("trunc"));
                EXPECT_EQ(diff, 0);
            }
        }
        // round
        {
            std::transform(input.cbegin(), input.cend(), expected.begin(),
                        [](const value_type& v) { return std::round(v); });
            batch_type in, out;
            for (size_t i = 0; i < nb_batches; i += size)
            {
                detail::load_batch(in, input, i);
                out = round(in);
                detail::store_batch(out, res, i);
            }
            for (size_t i = nb_batches; i < nb_input; ++i)
            {
                res[i] = std::round(input[i]);
            }
            size_t diff = detail::get_nb_diff(res, expected);
            {
                INFO(print_function_name("round"));
                EXPECT_EQ(diff, 0);
            }
        }
        // nearbyint
        {
            std::transform(input.cbegin(), input.cend(), expected.begin(),
                        [](const value_type& v) { return std::nearbyint(v); });
            batch_type in, out;
            for (size_t i = 0; i < nb_batches; i += size)
            {
                detail::load_batch(in, input, i);
                out = nearbyint(in);
                detail::store_batch(out, res, i);
            }
            for (size_t i = nb_batches; i < nb_input; ++i)
            {
                res[i] = std::nearbyint(input[i]);
            }
            size_t diff = detail::get_nb_diff(res, expected);
            {
                INFO(print_function_name("nearbyint"));
                EXPECT_EQ(diff, 0);
            }
        }
        // rint
        {
            std::transform(input.cbegin(), input.cend(), expected.begin(),
                        [](const value_type& v) { return std::rint(v); });
            batch_type in, out;
            for (size_t i = 0; i < nb_batches; i += size)
            {
                detail::load_batch(in, input, i);
                out = rint(in);
                detail::store_batch(out, res, i);
            }
            for (size_t i = nb_batches; i < nb_input; ++i)
            {
                res[i] = std::rint(input[i]);
            }
            size_t diff = detail::get_nb_diff(res, expected);
            {
                INFO(print_function_name("rint"));
                EXPECT_EQ(diff, 0);
            }
        }
    }
};


TEST_CASE_TEMPLATE_DEFINE("rounding", TypeParam, rounding_test_rounding)
{
    rounding_test<TypeParam> tester;
    tester.test_rounding_functions();
}

TEST_CASE_TEMPLATE_APPLY(rounding_test_rounding, batch_float_types);
