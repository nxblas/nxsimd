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
class select_test : public testing::Test
{
  public:
    using batch_type = B;
    using value_type = typename B::value_type;
    static constexpr size_t size = B::size;
    using vector_type = std::vector<value_type>;

    size_t nb_input;
    vector_type lhs_input;
    vector_type rhs_input;
    vector_type expected;
    vector_type res;

    select_test()
    {
        nb_input = size * 10000;
        lhs_input.resize(nb_input);
        rhs_input.resize(nb_input);
        for (size_t i = 0; i < nb_input; ++i)
        {
            lhs_input[i] = value_type(i) / 4 +
                           value_type(1.2) * std::sqrt(value_type(i + 0.25));
            rhs_input[i] = value_type(10.2) / (i + 2) + value_type(0.25);
        }
        expected.resize(nb_input);
        res.resize(nb_input);
    }

    void test_select_dynamic()
    {
        for (size_t i = 0; i < nb_input; ++i)
        {
            expected[i] =
                lhs_input[i] > value_type(3) ? lhs_input[i] : rhs_input[i];
        }

        batch_type lhs_in, rhs_in, out;
        for (size_t i = 0; i < nb_input; i += size)
        {
            detail::load_batch(lhs_in, lhs_input, i);
            detail::load_batch(rhs_in, rhs_input, i);
            out = xsimd::select(lhs_in > value_type(3), lhs_in, rhs_in);
            detail::store_batch(out, res, i);
        }
        size_t diff = detail::get_nb_diff(res, expected);
        {
            INFO(print_function_name("pow"));
            EXPECT_EQ(diff, 0);
        }
    }
    struct pattern
    {
        static constexpr bool get(std::size_t i, std::size_t) { return i % 2; }
    };

    void test_select_static()
    {
        constexpr auto mask =
            xsimd::make_batch_bool_constant<value_type, pattern, size>();

        for (size_t i = 0; i < nb_input; ++i)
        {
            expected[i] = mask[i % size] ? lhs_input[i] : rhs_input[i];
        }

        batch_type lhs_in, rhs_in, out;
        for (size_t i = 0; i < nb_input; i += size)
        {
            detail::load_batch(lhs_in, lhs_input, i);
            detail::load_batch(rhs_in, rhs_input, i);
            out = xsimd::select(mask, lhs_in, rhs_in);
            detail::store_batch(out, res, i);
        }
        size_t diff = detail::get_nb_diff(res, expected);
        {
            INFO(print_function_name("pow"));
            EXPECT_EQ(diff, 0);
        }
    }
};


TEST_CASE_TEMPLATE_DEFINE("select_dynamic", TypeParam, select_test_select_dynamic) 
{
    select_test<TypeParam> tester;
    tester.test_select_dynamic();
}
TEST_CASE_TEMPLATE_DEFINE("select_static", TypeParam, select_test_select_static) 
{
    select_test<TypeParam> tester;
    tester.test_select_static();
}
TEST_CASE_TEMPLATE_APPLY(select_test_select_dynamic, batch_types);
TEST_CASE_TEMPLATE_APPLY(select_test_select_static, batch_types);
