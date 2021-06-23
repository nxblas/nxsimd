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
class poly_evaluation_test : public testing::Test
{
public:

    using batch_type = B;
    using value_type = typename B::value_type;
    static constexpr size_t size = B::size;
    using vector_type = std::vector<value_type>;

    size_t nb_input;
    vector_type input;
    vector_type horner_res;
    vector_type estrin_res;

    poly_evaluation_test()
    {
        nb_input = size * 10000;
        input.resize(nb_input);
        for (size_t i = 0; i < nb_input; ++i)
        {
            input[i] = value_type(i) / 4 + value_type(1.2) * std::sqrt(value_type(i + 0.25));
        }
        horner_res.resize(nb_input);
        estrin_res.resize(nb_input);
    }

    void test_poly_evaluation()
    {
        batch_type in, out;
        for (size_t i = 0; i < nb_input; i += size)
        {
            detail::load_batch(in, input, i);
            out = xsimd::horner<batch_type, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16>(in);
            detail::store_batch(out, horner_res, i);
            out = xsimd::estrin<batch_type, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16>(in);
            detail::store_batch(out, estrin_res, i);
        }
        size_t diff = detail::get_nb_diff(horner_res, estrin_res);
        {
            INFO(print_function_name("estrin"));
            EXPECT_EQ(diff, 0);
        }
    }
};


TEST_CASE_TEMPLATE_DEFINE("poly_evaluation", TypeParam, poly_evaluation_test_poly_evaluation)
{
    poly_evaluation_test<TypeParam> tester;
    tester.test_poly_evaluation();
}
TEST_CASE_TEMPLATE_APPLY(poly_evaluation_test_poly_evaluation, batch_float_types);
