/***************************************************************************
 *                                                                          *
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/
#include "test_utils.hpp"

#include <iostream>

namespace xsimd
{
    template <typename T, std::size_t N>
    struct init_shuffle_128_base
    {
        using shuffle_vector_type = std::array<T, N>;
        shuffle_vector_type lhs_in, rhs_in, exp_lo, exp_hi;

        std::vector<shuffle_vector_type> create_vectors()
        {
            std::vector<shuffle_vector_type> vects;
            vects.reserve(4);

            /* Generate input data: lhs, rhs */
            for (size_t i = 0; i < N; ++i)
            {
                lhs_in[i] = 2*i + 1;
                rhs_in[i] = 2*i + 2;
            }
            vects.push_back(std::move(lhs_in));
            vects.push_back(std::move(rhs_in));

            /* Expected shuffle data */
            for (size_t i = 0, j= 0; i < N/2; ++i, j=j+2)
            {
                exp_lo[j] = lhs_in[i];
                exp_hi[j] = lhs_in[i + N/2];

                exp_lo[j + 1] = rhs_in[i];
                exp_hi[j + 1] = rhs_in[i + N/2];
            }
            vects.push_back(std::move(exp_lo));
            vects.push_back(std::move(exp_hi));

            return vects;
        }
    };
}

template <class B>
class shuffle_128_test : public testing::Test
{
  public:
    using batch_type = B;
    using value_type = typename B::value_type;
    static constexpr size_t size = B::size;

    shuffle_128_test()
    {
        std::cout << "shuffle-128 test" << std::endl;
    }

    void shuffle_128_low_high()
    {
        xsimd::init_shuffle_128_base<value_type, size> shuffle_base;
        auto shuffle_base_vecs = shuffle_base.create_vectors();
        auto v_lhs = shuffle_base_vecs[0];
        auto v_rhs = shuffle_base_vecs[1];
        auto v_exp_lo = shuffle_base_vecs[2];
        auto v_exp_hi = shuffle_base_vecs[3];

        B b_lhs, b_rhs, b_exp_lo, b_exp_hi, b_res_lo, b_res_hi;
        b_lhs.load_unaligned(v_lhs.data());
        b_rhs.load_unaligned(v_rhs.data());
        b_exp_lo.load_unaligned(v_exp_lo.data());
        b_exp_hi.load_unaligned(v_exp_hi.data());

        /* Only Test 128bit */
        if ((sizeof(value_type) * size) == 16)
        {
            b_res_lo = xsimd::zip_lo(b_lhs, b_rhs);
            {
                INFO(print_function_name("shuffle-128 low test"));
                EXPECT_BATCH_EQ(b_res_lo, b_exp_lo);
            }

            b_res_hi = xsimd::zip_hi(b_lhs, b_rhs);
            {
                INFO(print_function_name("shuffle-128 high test"));
                EXPECT_BATCH_EQ(b_res_hi, b_exp_hi);
            }
        }
    }
};


TEST_CASE_TEMPLATE_DEFINE("shuffle_128_low_high", TypeParam, shuffle_128_test_shuffle_128_low_high)
{
    shuffle_128_test<TypeParam> tester;
    tester.shuffle_128_low_high();
}
TEST_CASE_TEMPLATE_APPLY(shuffle_128_test_shuffle_128_low_high, batch_types);
