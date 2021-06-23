/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay, Wolf Vollprecht and         *
* Martin Renou                                                             *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include <vector>
#include <functional>

#include "test_utils.hpp"

namespace xsimd
{
    template <class T, std::size_t N>
    struct get_bool_base
    {
        using vector_type = std::array<bool, N>;

        std::vector<vector_type> almost_all_false()
        {
            std::vector<vector_type> vectors;
            vectors.reserve(N);
            for (size_t i = 0; i < N; ++i)
            {
                vector_type v;
                v.fill(false);
                v[i] = true;
                vectors.push_back(std::move(v));
            }
            return vectors;
        }

        std::vector<vector_type> almost_all_true()
        {
            auto vectors = almost_all_false();
            flip(vectors);
            return vectors;
        }

        void flip(vector_type& vec)
        {
            std::transform(vec.begin(), vec.end(), vec.begin(), std::logical_not<bool>{});
        }

        void flip(std::vector<vector_type>& vectors)
        {
            for (auto& vec : vectors)
            {
                flip(vec);
            }
        }
    };

    template <class T>
    struct get_bool;

    template <class T>
    struct get_bool<batch_bool<T, 2>> : public get_bool_base<T, 2>
    {
        using type = batch_bool<T, 2>;
        type all_true = type(true);
        type all_false = type(false);
        type half = type(0, 1);
        type ihalf = type(1, 0);
        type interspersed = type(0, 1);
    };

    template <class T>
    struct get_bool<batch_bool<T, 4>> : public get_bool_base<T, 4>
    {
        using type = batch_bool<T, 4>;

        type all_true = type(1);
        type all_false = type(0);
        type half = type(0, 0, 1, 1);
        type ihalf = type(1, 1, 0, 0);
        type interspersed = type(0, 1, 0, 1);
    };

    template <class T>
    struct get_bool<batch_bool<T, 8>> : public get_bool_base<T, 8>
    {
        using type = batch_bool<T, 8>;
        type all_true = type(true);
        type all_false = type(false);
        type half = type(0, 0, 0, 0, 1, 1, 1, 1);
        type ihalf = type(1, 1, 1, 1, 0, 0, 0, 0);
        type interspersed = type(0, 1, 0, 1, 0, 1, 0, 1);
    };

    template <class T>
    struct get_bool<batch_bool<T, 16>> : public get_bool_base<T, 16>
    {
        using type = batch_bool<T, 16>;
        type all_true = type(true);
        type all_false = type(false);
        type half = type(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1);
        type ihalf = type(1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0);
        type interspersed = type(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1);
    };

    template <class T>
    struct get_bool<batch_bool<T, 32>> : public get_bool_base<T, 32>
    {
        using type = batch_bool<T, 32>;
        type all_true = type(true);
        type all_false = type(false);
        type half = type(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
        type ihalf = type(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        type interspersed = type(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1);
    };

    template <class T>
    struct get_bool<batch_bool<T, 64>> : public get_bool_base<T, 64>
    {
        using type = batch_bool<T, 64>;
        type all_true = type(true);
        type all_false = type(false);
        type half = type (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
        type ihalf = type(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        type interspersed = type(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1);
    };

    // For fallbacks
    template <class T>
    struct get_bool<batch_bool<T, 3>> : public get_bool_base<T, 3>
    {
        using type = batch_bool<T, 3>;
        type all_true = type(true);
        type all_false = type(false);
        type half = type(false, false, true);
        type ihalf = type(true, true, false);
    };

    template <class T>
    struct get_bool<batch_bool<T, 7>> : public get_bool_base<T, 7>
    {
        using type = batch_bool<T, 7>;
        type all_true = type(true);
        type all_false = type(false);
        type half = type(false, false, false, false, true, true, true);
        type ihalf = type(true, true, true, true, false, false, false);
    };
}

template <class B>
class batch_bool_test : public testing::Test
{
public:

    using batch_type = B;
    using value_type = typename B::value_type;
    static constexpr size_t size = B::size;
    using batch_bool_type = typename B::batch_bool_type;
    using array_type = std::array<value_type, size>;
    using bool_array_type = std::array<bool, size>;

    array_type lhs;
    array_type rhs;
    bool_array_type ba;

    batch_bool_test()
    {
        for (size_t i = 0; i < size; ++i)
        {
            lhs[i] = value_type(i);
            rhs[i] = i == 0%2 ? lhs[i] : lhs[i] * 2;
            ba[i] = i == 0%2 ? true : false;
        }
    }

    void test_load_store() const
    {
        bool_array_type res;
        batch_bool_type b;
        b.load_unaligned(ba);
        b.store_unaligned(res.data());
        {
            INFO(print_function_name("load_unaligned / store_unaligned"));
            EXPECT_EQ(res, ba);
        }

        alignas(xsimd::arch::default_::alignment) bool_array_type arhs(this->ba);
        alignas(xsimd::arch::default_::alignment) bool_array_type ares;
        b.load_aligned(arhs.data());
        b.store_aligned(ares.data());
        {
            INFO(print_function_name("load_aligned / store_aligned"));
            EXPECT_EQ(ares, arhs);
        }
    }

    void test_any_all() const
    {
        auto bool_g = xsimd::get_bool<batch_bool_type>{};
        // any
        {
            auto any_check_false = (batch_lhs() != batch_lhs());
            bool any_res_false = xsimd::any(any_check_false);
            {
                INFO(print_function_name("any (false)"));
                EXPECT_FALSE(any_res_false);
            }
            auto any_check_true = (batch_lhs() == batch_rhs());
            bool any_res_true = xsimd::any(any_check_true);
            {
                INFO(print_function_name("any (true)"));
                EXPECT_TRUE(any_res_true);
            }

            for (const auto& vec : bool_g.almost_all_false())
            {
                batch_bool_type b;
                b.load_unaligned(vec.data());
                bool any_res = xsimd::any(b);
                {
                    INFO(print_function_name("any (almost_all_false)"));
                    EXPECT_TRUE(any_res);
                }
            }

            for (const auto& vec : bool_g.almost_all_true())
            {
                batch_bool_type b;
                b.load_unaligned(vec.data());
                bool any_res = xsimd::any(b);
                {
                    INFO(print_function_name("any (almost_all_true)"));
                    EXPECT_TRUE(any_res);
                }
            }
        }
        // all
        {
            auto all_check_false = (batch_lhs() == batch_rhs());
            bool all_res_false = xsimd::all(all_check_false);
            {
                INFO(print_function_name("all (false)"));
                EXPECT_FALSE(all_res_false);
            }
            auto all_check_true = (batch_lhs() == batch_lhs());
            bool all_res_true = xsimd::all(all_check_true);
            {
                INFO(print_function_name("all (true)"));
                EXPECT_TRUE(all_res_true);
            }

            for (const auto& vec : bool_g.almost_all_false())
            {
                // TODO: implement batch_bool(bool*)
                // It currently compiles (need to understand why) but does not
                // give expected result
                batch_bool_type b;
                b.load_unaligned(vec.data());
                bool all_res = xsimd::all(b);
                {
                    INFO(print_function_name("all (almost_all_false)"));
                    EXPECT_FALSE(all_res);
                }
            }

            for (const auto& vec : bool_g.almost_all_true())
            {
                batch_bool_type b;
                b.load_unaligned(vec.data());
                bool all_res = xsimd::all(b);
                {
                    INFO(print_function_name("all (almost_all_true)"));
                    EXPECT_FALSE(all_res);
                }
            }
        }
    }

    void test_logical_operations() const
    {
        auto bool_g = xsimd::get_bool<batch_bool_type>{};
        size_t s = size;
        // operator!=
        {
            bool res = xsimd::all(bool_g.half != bool_g.ihalf);
            {
                INFO(print_function_name("operator!="));
                EXPECT_TRUE(res);
            }
        }
        // operator==
        {
            bool res = xsimd::all(bool_g.half == !bool_g.ihalf);
            {
                INFO(print_function_name("operator=="));
                EXPECT_TRUE(res);
            }
        }
        // operator &&
        {
            batch_bool_type res = bool_g.half && bool_g.ihalf;
            bool_array_type ares;
            res.store_unaligned(ares.data());
            size_t nb_false = std::count(ares.cbegin(), ares.cend(), false);
            {
                INFO(print_function_name("operator&&"));
                EXPECT_EQ(nb_false, s);
            }
        }
        // operator ||
        {
            batch_bool_type res = bool_g.half || bool_g.ihalf;
            bool_array_type ares;
            res.store_unaligned(ares.data());
            size_t nb_false = std::count(ares.cbegin(), ares.cend(), true);
            {
                INFO(print_function_name("operator||"));
                EXPECT_EQ(nb_false, s);
            }
        }
    }

    void test_bitwise_operations() const
    {
        auto bool_g = xsimd::get_bool<batch_bool_type>{};
        // operator~
        {
            bool res = xsimd::all(bool_g.half == ~bool_g.ihalf);
            {
                INFO(print_function_name("operator~"));
                EXPECT_TRUE(res);
            }
        }
        // operator|
        {
            bool res = xsimd::all((bool_g.half | bool_g.ihalf) == bool_g.all_true);
            {
                INFO(print_function_name("operator|"));
                EXPECT_TRUE(res);
            }
        }
        // operator&
        {
            bool res = xsimd::all((bool_g.half & bool_g.ihalf) == bool_g.all_false);
            {
                INFO(print_function_name("operator&"));
                EXPECT_TRUE(res);
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
};

TEST_SUITE_BEGIN("batch_bool_test");

TEST_CASE_TEMPLATE_DEFINE("load_store", TypeParam, batch_bool_test_load_store)
{
    batch_bool_test<TypeParam> tester;
    tester.test_load_store();
}
TEST_CASE_TEMPLATE_APPLY(batch_bool_test_load_store, batch_types);

TEST_CASE_TEMPLATE_DEFINE("any_all", TypeParam, batch_bool_test_any_all)
{
    batch_bool_test<TypeParam> tester;
    tester.test_any_all();
}
TEST_CASE_TEMPLATE_APPLY(batch_bool_test_any_all, batch_types);

TEST_CASE_TEMPLATE_DEFINE("logical_operations", TypeParam, batch_bool_test_logical_operations)
{
    batch_bool_test<TypeParam> tester;
    tester.test_logical_operations();
}
TEST_CASE_TEMPLATE_APPLY(batch_bool_test_logical_operations, batch_types);

TEST_CASE_TEMPLATE_DEFINE("bitwise_operations", TypeParam, batch_bool_test_bitwise_operations)
{
    batch_bool_test<TypeParam> tester;
    tester.test_bitwise_operations();
}
TEST_CASE_TEMPLATE_APPLY(batch_bool_test_bitwise_operations, batch_types);

TEST_SUITE_END();