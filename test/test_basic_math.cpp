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

namespace detail
{
    template <class T, bool is_int = std::is_integral<typename T::value_type>::value>
    struct infinity_tester
    {
        static void test_isfinite()
        {
            T input(1.);
            INFO(print_function_name("isfinite"));
            EXPECT_TRUE(xsimd::all(xsimd::isfinite(input)));
        }

        static void test_isinf()
        {
            T input(1.);
            INFO(print_function_name("isinf"));
            EXPECT_FALSE(xsimd::any(xsimd::isinf(input)));
        }
    };

    template <class T>
    struct infinity_tester<T, false>
    {
        static void test_isfinite()
        {
            T input = xsimd::infinity<T>();
            INFO(print_function_name("isfinite"));
            EXPECT_FALSE(xsimd::any(xsimd::isfinite(input)));
        }
        static void test_isinf()
        {
            T input = xsimd::infinity<T>();
            INFO(print_function_name("isinf"));
            EXPECT_TRUE(xsimd::all(xsimd::isinf(input)));
        }
    };
}

template <class B>
class basic_math_test : public testing::Test
{
public:

    using batch_type = B;
    using value_type = typename B::value_type;
    static constexpr size_t size = B::size;
    using array_type = std::array<value_type, size>;
    
    array_type lhs;
    array_type rhs;
    array_type clip_input;
    array_type from_input;

    basic_math_test()
    {
        for (size_t i = 0; i < size; ++i)
        {
            lhs[i] = value_type(i) / 4 + value_type(1.2) * std::sqrt(value_type(i + 0.25)) + value_type(1.);
            rhs[i] = value_type(10.2) / (i + 2) + value_type(0.25) + value_type(1.);
            clip_input[i] = i * value_type(0.25);
            from_input[i] = rhs[i] - value_type(1);
        }
    }

    void test_basic_functions() const
    {
        // fmod
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), expected.begin(),
                            [](const value_type& l, const value_type& r) { return std::fmod(l, r); });
            batch_type res = xsimd::fmod(batch_lhs(), batch_rhs());
            EXPECT_BATCH_EQ_MESSAGE(res, expected, print_function_name("fmod"));
        }
        // remainder
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), expected.begin(),
                            [](const value_type& l, const value_type& r) { return std::remainder(l, r); });
            batch_type res = xsimd::remainder(batch_lhs(), batch_rhs());
            EXPECT_BATCH_EQ_MESSAGE(res, expected, print_function_name("remainder"));
        }
        // fdim
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), expected.begin(),
                            [](const value_type& l, const value_type& r) { return std::fdim(l, r); });
            batch_type res = xsimd::fdim(batch_lhs(), batch_rhs());
            EXPECT_BATCH_EQ_MESSAGE(res, expected, print_function_name("fdim"));
        }
        // clip
        {
            value_type clip_lo = static_cast<value_type>(0.5);
            value_type clip_hi = static_cast<value_type>(1.);
            array_type expected;
            std::transform(clip_input.cbegin(), clip_input.cend(), expected.begin(),
                           [clip_lo, clip_hi](const value_type& l) {
                               return l < clip_lo ? clip_lo : clip_hi < l ? clip_hi : l;
                           });
            batch_type res = xsimd::clip(batch_clip_input(), batch_type(clip_lo), batch_type(clip_hi));
            EXPECT_BATCH_EQ_MESSAGE(res, expected, print_function_name("clip"));
        }
        // isfinite
        {
            detail::infinity_tester<batch_type>::test_isfinite();
        }
        // isinf
        {
            detail::infinity_tester<batch_type>::test_isinf();
        }
        // nextafter
        {
            array_type expected;
            std::transform(from_input.cbegin(), from_input.cend(), rhs.cbegin(), expected.begin(),
                            [](const value_type& l, const value_type& r) { return std::nextafter(l, r); });
            batch_type res = xsimd::nextafter(batch_from_input(), batch_rhs());
            EXPECT_BATCH_EQ_MESSAGE(res, expected, print_function_name("nextafter"));
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

    batch_type batch_clip_input() const
    {
        return batch_type(clip_input.data());
    }

    batch_type batch_from_input() const
    {
        return batch_type(from_input.data());
    }
};


TEST_SUITE("basic_math_test")
{

    TEST_CASE_TEMPLATE_DEFINE("basic_functions", TypeParam, basic_math_test_basic_functions)
    {
        basic_math_test<TypeParam> tester;
        tester.test_basic_functions();
    }
    TEST_CASE_TEMPLATE_APPLY(basic_math_test_basic_functions, batch_math_types);
}

