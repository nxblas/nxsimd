/***************************************************************************
 * Copyright (c) Serge Guelton * Copyright (c) QuantStack *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#include "test_utils.hpp"
#include <functional>

using namespace std::placeholders;

template <class B>
class constant_batch_test : public testing::Test
{
  public:
    using batch_type = B;
    using value_type = typename B::value_type;
    static constexpr size_t size = B::size;
    using array_type = std::array<value_type, size>;
    using bool_array_type = std::array<bool, size>;

    struct generator
    {
        static constexpr value_type get(size_t index, size_t /*size*/)
        {
            return index % 2 ? 0 : 1;
        }
    };

    void test_init_from_generator() const
    {
        array_type expected;
        size_t i = 0;
        std::generate(expected.begin(), expected.end(),
                      [&i]() { return generator::get(i++, size); });
        constexpr auto b = xsimd::make_batch_constant<generator, size>();
        {
            INFO(print_function_name("batch(value_type)"));
            EXPECT_BATCH_EQ(b(), expected);
        }
    }

    struct arange
    {
        static constexpr value_type get(size_t index, size_t /*size*/)
        {
            return index;
        }
    };

    void test_init_from_generator_arange() const
    {
        array_type expected;
        size_t i = 0;
        std::generate(expected.begin(), expected.end(),
                      [&i]() { return arange::get(i++, size); });
        constexpr auto b = xsimd::make_batch_constant<arange, size>();
        {
            INFO(print_function_name("batch(value_type)"));
            EXPECT_BATCH_EQ(b(), expected);
        }
    }

    struct constant
    {
        static constexpr value_type get(size_t /*index*/, size_t /*size*/)
        {
            return 3;
        }
    };

    void test_init_from_constant() const
    {
        array_type expected;
        std::fill(expected.begin(), expected.end(), constant::get(0, 0));
        constexpr auto b = xsimd::make_batch_constant<constant, size>();
        {
            INFO(print_function_name("batch(value_type)"));
            EXPECT_BATCH_EQ(b(), expected);
        }
    }
};


TEST_CASE_TEMPLATE_DEFINE("init_from_generator", TypeParam, constant_batch_test_init_from_generator)
{
    constant_batch_test<TypeParam> tester;
    tester.test_init_from_generator();
}

TEST_CASE_TEMPLATE_DEFINE("init_from_generator_arange", TypeParam, constant_batch_test_init_from_generator_arange)
{
    constant_batch_test<TypeParam> tester;
    tester.test_init_from_generator_arange();
}

TEST_CASE_TEMPLATE_DEFINE("init_from_constant", TypeParam, constant_batch_test_init_from_constant)
{
    constant_batch_test<TypeParam> tester;
    tester.test_init_from_constant();
}

template <class B>
class constant_bool_batch_test : public testing::Test
{
  public:
    using batch_type = B;
    using value_type = typename B::value_type;
    static constexpr size_t size = B::size;
    using array_type = std::array<value_type, size>;
    using bool_array_type = std::array<bool, size>;

    struct generator
    {
        static constexpr bool get(size_t index, size_t /*size*/)
        {
            return index % 2;
        }
    };

    void test_init_from_generator() const
    {
        bool_array_type expected;
        size_t i = 0;
        std::generate(expected.begin(), expected.end(),
                      [&i]() { return generator::get(i++, size); });
        constexpr auto b =
            xsimd::make_batch_bool_constant<value_type, generator, size>();
        {
            INFO(print_function_name("batch_bool_constant(value_type)"));
            EXPECT_BATCH_EQ(b(), expected);
        }
    }

    struct split
    {
        static constexpr bool get(size_t index, size_t size)
        {
            return index < size / 2;
        }
    };

    void test_init_from_generator_split() const
    {
        bool_array_type expected;
        size_t i = 0;
        std::generate(expected.begin(), expected.end(),
                      [&i]() { return split::get(i++, size); });
        constexpr auto b =
            xsimd::make_batch_bool_constant<value_type, split, size>();
        {
            INFO(print_function_name("batch_bool_constant(value_type)"));
            EXPECT_BATCH_EQ(b(), expected);
        }
    }
};


TEST_CASE_TEMPLATE_DEFINE("init_from_generator", TypeParam, constant_bool_batch_test_init_from_generator)
{
    constant_bool_batch_test<TypeParam> tester;
    tester.test_init_from_generator();
}

TEST_CASE_TEMPLATE_DEFINE("init_from_generator_split", TypeParam, constant_bool_batch_test_init_from_generator_split)
{
    constant_bool_batch_test<TypeParam> tester;
    tester.test_init_from_generator_split();
}
TEST_CASE_TEMPLATE_APPLY(constant_batch_test_init_from_generator, batch_int_types);
TEST_CASE_TEMPLATE_APPLY(constant_batch_test_init_from_generator_arange, batch_int_types);
TEST_CASE_TEMPLATE_APPLY(constant_batch_test_init_from_constant, batch_int_types);
TEST_CASE_TEMPLATE_APPLY(constant_bool_batch_test_init_from_generator, batch_int_types);
TEST_CASE_TEMPLATE_APPLY(constant_bool_batch_test_init_from_generator_split, batch_int_types);
