/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay, Wolf Vollprecht and         *
* Martin Renou                                                             *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include <random>

#include "test_utils.hpp"

template <class B>
class load_store_test : public testing::Test
{
public:

    using batch_type = B;
    using value_type = typename B::value_type;
    static constexpr size_t size = B::size;
    using array_type = std::array<value_type, size>;
    using int8_vector_type = std::vector<int8_t, XSIMD_DEFAULT_ALLOCATOR(int8_t)>;
    using uint8_vector_type = std::vector<uint8_t, XSIMD_DEFAULT_ALLOCATOR(uint8_t)>;
    using int16_vector_type = std::vector<int16_t, XSIMD_DEFAULT_ALLOCATOR(int16_t)>;
    using uint16_vector_type = std::vector<uint16_t, XSIMD_DEFAULT_ALLOCATOR(uint16_t)>;
    using int32_vector_type = std::vector<int32_t, XSIMD_DEFAULT_ALLOCATOR(int32_t)>;
    using uint32_vector_type = std::vector<uint32_t, XSIMD_DEFAULT_ALLOCATOR(uint32_t)>;
    using int64_vector_type = std::vector<int64_t, XSIMD_DEFAULT_ALLOCATOR(int64_t)>;
    using uint64_vector_type = std::vector<uint64_t, XSIMD_DEFAULT_ALLOCATOR(uint64_t)>;
#ifdef XSIMD_32_BIT_ABI
    using long_vector_type = std::vector<long, XSIMD_DEFAULT_ALLOCATOR(long)>;
    using ulong_vector_type = std::vector<unsigned long, XSIMD_DEFAULT_ALLOCATOR(unsigned long)>;
#endif
    using float_vector_type = std::vector<float, XSIMD_DEFAULT_ALLOCATOR(float)>;
    using double_vector_type = std::vector<double, XSIMD_DEFAULT_ALLOCATOR(double)>;

    int8_vector_type i8_vec;
    uint8_vector_type ui8_vec;
    int16_vector_type i16_vec;
    uint16_vector_type ui16_vec;
    int32_vector_type i32_vec;
    uint32_vector_type ui32_vec;
    int64_vector_type i64_vec;
    uint64_vector_type ui64_vec;
#ifdef XSIMD_32_BIT_ABI
    long_vector_type l_vec;
    ulong_vector_type ul_vec;
#endif
    float_vector_type f_vec;
    double_vector_type d_vec;

    array_type expected;

    load_store_test()
    {
        init_test_vector(i8_vec);
        init_test_vector(ui8_vec);
        init_test_vector(i16_vec);
        init_test_vector(ui16_vec);
        init_test_vector(i32_vec);
        init_test_vector(ui32_vec);
        init_test_vector(i64_vec);
        init_test_vector(ui64_vec);
#ifdef XSIMD_32_BIT_ABI
        init_test_vector(l_vec);
        init_test_vector(ul_vec);
#endif
        init_test_vector(f_vec);
        init_test_vector(d_vec);
    }

    void test_load()
    {
        test_load_impl(i8_vec, "load int8_t");
        test_load_impl(ui8_vec, "load uint8_t");
        test_load_impl(i16_vec, "load int16_t");
        test_load_impl(ui16_vec, "load uint16_t");
        test_load_impl(i32_vec, "load int32_t");
        test_load_impl(ui32_vec, "load uint32_t");
        test_load_impl(i64_vec, "load int64_t");
        test_load_impl(ui64_vec, "load uint64_t");
#ifdef XSIMD_32_BIT_ABI
        test_load_impl(l_vec, "load long");
        test_load_impl(ul_vec, "load unsigned long");
#endif
        test_load_impl(f_vec, "load float");
        test_load_impl(d_vec, "load double");
    }

    void test_store()
    {
        test_store_impl(i8_vec, "load int8_t");
        test_store_impl(ui8_vec, "load uint8_t");
        test_store_impl(i16_vec, "load int16_t");
        test_store_impl(ui16_vec, "load uint16_t");
        test_store_impl(i32_vec, "load int32_t");
        test_store_impl(ui32_vec, "load uint32_t");
        test_store_impl(i64_vec, "load int64_t");
        test_store_impl(ui64_vec, "load uint64_t");
#ifdef XSIMD_32_BIT_ABI
        test_store_impl(l_vec, "load long");
        test_store_impl(ul_vec, "load unsigned long");
#endif
        test_store_impl(f_vec, "load float");
        test_store_impl(d_vec, "load double");
 
    }

private:

    template <class V>
    void test_load_impl(const V& v, const std::string& name)
    {
        batch_type b;
        std::copy(v.cbegin(), v.cend(), expected.begin());

        b.load_unaligned(v.data());
        {
            INFO(print_function_name(name + " unaligned"));
            EXPECT_BATCH_EQ(b, expected);
        }
        
        b.load_aligned(v.data());
        {
            INFO(print_function_name(name + " aligned"));
            EXPECT_BATCH_EQ(b, expected);
        }
    }
    
    template <class V>
    void test_store_impl(const V& v, const std::string& name)
    {
        batch_type b;
        b.load_aligned(v.data());
        V res(size);

        b.store_unaligned(res.data());
        {
            INFO(print_function_name(name + " unaligned"));
            EXPECT_VECTOR_EQ(res, v);
        }
        
        b.store_aligned(res.data());
        {
            INFO(print_function_name(name + " aligned"));
            EXPECT_VECTOR_EQ(res, v);
        }
    }

    template <class V>
    void init_test_vector(V& vec)
    {
        vec.resize(size);

        value_type min = value_type(0);
        value_type max = value_type(100);

        std::default_random_engine generator;
        std::uniform_int_distribution<int> distribution(min, max);

        auto gen = [&distribution, &generator](){
            return static_cast<value_type>(distribution(generator));
        };

        std::generate(vec.begin(), vec.end(), gen);
    }
};


TEST_CASE_TEMPLATE_DEFINE("load", TypeParam, load_store_test_load)
{
    load_store_test<TypeParam> tester;
    tester.test_load();
}

TEST_CASE_TEMPLATE_DEFINE("store", TypeParam, load_store_test_store)
{
    load_store_test<TypeParam> tester;
    tester.test_store();
}

TEST_CASE_TEMPLATE_APPLY(load_store_test_load, batch_types);
TEST_CASE_TEMPLATE_APPLY(load_store_test_store, batch_types);
