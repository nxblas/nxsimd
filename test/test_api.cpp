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
class xsimd_api_test : public testing::Test
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
    float_vector_type f_vec;
    double_vector_type d_vec;

    array_type expected;

    xsimd_api_test()
    {
        init_test_vector(i8_vec);
        init_test_vector(ui8_vec);
        init_test_vector(i16_vec);
        init_test_vector(ui16_vec);
        init_test_vector(i32_vec);
        init_test_vector(ui32_vec);
        init_test_vector(i64_vec);
        init_test_vector(ui64_vec);
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
        test_store_impl(f_vec, "load float");
        test_store_impl(d_vec, "load double");
 
    }

    void test_set()
    {
        test_set_impl<int8_t>("set int8_t");
        test_set_impl<uint8_t>("set uint8_t");
        test_set_impl<int16_t>("set int16_t");
        test_set_impl<uint16_t>("set uint16_t");
        test_set_impl<int32_t>("set int32_t");
        test_set_impl<uint32_t>("set uint32_t");
        test_set_impl<int64_t>("set int64_t");
        test_set_impl<uint64_t>("set uint64_t");
        test_set_impl<float>("set float");
        test_set_impl<double>("set double");
    }

private:

    template <class V>
    void test_load_impl(const V& v, const std::string& name)
    {
        using src_value_type = typename V::value_type;
        batch_type b;
        std::copy(v.cbegin(), v.cend(), expected.begin());

        b = xsimd::load_simd<src_value_type, value_type>(v.data(), xsimd::unaligned_mode());
        EXPECT_BATCH_EQ_MESSAGE(b, expected,  print_function_name(name + " unaligned"));
        
        b = xsimd::load_simd<src_value_type, value_type>(v.data(), xsimd::aligned_mode());
        EXPECT_BATCH_EQ_MESSAGE(b, expected, print_function_name(name + " aligned"));
    }
    
    template <class V>
    void test_store_impl(const V& v, const std::string& name)
    {
        using src_value_type = typename V::value_type;
        batch_type b = xsimd::load_simd<src_value_type, value_type>(v.data(), xsimd::aligned_mode());
        V res(size);

        xsimd::store_simd<src_value_type, value_type>(res.data(), b, xsimd::unaligned_mode());
        EXPECT_VECTOR_EQ_MESSAGE(res, v, print_function_name(name + " unaligned"));
        
        xsimd::store_simd<src_value_type, value_type>(res.data(), b, xsimd::aligned_mode());
        EXPECT_VECTOR_EQ_MESSAGE(res, v, print_function_name(name + " aligned"));
    }

    template <class T>
    void test_set_impl(const std::string& name)
    {
        T v = T(1);
        batch_type expected(v);
        batch_type res = xsimd::set_simd<T, value_type>(v);
        EXPECT_BATCH_EQ_MESSAGE(res, expected, print_function_name(name));
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

using xsimd_api_types = testing::Types<

#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX512_VERSION
                            xsimd::batch<uint8_t, 64>,
                            xsimd::batch<int8_t, 64>,
                            xsimd::batch<uint16_t, 32>,
                            xsimd::batch<int16_t, 32>,
                            xsimd::batch<uint32_t, 16>,
                            xsimd::batch<int32_t, 16>,
                            xsimd::batch<uint64_t, 8>,
                            xsimd::batch<int64_t, 8>,
                            xsimd::batch<float, 16>,
                            xsimd::batch<double, 8>
#elif XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX_VERSION
                            xsimd::batch<uint8_t, 32>,
                            xsimd::batch<int8_t, 32>,
                            xsimd::batch<uint16_t, 16>,
                            xsimd::batch<int16_t, 16>,
                            xsimd::batch<uint32_t, 8>,
                            xsimd::batch<int32_t, 8>,
                            xsimd::batch<uint64_t, 4>,
                            xsimd::batch<int64_t, 4>,
                            xsimd::batch<float, 8>,
                            xsimd::batch<double, 4>
#elif XSIMD_X86_INSTR_SET >= XSIMD_X86_SSE2_VERSION
                            xsimd::batch<uint8_t, 16>,
                            xsimd::batch<int8_t, 16>,
                            xsimd::batch<uint16_t, 8>,
                            xsimd::batch<int16_t, 8>,
                            xsimd::batch<uint32_t, 4>,
                            xsimd::batch<int32_t, 4>,
                            xsimd::batch<uint64_t, 2>,
                            xsimd::batch<int64_t, 2>,
                            xsimd::batch<float, 4>,
                            xsimd::batch<double, 2>
#elif XSIMD_ARM_INSTR_SET >= XSIMD_ARM7_NEON_VERSION
                            xsimd::batch<uint8_t, 16>,
                            xsimd::batch<int8_t, 16>,
                            xsimd::batch<uint16_t, 8>,
                            xsimd::batch<int16_t, 8>,
                            xsimd::batch<uint32_t, 4>,
                            xsimd::batch<int32_t, 4>,
                            xsimd::batch<uint64_t, 2>,
                            xsimd::batch<int64_t, 2>,
                            xsimd::batch<float, 4>
#if XSIMD_ARM_INSTR_SET >= XSIMD_ARM8_64_NEON_VERSION
                            ,
                            xsimd::batch<double, 2>
#endif
#endif
                        >;


TEST_SUITE("xsimd_api_test")
{
    TEST_CASE_TEMPLATE_DEFINE("load", TypeParam, xsimd_api_test_load)
    {
        xsimd_api_test<TypeParam> api_tester;
        api_tester.test_load();
    }
    TEST_CASE_TEMPLATE_APPLY(xsimd_api_test_load, xsimd_api_types);

    TEST_CASE_TEMPLATE_DEFINE("store", TypeParam, xsimd_api_test_store)
    {
        xsimd_api_test<TypeParam> api_tester;
        api_tester.test_store();
    }
    TEST_CASE_TEMPLATE_APPLY(xsimd_api_test_store, xsimd_api_types);

    #ifdef XSIMD_BATCH_DOUBLE_SIZE
    TEST_CASE_TEMPLATE_DEFINE("set", TypeParam, xsimd_api_test_store_set)
    {
        xsimd_api_test<TypeParam> api_tester;
        api_tester.test_set();
    }
    TEST_CASE_TEMPLATE_APPLY(xsimd_api_test_store_set, xsimd_api_types);
    #endif
}
