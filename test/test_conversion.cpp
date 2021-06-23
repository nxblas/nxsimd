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

template <class CP>
class conversion_test : public testing::Test
{
public:

    static constexpr size_t N = CP::size;
    static constexpr size_t A = CP::alignment;

    using int32_batch = xsimd::batch<int32_t, N * 2>;
    using int64_batch = xsimd::batch<int64_t, N>;
    using float_batch = xsimd::batch<float, N * 2>;
    using double_batch = xsimd::batch<double, N>;

    using uint8_batch = xsimd::batch<uint8_t, N * 8>;
    using uint16_batch = xsimd::batch<uint16_t, N * 4>;
    using uint32_batch = xsimd::batch<uint32_t, N * 2>;
    using uint64_batch = xsimd::batch<uint64_t, N>;

    using int32_vector = std::vector<int32_t, xsimd::aligned_allocator<int32_t, A>>;
    using int64_vector = std::vector<int64_t, xsimd::aligned_allocator<int64_t, A>>;
    using float_vector = std::vector<float, xsimd::aligned_allocator<float, A>>;
    using double_vector = std::vector<double, xsimd::aligned_allocator<double, A>>;

    using uint8_vector = std::vector<uint8_t, xsimd::aligned_allocator<uint8_t, A>>;

    /*int32_batch i32pos;
    int32_batch i32neg;
    int64_batch i64pos;
    int64_batch i64neg;
    float_batch fpos;
    float_batch fneg;
    double_batch dpos;
    double_batch dneg;*/

    int32_vector fposres;
    int32_vector fnegres;
    int64_vector dposres;
    int64_vector dnegres;
    float_vector i32posres;
    float_vector i32negres;
    double_vector i64posres;
    double_vector i64negres;

    uint8_vector ui8res;

    conversion_test()
        : fposres(2 * N, 7), fnegres(2 * N, -6), dposres(N, 5), dnegres(N, -1),
          i32posres(2 * N, float(2)), i32negres(2 * N, float(-3)),
          i64posres(N, double(2)), i64negres(N, double(-3)),
          ui8res(8 * N, 4)
    {
    }

    void test_to_int32()
    {
        float_batch fpos(float(7.4)), fneg(float(-6.2));
        int32_vector fvres(int32_batch::size);
        {
            int32_batch fbres = to_int(fpos);
            fbres.store_aligned(fvres.data());
            {
                INFO(print_function_name("to_int(positive float)"));
                EXPECT_VECTOR_EQ(fvres, fposres);
            }
        }
        {
            int32_batch fbres = to_int(fneg);
            fbres.store_aligned(fvres.data());
            {
                INFO(print_function_name("to_int(negative float)"));
                EXPECT_VECTOR_EQ(fvres, fnegres);
            }
        }
    }

    void test_to_int64()
    {
        double_batch dpos(double(5.4)), dneg(double(-1.2));
        int64_vector dvres(int64_batch::size);
        {
            int64_batch dbres = to_int(dpos);
            dbres.store_aligned(dvres.data());
            {
                INFO(print_function_name("to_int(positive double)"));
                EXPECT_VECTOR_EQ(dvres, dposres);
            }
        }
        {
            int64_batch dbres = to_int(dneg);
            dbres.store_aligned(dvres.data());
            {
                INFO(print_function_name("to_int(negative double)"));
                EXPECT_VECTOR_EQ(dvres, dnegres);
            }
        }
    }

    void test_to_float()
    {
        int32_batch i32pos(2), i32neg(-3);
        float_vector i32vres(float_batch::size);
        {
            float_batch i32bres = to_float(i32pos);
            i32bres.store_aligned(i32vres.data());
            {
                INFO(print_function_name("to_float(positive int32)"));
                EXPECT_VECTOR_EQ(i32vres, i32posres);
            }
        }
        {
            float_batch i32bres = to_float(i32neg);
            i32bres.store_aligned(i32vres.data());
            {
                INFO(print_function_name("to_float(negative int32)"));
                EXPECT_VECTOR_EQ(i32vres, i32negres);
            }
        }
    }

    void test_to_double()
    {
        int64_batch i64pos(2), i64neg(-3);
        double_vector i64vres(double_batch::size);
        {
            double_batch i64bres = to_float(i64pos);
            i64bres.store_aligned(i64vres.data());
            {
                INFO(print_function_name("to_float(positive int64)"));
                EXPECT_VECTOR_EQ(i64vres, i64posres);
            }
        }
        {
            double_batch i64bres = to_float(i64neg);
            i64bres.store_aligned(i64vres.data());
            {
                INFO(print_function_name("to_float(negative int64)"));
                EXPECT_VECTOR_EQ(i64vres, i64negres);
            }
        }
    }

    void test_u8_casting()
    {
        uint8_batch ui8tmp(4);
        uint8_vector ui8vres(uint8_batch::size);
        {
            uint16_batch ui16casting = u8_to_u16(ui8tmp);
            uint8_batch ui8casting = u16_to_u8(ui16casting);
            ui8casting.store_aligned(ui8vres.data());
            {
                INFO(print_function_name("u8_to_16"));
                EXPECT_VECTOR_EQ(ui8vres, ui8res);
            }
        }
        {
            uint32_batch ui32casting = u8_to_u32(ui8tmp);
            uint8_batch ui8casting = u32_to_u8(ui32casting);
            ui8casting.store_aligned(ui8vres.data());
            {
                INFO(print_function_name("u8_to_32"));
                EXPECT_VECTOR_EQ(ui8vres, ui8res);
            }
        }
        {
            uint64_batch ui64casting = u8_to_u64(ui8tmp);
            uint8_batch ui8casting = u64_to_u8(ui64casting);
            ui8casting.store_aligned(ui8vres.data());
            {
                INFO(print_function_name("u8_to_64"));
                EXPECT_VECTOR_EQ(ui8vres, ui8res);
            }
        }
    }
};


TEST_CASE_TEMPLATE_DEFINE("to_int32", TypeParam, conversion_test_to_int32)
{
    conversion_test<TypeParam> tester;
    tester.test_to_int32();
}

TEST_CASE_TEMPLATE_DEFINE("to_int64", TypeParam, conversion_test_to_int64)
{
    conversion_test<TypeParam> tester;
    tester.test_to_int64();
}

TEST_CASE_TEMPLATE_DEFINE("to_float", TypeParam, conversion_test_to_float)
{
    conversion_test<TypeParam> tester;
    tester.test_to_float();
}

TEST_CASE_TEMPLATE_DEFINE("to_double", TypeParam, conversion_test_to_double)
{
    conversion_test<TypeParam> tester;
    tester.test_to_double();
}

TEST_CASE_TEMPLATE_DEFINE("u8_casting", TypeParam, conversion_test_u8_casting)
{
    conversion_test<TypeParam> tester;
    tester.test_u8_casting();
}
TEST_CASE_TEMPLATE_APPLY(conversion_test_to_int32, conversion_types);
TEST_CASE_TEMPLATE_APPLY(conversion_test_to_int64, conversion_types);
TEST_CASE_TEMPLATE_APPLY(conversion_test_to_float, conversion_types);
TEST_CASE_TEMPLATE_APPLY(conversion_test_to_double, conversion_types);
TEST_CASE_TEMPLATE_APPLY(conversion_test_u8_casting, conversion_types);
