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
class bitwise_cast_test : public testing::Test
{
public:

    static constexpr size_t N = CP::size;
    static constexpr size_t A = CP::alignment;

    using int32_batch = xsimd::batch<int32_t, N * 2>;
    using int64_batch = xsimd::batch<int64_t, N>;
    using float_batch = xsimd::batch<float, N * 2>;
    using double_batch = xsimd::batch<double, N>;

    using int32_vector = std::vector<int32_t, xsimd::aligned_allocator<int32_t, A>>;
    using int64_vector = std::vector<int64_t, xsimd::aligned_allocator<int64_t, A>>;
    using float_vector = std::vector<float, xsimd::aligned_allocator<float, A>>;
    using double_vector = std::vector<double, xsimd::aligned_allocator<double, A>>;

    int32_vector ftoi32_res;
    int32_vector dtoi32_res;
    int64_vector ftoi64_res;
    int64_vector dtoi64_res;
    float_vector i32tof_res;
    float_vector i64tof_res;
    float_vector dtof_res;
    double_vector i32tod_res;
    double_vector i64tod_res;
    double_vector ftod_res;

    bitwise_cast_test()
        : ftoi32_res(2 * N), dtoi32_res(2 * N), ftoi64_res(N), dtoi64_res(N),
          i32tof_res(2 * N), i64tof_res(2 * N), dtof_res(2 * N),
          i32tod_res(N), i64tod_res(N), ftod_res(N)
    {
        {
            int32_batch input = i32_input();
            bitcast b;
            b.i32[0] = input[0];
            b.i32[1] = input[1];
            std::fill(i32tof_res.begin(), i32tof_res.end(), b.f[0]);
            std::fill(i32tod_res.begin(), i32tod_res.end(), b.d);
        }
        {
            int64_batch input = i64_input();
            bitcast b;
            b.i64 = input[0];
            std::fill(i64tod_res.begin(), i64tod_res.end(), b.d);
            for (size_t i = 0; i < N; ++i)
            {
                i64tof_res[2 * i] = b.f[0];
                i64tof_res[2 * i + 1] = b.f[1];
            }
        }
        {
            float_batch input = f_input();
            bitcast b;
            b.f[0] = input[0];
            b.f[1] = input[1];
            std::fill(ftoi32_res.begin(), ftoi32_res.end(), b.i32[0]);
            std::fill(ftoi64_res.begin(), ftoi64_res.end(), b.i64);
            std::fill(ftod_res.begin(), ftod_res.end(), b.d);
        }
        {
            double_batch input = d_input();
            bitcast b;
            b.d = input[0];
            //std::fill(dtoi32_res.begin(), dtoi32_res.end(), b.i32[0]);
            std::fill(dtoi64_res.begin(), dtoi64_res.end(), b.i64);
            for (size_t i = 0; i < N; ++i)
            {
                dtoi32_res[2 * i] = b.i32[0];
                dtoi32_res[2 * i + 1] = b.i32[1];
                dtof_res[2 * i] = b.f[0];
                dtof_res[2 * i + 1] = b.f[1];
            }
        }
    }

    void test_to_int32()
    {
        int32_vector i32vres(int32_batch::size);
        {
            int32_batch i32bres = xsimd::bitwise_cast<int32_batch>(f_input());
            i32bres.store_aligned(i32vres.data());
            {
                INFO(print_function_name("to_int32(float)"));
                EXPECT_VECTOR_EQ(i32vres, ftoi32_res);
            }
        }
        {
            int32_batch i32bres = xsimd::bitwise_cast<int32_batch>(d_input());
            i32bres.store_aligned(i32vres.data());
            {
                INFO(print_function_name("to_int32(double)"));
                EXPECT_VECTOR_EQ(i32vres, dtoi32_res);
            }
        }
    }

    void test_to_int64()
    {
        int64_vector i64vres(int64_batch::size);
        {
            int64_batch i64bres = xsimd::bitwise_cast<int64_batch>(f_input());
            i64bres.store_aligned(i64vres.data());
            {
                INFO(print_function_name("to_int64(float)"));
                EXPECT_VECTOR_EQ(i64vres, ftoi64_res);
            }
        }
        {
            int64_batch i64bres = xsimd::bitwise_cast<int64_batch>(d_input());
            i64bres.store_aligned(i64vres.data());
            {
                INFO(print_function_name("to_int64(double)"));
                EXPECT_VECTOR_EQ(i64vres, dtoi64_res);
            }
        }
    }

    void test_to_float()
    {
        float_vector fvres(float_batch::size);
        {
            float_batch fbres = xsimd::bitwise_cast<float_batch>(i32_input());
            fbres.store_aligned(fvres.data());
            {
                INFO(print_function_name("to_float(int32_t)"));
                EXPECT_VECTOR_EQ(fvres, i32tof_res);
            }
        }
        {
            float_batch fbres = xsimd::bitwise_cast<float_batch>(i64_input());
            fbres.store_aligned(fvres.data());
            {
                INFO(print_function_name("to_float(int64_t)"));
                EXPECT_VECTOR_EQ(fvres, i64tof_res);
            }
        }
        {
            float_batch fbres = xsimd::bitwise_cast<float_batch>(d_input());
            fbres.store_aligned(fvres.data());
            {
                INFO(print_function_name("to_float(double)"));
                EXPECT_VECTOR_EQ(fvres, dtof_res);
            }
        }
    }

    void test_to_double()
    {
        double_vector dvres(double_batch::size);
        {
            double_batch dbres = xsimd::bitwise_cast<double_batch>(i32_input());
            dbres.store_aligned(dvres.data());
            {
                INFO(print_function_name("to_double(int32_t)"));
                EXPECT_VECTOR_EQ(dvres, i32tod_res);
            }
        }
        {
            double_batch dbres = xsimd::bitwise_cast<double_batch>(i64_input());
            dbres.store_aligned(dvres.data());
            {
                INFO(print_function_name("to_double(int64_t)"));
                EXPECT_VECTOR_EQ(dvres, i64tod_res);
            }
        }
        {
            double_batch dbres = xsimd::bitwise_cast<double_batch>(f_input());
            dbres.store_aligned(dvres.data());
            {
                INFO(print_function_name("to_double(float)"));
                EXPECT_VECTOR_EQ(dvres, ftod_res);
            }
        }
    }

private:

    int32_batch i32_input() const
    {
        return int32_batch(2);
    }

    int64_batch i64_input() const
    {
        return int64_batch(2);
    }

    float_batch f_input() const
    {
        return float_batch(3.);
    }

    double_batch d_input() const
    {
        return double_batch(2.5e17);
    }

    union bitcast
    {
        float f[2];
        int32_t i32[2];
        int64_t i64;
        double d;
    };
};


TEST_CASE_TEMPLATE_DEFINE("to_int32", TypeParam, bitwise_cast_test_to_int32)
{
    bitwise_cast_test<TypeParam> tester;
    tester.test_to_int32();
}

TEST_CASE_TEMPLATE_DEFINE("to_int64", TypeParam, bitwise_cast_test_to_int64)
{
    bitwise_cast_test<TypeParam> tester;
    tester.test_to_int64();
}

TEST_CASE_TEMPLATE_DEFINE("to_float", TypeParam, bitwise_cast_test_to_float)
{
    bitwise_cast_test<TypeParam> tester;
    tester.test_to_float();
}

TEST_CASE_TEMPLATE_DEFINE("to_double", TypeParam, bitwise_cast_test_to_double)
{
    bitwise_cast_test<TypeParam> tester;
    tester.test_to_double();
}
TEST_CASE_TEMPLATE_APPLY(bitwise_cast_test_to_int32, conversion_types);
TEST_CASE_TEMPLATE_APPLY(bitwise_cast_test_to_int64, conversion_types);
TEST_CASE_TEMPLATE_APPLY(bitwise_cast_test_to_float, conversion_types);
TEST_CASE_TEMPLATE_APPLY(bitwise_cast_test_to_double, conversion_types);
