/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay, Wolf Vollprecht and         *
* Martin Renou                                                             *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include <cmath>

#include "test_utils.hpp"

#include "xsimd/math/xsimd_numerical_constant.hpp"

template<typename T>
struct numerical_constant : public testing::Test
{
      using type = T;
};

using FloatingPointTypes = testing::Types<float, double>;
TYPED_TEST_SUITE(numerical_constant, FloatingPointTypes);

TYPED_TEST(numerical_constant, constants)
{
    using T = typename TestFixture::type;
    EXPECT_TRUE(std::isinf(xsimd::infinity<T>()) && xsimd::infinity<T>() > 0);
    EXPECT_EQ(xsimd::invlog_2<T>(), T(1) / std::log(T(2)));
    EXPECT_EQ(xsimd::invlog10_2<T>(), T(1) / std::log10(T(2)));
    EXPECT_EQ(xsimd::log_2<T>(), std::log(T(2)));
    EXPECT_TRUE(std::isinf(xsimd::minusinfinity<T>()) && xsimd::minusinfinity<T>() < 0);
    EXPECT_EQ(xsimd::minuszero<T>(), -T(0));
    EXPECT_TRUE(std::isnan(xsimd::nan<T>()));
    EXPECT_TRUE(xsimd::smallestposval<T>() < std::numeric_limits<T>::epsilon());
}

