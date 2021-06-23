#ifndef TEST_COMMON_MACROS_HPP
#define TEST_COMMON_MACROS_HPP

#if defined(XSIMD_NO_EXCEPTIONS)
    #define DOCTEST_CONFIG_NO_EXCEPTIONS
#endif

#include "doctest/doctest.h"
#include "xsimd/config/xsimd_config.hpp"

//
#if defined(XSIMD_NO_EXCEPTIONS)

#define XT_EXPECT_THROW(x, y)  #warning "XT_EXPECT_THROW IS DISABLED WHEN EXCEPTIONS ARE DISABLED" ;
#define XT_ASSERT_THROW(x, y)  #warning "XT_ASSERT_THROW IS DISABLED WHEN EXCEPTIONS ARE DISABLED" ;
#define XT_EXPECT_ANY_THROW(x) #warning "XT_EXPECT_ANY_THROW IS DISABLED WHEN EXCEPTIONS ARE DISABLED" ;
#define XT_ASSERT_ANY_THROW(x) #warning "XT_ASSERT_ANY_THROW IS DISABLED WHEN EXCEPTIONS ARE DISABLED" ;
#define XT_EXPECT_NO_THROW(x) x;
#define XT_ASSERT_NO_THROW(x) x;

#else

#define XT_EXPECT_THROW(x, y) CHECK_THROWS_AS(x, y);
#define XT_ASSERT_THROW(x, y) REQUIRE_THROWS_AS(x, y);
#define XT_EXPECT_ANY_THROW(x) CHECK_THROWS_AS(x, std::exception);
#define XT_ASSERT_ANY_THROW(x) REQUIRE_THROWS_AS(x, std::exception);
#define XT_EXPECT_NO_THROW(x) x;
#define XT_ASSERT_NO_THROW(x) x;
#endif

#endif

#define EXPECT_NO_THROW(x) XT_EXPECT_NO_THROW(x)
#define EXPECT_THROW(x, y) XT_EXPECT_THROW(x,y)

#define TEST(A,B) TEST_CASE(#A"."#B )
#define EXPECT_EQ(A,B) CHECK_EQ(A,B)
#define EXPECT_NE(A,B) CHECK_NE(A,B)
#define EXPECT_LE(A,B) CHECK_LE(A,B)
#define EXPECT_GE(A,B) CHECK_GE(A,B)
#define EXPECT_LT(A,B) CHECK_LT(A,B)
#define EXPECT_GT(A,B) CHECK_GT(A,B)
#define EXPECT_TRUE(A) CHECK_UNARY(A)
#define EXPECT_FALSE(A) CHECK_UNARY_FALSE(A)
#define EXPECT_NEAR(x,y,t) CHECK(x == doctest::Approx(y).epsilon(t))

#define ASSERT_EQ(A,B) REQUIRE_EQ(A,B)
#define ASSERT_NE(A,B) REQUIRE_NE(A,B)
#define ASSERT_LE(A,B) REQUIRE_LE(A,B)
#define ASSERT_GE(A,B) REQUIRE_GE(A,B)
#define ASSERT_LT(A,B) REQUIRE_LT(A,B)
#define ASSERT_GT(A,B) REQUIRE_GT(A,B)
#define ASSERT_TRUE(A) REQUIRE_UNARY(A)
#define ASSERT_FALSE(A) REQUIRE_UNARY_FALSE(A)
#define REQUIRE_NEAR(x,y,t) REQUIRE(x == doctest::Approx(y).epsilon(t))
#define EXPECT_DOUBLE_EQ(x,y)   CHECK(x == doctest::Approx(y));

#define TEST_F(FIXTURE_CLASS, NAME)\
    TEST_CASE_FIXTURE(FIXTURE_CLASS, #NAME)
