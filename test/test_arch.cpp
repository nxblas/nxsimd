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
#include <numeric>

#include "test_utils.hpp"

#if XSIMD_INSTR_SET != XSIMD_VERSION_NUMBER_NOT_AVAILABLE || XSIMD_ENABLE_FALLBACK

static_assert(xsimd::arch::default_::supported, "default arch must be supported");
static_assert(xsimd::arch::supported::contains<xsimd::arch::default_>(), "default arch is supported");
static_assert(xsimd::arch::all::contains<xsimd::arch::default_>(), "default arch is a valid arch");
static_assert(!(xsimd::arch::x86::supported && xsimd::arch::arm::supported), "either x86 or arm, but not both");

struct check_supported {
  template<class Arch>
  void operator()(Arch) const {
    static_assert(Arch::supported, "not supported?");
  }
};

TEST(arch, supported)
{
  xsimd::arch::supported::for_each(check_supported{});
}

struct check_available {
  template<class Arch>
  void operator()(Arch) const {
    EXPECT_TRUE(Arch::available());
  }
};

TEST(arch, available)
{
  EXPECT_TRUE(xsimd::arch::default_::available());
  xsimd::arch::supported::for_each(check_available{});
}

struct sum {
  template<class Arch, class T>
  T operator()(Arch, T const* data, unsigned size)
  {
    using batch = xsimd::arch::batch<T, Arch>;
    batch acc(static_cast<T>(0));
    const unsigned n = size / batch::size * batch::size;
    for(unsigned i = 0; i != n; i += batch::size)
        acc += batch(data + i);
    T star_acc = xsimd::hadd(acc);
    for(unsigned i = n; i < size; ++i)
      star_acc += data[i];
    return star_acc;
  }
};

TEST(arch, dispatcher)
{
  uint32_t data[17] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17 };
  uint32_t ref = std::accumulate(std::begin(data), std::end(data), 0);

  // platform specific
  {
    auto dispatched = xsimd::arch::dispatch(sum{});
    uint32_t res = dispatched(data, 17);
    EXPECT_EQ(ref, res);
  }

#if defined(__SSE2__) && defined(__AVX__)
  {
    namespace xarch = xsimd::arch;
    auto dispatched = xarch::dispatch<sum, xarch::arch_list<xarch::avx, xarch::sse2>>(sum{});
    uint32_t res = dispatched(data, 17);
    EXPECT_EQ(ref, res);
  }
#endif
}

#if XSIMD_ENABLE_FALLBACK
// FIXME: this should be different from fallback
TEST(arch, scalar)
{
  uint32_t data[17] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17 };
  uint32_t ref = std::accumulate(std::begin(data), std::end(data), 0);

  uint32_t res = sum{}(xsimd::arch::scalar{}, data, 17);
  EXPECT_EQ(ref, res);

}
#endif

#endif
