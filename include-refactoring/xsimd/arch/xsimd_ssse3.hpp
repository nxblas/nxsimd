#ifndef XSIMD_SSSE3_HPP
#define XSIMD_SSSE3_HPP

#include "../types/xsimd_ssse3_register.hpp"

namespace xsimd {

  namespace kernel {
    using namespace types;

    // abs
    template<class A, class T, typename std::enable_if<std::is_integral<T>::value && std::is_signed<T>::value, void>::type>
    batch<T, A> abs(batch<T, A> const& self, requires<ssse3>) {
      switch(sizeof(T)) {
        case 1: return _mm_abs_epi8(self);
        case 2: return _mm_abs_epi16(self);
        case 4: return _mm_abs_epi32(self);
        case 8: return _mm_abs_epi64(self);
        default: assert(false && "unsupported arch/op combination"); return {};
      }
    }

    // hadd
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    T hadd(batch<T, A> const& self, requires<ssse3>) {
      switch(sizeof(T)) {
        case 2: {
                __m128i tmp1 = _mm_hadd_epi16(self, self);
                __m128i tmp2 = _mm_hadd_epi16(tmp1, tmp1);
                __m128i tmp3 = _mm_hadd_epi16(tmp2, tmp2);
                return _mm_cvtsi128_si32(tmp3) & 0xFFFF;
                }
        case 4: {
                __m128i tmp1 = _mm_hadd_epi32(self, self);
                __m128i tmp2 = _mm_hadd_epi32(tmp1, tmp1);
                return _mm_cvtsi128_si32(tmp2);
                }
        default: return hadd(self, sse3{});
      }
    }
  }

}

#endif

