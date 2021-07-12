#ifndef XSIMD_SSE3_HPP
#define XSIMD_SSE3_HPP

#include "../types/xsimd_sse3_register.hpp"

namespace xsimd {

  namespace kernel {
    using namespace types;

    // hadd
    template<class A> float hadd(batch<float, A> const& self, requires<sse3>) {
      __m128 tmp0 = _mm_hadd_ps(self, self);
      __m128 tmp1 = _mm_hadd_ps(tmp0, tmp0);
      return _mm_cvtss_f32(tmp1);
    }
    template <class A>
    double hadd(batch<double, A> const &self, requires<sse3>) {
      __m128d tmp0 = _mm_hadd_pd(self, self);
      return _mm_cvtsd_f64(tmp0);
    }

    // haddp
    template<class A> batch<float, A> haddp(batch<float, A> const* row, requires<sse3>) {
      return _mm_hadd_ps(_mm_hadd_ps(row[0], row[1]),
                              _mm_hadd_ps(row[2], row[3]));
    }
    template <class A>
    batch<double, A> haddp(batch<double, A> const *row, requires<sse3>) {
      return _mm_hadd_pd(row[0], row[1]);
    }

  }

}

#endif

