#ifndef XSIMD_SSE_HPP
#define XSIMD_SSE_HPP

#include "../types/xsimd_sse_register.hpp"


namespace xsimd {

  namespace kernel {
    using namespace types;

    // abs
    template<class A> batch<float, A> abs(batch<float, A> const& self, requires<sse>) {
      __m128 sign_mask = _mm_set1_ps(-0.f);  // -0.f = 1 << 31
      return _mm_andnot_ps(sign_mask, self);
    }
    template<class A> batch<double, A> abs(batch<double, A> const& self, requires<sse>) {
      __m128d sign_mask = _mm_set1_pd(-0.f);  // -0.f = 1 << 31
      return _mm_andnot_pd(sign_mask, self);
    }

    // add
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> add(batch<T, A> const&, batch<T, A> const&, requires<sse>) {
      static_assert(std::is_same<A, sse>::value, "unsupported arch / op combination");
    }
    template<class A> batch<float, A> add(batch<float, A> const& self, batch<float, A> const& other, requires<sse>) {
      return _mm_add_ps(self, other);
    }
    template<class A> batch<double, A> add(batch<double, A> const& self, batch<double, A> const& other, requires<sse>) {
      return _mm_add_pd(self, other);
    }

    // all
    template<class A> bool all(batch_bool<float, A> const& self, requires<sse>) {
      return _mm_movemask_ps(self) == 0x0F;
    }

    // any
    template<class A> bool any(batch_bool<float, A> const& self, requires<sse>) {
      return _mm_movemask_ps(self) != 0;
    }

    // bitwise_and
    template<class A> batch<float, A> bitwise_and(batch<float, A> const& self, batch<float, A> const& other, requires<sse>) {
      return _mm_and_ps(self, other);
    }
    template<class A> batch<double, A> bitwise_and(batch<double, A> const& self, batch<double, A> const& other, requires<sse>) {
      return _mm_and_pd(self, other);
    }

    template<class A> batch_bool<float, A> bitwise_and(batch_bool<float, A> const& self, batch_bool<float, A> const& other, requires<sse>) {
      return _mm_and_ps(self, other);
    }
    template<class A> batch_bool<double, A> bitwise_and(batch_bool<double, A> const& self, batch_bool<double, A> const& other, requires<sse>) {
      return _mm_and_pd(self, other);
    }

    // bitwise_or
    template<class A> batch<float, A> bitwise_or(batch<float, A> const& self, batch<float, A> const& other, requires<sse>) {
      return _mm_or_ps(self, other);
    }
    template<class A> batch<double, A> bitwise_or(batch<double, A> const& self, batch<double, A> const& other, requires<sse>) {
      return _mm_or_pd(self, other);
    }
    template<class A> batch_bool<float, A> bitwise_or(batch_bool<float, A> const& self, batch_bool<float, A> const& other, requires<sse>) {
      return _mm_or_ps(self, other);
    }
    template<class A> batch_bool<double, A> bitwise_or(batch_bool<double, A> const& self, batch_bool<double, A> const& other, requires<sse>) {
      return _mm_or_pd(self, other);
    }

    // bitwise_xor
    template<class A> batch<float, A> bitwise_xor(batch<float, A> const& self, batch<float, A> const& other, requires<sse>) {
      return _mm_xor_ps(self, other);
    }
    template<class A> batch<double, A> bitwise_xor(batch<double, A> const& self, batch<double, A> const& other, requires<sse>) {
      return _mm_xor_pd(self, other);
    }
    template<class A> batch_bool<float, A> bitwise_xor(batch_bool<float, A> const& self, batch_bool<float, A> const& other, requires<sse>) {
      return _mm_xor_ps(self, other);
    }
    template<class A> batch_bool<double, A> bitwise_xor(batch_bool<double, A> const& self, batch_bool<double, A> const& other, requires<sse>) {
      return _mm_xor_pd(self, other);
    }

    // bitwise_cast
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<float, A> bitwise_cast(batch<T, A> const& self, batch<float, A> const &, requires<sse>) {
      return _mm_castsi128_ps(self);
    }
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<double, A> bitwise_cast(batch<T, A> const& self, batch<double, A> const &, requires<sse>) {
      return _mm_castsi128_pd(self);
    }
    template<class A, class T, class Tp, class=typename std::enable_if<std::is_integral<typename std::common_type<T, Tp>::type>::value, void>::type>
    batch<Tp, A> bitwise_cast(batch<T, A> const& self, batch<Tp, A> const &, requires<sse>) {
      return batch<Tp, A>(self.data);
    }
    template<class A>
    batch<double, A> bitwise_cast(batch<float, A> const& self, batch<double, A> const &, requires<sse>) {
      return _mm_castps_pd(self);
    }
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> bitwise_cast(batch<float, A> const& self, batch<T, A> const &, requires<sse>) {
      return _mm_castps_si128(self);
    }
    template<class A>
    batch<float, A> bitwise_cast(batch<double, A> const& self, batch<float, A> const &, requires<sse>) {
      return _mm_castpd_ps(self);
    }
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> bitwise_cast(batch<double, A> const& self, batch<T, A> const &, requires<sse>) {
      return _mm_castpd_si128(self);
    }

    // bitwise_not
    template<class A> batch<float, A> bitwise_not(batch<float, A> const& self, requires<sse>) {
      return _mm_xor_ps(self, _mm_castsi128_ps(_mm_set1_epi32(-1)));
    }
    template <class A>
    batch<double, A> bitwise_not(batch<double, A> const &self, requires<sse>) {
      return _mm_xor_pd(self, _mm_castsi128_pd(_mm_set1_epi32(-1)));
    }
    template<class A> batch_bool<float, A> bitwise_not(batch_bool<float, A> const& self, requires<sse>) {
      return _mm_xor_ps(self, _mm_castsi128_ps(_mm_set1_epi32(-1)));
    }
    template <class A>
    batch_bool<double, A> bitwise_not(batch_bool<double, A> const &self, requires<sse>) {
      return _mm_xor_pd(self, _mm_castsi128_pd(_mm_set1_epi32(-1)));
    }

    // bool_cast
    template<class A> batch_bool<int32_t, A> bool_cast(batch_bool<float, A> const& self, requires<sse>) {
        return _mm_castps_si128(self);
    }
    template<class A> batch_bool<float, A> bool_cast(batch_bool<int32_t, A> const& self, requires<sse>) {
        return _mm_castsi128_ps(self);
    }
    template<class A> batch_bool<int64_t, A> bool_cast(batch_bool<double, A> const& self, requires<sse>) {
        return _mm_castpd_si128(self);
    }
    template<class A> batch_bool<double, A> bool_cast(batch_bool<int64_t, A> const& self, requires<sse>) {
        return _mm_castsi128_pd(self);
    }

    // broadcast
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> broadcast(T val, requires<sse>) {
      switch(sizeof(T)) {
        case 1: return _mm_set1_epi8(val);
        case 2: return _mm_set1_epi16(val);
        case 4: return _mm_set1_epi32(val);
        default: assert(false && "unsupported"); return {};
      }
    }
    template<class A> batch<float, A> broadcast(float val, requires<sse>) {
      return _mm_set1_ps(val);
    }
    template<class A> batch<double, A> broadcast(double, requires<sse>) {
      assert(false && "unsupported");
      return {};
    }

    // complex_low
    template<class A> batch<float, A> complex_low(batch<std::complex<float>, A> const& self, requires<sse>) {
      return _mm_unpacklo_ps(self.real(), self.imag());
    }
    template<class A> batch<double, A> complex_low(batch<std::complex<double>, A> const& self, requires<sse>) {
      return _mm_unpacklo_pd(self.real(), self.imag());
    }

    // complex_high
    template<class A> batch<float, A> complex_high(batch<std::complex<float>, A> const& self, requires<sse>) {
      return _mm_unpackhi_ps(self.real(), self.imag());
    }
    template<class A> batch<double, A> complex_high(batch<std::complex<double>, A> const& self, requires<sse>) {
      return _mm_unpackhi_pd(self.real(), self.imag());
    }

    // div
    template<class A> batch<float, A> div(batch<float, A> const& self, batch<float, A> const& other, requires<sse>) {
      return _mm_div_ps(self, other);
    }
    template<class A> batch<double, A> div(batch<double, A> const& self, batch<double, A> const& other, requires<sse>) {
      return _mm_div_pd(self, other);
    }

    // eq
    template<class A> batch_bool<float, A> eq(batch<float, A> const& self, batch<float, A> const& other, requires<sse>) {
      return _mm_cmpeq_ps(self, other);
    }
    template<class A> batch_bool<double, A> eq(batch<double, A> const& self, batch<double, A> const& other, requires<sse>) {
      return _mm_cmpeq_pd(self, other);
    }
    template<class A> batch_bool<float, A> eq(batch_bool<float, A> const& self, batch_bool<float, A> const& other, requires<sse>) {
      return  _mm_castsi128_ps(_mm_cmpeq_epi32(_mm_castps_si128(self), _mm_castps_si128(other)));
    }
    template<class A> batch_bool<double, A> eq(batch_bool<double, A> const& self, batch_bool<double, A> const& other, requires<sse>) {
      return  _mm_castsi128_pd(_mm_cmpeq_epi32(_mm_castpd_si128(self), _mm_castpd_si128(other)));
    }

    // ge
    template<class A> batch_bool<float, A> ge(batch<float, A> const& self, batch<float, A> const& other, requires<sse>) {
      return _mm_cmpge_ps(self, other);
    }
    template<class A> batch_bool<double, A> ge(batch<double, A> const& self, batch<double, A> const& other, requires<sse>) {
      return _mm_cmpge_pd(self, other);
    }

    // gt
    template<class A> batch_bool<float, A> gt(batch<float, A> const& self, batch<float, A> const& other, requires<sse>) {
      return _mm_cmpgt_ps(self, other);
    }
    template<class A> batch_bool<double, A> gt(batch<double, A> const& self, batch<double, A> const& other, requires<sse>) {
      return _mm_cmpgt_pd(self, other);
    }

    // hadd
    template<class A> float hadd(batch<float, A> const& self, requires<sse>) {
      __m128 tmp0 = _mm_add_ps(self, _mm_movehl_ps(self, self));
      __m128 tmp1 = _mm_add_ss(tmp0, _mm_shuffle_ps(tmp0, tmp0, 1));
      return _mm_cvtss_f32(tmp1);
    }
    template <class A>
    double hadd(batch<double, A> const &self, requires<sse>) {
      return _mm_cvtsd_f64(_mm_add_sd(self, _mm_unpackhi_pd(self, self)));
    }
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    T hadd(batch<T, A> const& self, requires<sse>) {
        alignas(A::alignment()) T buffer[batch<T, A>::size];
        self.store_aligned(buffer);
        T res = 0;
        for (T val : buffer)
        {
            res += val;
        }
        return res;
    }

    // haddp
    template<class A> batch<float, A> haddp(batch<float, A> const* row, requires<sse>) {
      __m128 tmp0 = _mm_unpacklo_ps(row[0], row[1]);
      __m128 tmp1 = _mm_unpackhi_ps(row[0], row[1]);
      __m128 tmp2 = _mm_unpackhi_ps(row[2], row[3]);
      tmp0 = _mm_add_ps(tmp0, tmp1);
      tmp1 = _mm_unpacklo_ps(row[2], row[3]);
      tmp1 = _mm_add_ps(tmp1, tmp2);
      tmp2 = _mm_movehl_ps(tmp1, tmp0);
      tmp0 = _mm_movelh_ps(tmp0, tmp1);
      return _mm_add_ps(tmp0, tmp2);
    }
    template <class A>
    batch<double, A> haddp(batch<double, A> const *row, requires<sse>) {
      return _mm_add_pd(_mm_unpacklo_pd(row[0], row[1]),
          _mm_unpackhi_pd(row[0], row[1]));
    }

    // isnan
    template<class A> batch_bool<float, A> isnan(batch<float, A> const& self, requires<sse>) {
      return _mm_cmpunord_ps(self, self);
    }
    template<class A> batch_bool<double, A> isnan(batch<double, A> const& self, requires<sse>) {
      return _mm_cmpunord_pd(self, self);
    }

    // le
    template<class A> batch_bool<float, A> le(batch<float, A> const& self, batch<float, A> const& other, requires<sse>) {
      return _mm_cmple_ps(self, other);
    }
    template<class A> batch_bool<double, A> le(batch<double, A> const& self, batch<double, A> const& other, requires<sse>) {
      return _mm_cmple_pd(self, other);
    }

    // load_aligned
    template<class A> batch<float, A> load_aligned(float const* mem, convert<float>, requires<sse>) {
      return _mm_load_ps(mem);
    }
    template<class A> batch<double, A> load_aligned(double const* mem, convert<double>, requires<sse>) {
      return _mm_load_pd(mem);
    }

    // load_complex
    template<class A> batch<std::complex<float>, A> load_complex(batch<float, A> const& hi, batch<float, A> const& lo, requires<sse>) {
      return {_mm_shuffle_ps(hi, lo, _MM_SHUFFLE(2, 0, 2, 0)), _mm_shuffle_ps(hi, lo, _MM_SHUFFLE(3, 1, 3, 1))};
    }
    template<class A> batch<std::complex<double>, A> load_complex(batch<double, A> const& hi, batch<double, A> const& lo, requires<sse>) {
      return {_mm_shuffle_pd(hi, lo, _MM_SHUFFLE2(0, 0)), _mm_shuffle_pd(hi, lo, _MM_SHUFFLE2(1, 1))};
    }

    // load_unaligned
    template<class A> batch<float, A> load_unaligned(float const* mem, convert<float>, requires<sse>){
      return _mm_loadu_ps(mem);
    }
    template<class A> batch<double, A> load_unaligned(double const* mem, convert<double>, requires<sse>){
      return _mm_loadu_pd(mem);
    }

    // lt
    template<class A> batch_bool<float, A> lt(batch<float, A> const& self, batch<float, A> const& other, requires<sse>) {
      return _mm_cmplt_ps(self, other);
    }
    template<class A> batch_bool<double, A> lt(batch<double, A> const& self, batch<double, A> const& other, requires<sse>) {
      return _mm_cmplt_pd(self, other);
    }

    // max
    template<class A> batch<float, A> max(batch<float, A> const& self, batch<float, A> const& other, requires<sse>) {
      return _mm_max_ps(self, other);
    }
    template<class A> batch<double, A> max(batch<double, A> const& self, batch<double, A> const& other, requires<sse>) {
      return _mm_max_pd(self, other);
    }

    // min
    template<class A> batch<float, A> min(batch<float, A> const& self, batch<float, A> const& other, requires<sse>) {
      return _mm_min_ps(self, other);
    }
    template<class A> batch<double, A> min(batch<double, A> const& self, batch<double, A> const& other, requires<sse>) {
      return _mm_min_pd(self, other);
    }

    // mul
    template<class A> batch<float, A> mul(batch<float, A> const& self, batch<float, A> const& other, requires<sse>) {
      return _mm_mul_ps(self, other);
    }
    template<class A> batch<double, A> mul(batch<double, A> const& self, batch<double, A> const& other, requires<sse>) {
      return _mm_mul_pd(self, other);
    }

    // neg
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> neg(batch<T, A> const& self, requires<sse>) {
      return 0 - self;
    }
    template<class A> batch<float, A> neg(batch<float, A> const& self, requires<sse>) {
      return _mm_xor_ps(self, _mm_castsi128_ps(_mm_set1_epi32(0x80000000)));
    }
    template <class A>
    batch<double, A> neg(batch<double, A> const &self, requires<sse>) {
      return _mm_xor_pd(
          self, _mm_castsi128_pd(_mm_setr_epi32(0, 0x80000000, 0, 0x80000000)));
    }

    // neq
    template<class A> batch_bool<float, A> neq(batch<float, A> const& self, batch<float, A> const& other, requires<sse>) {
      return _mm_cmpneq_ps(self, other);
    }
    template<class A> batch_bool<double, A> neq(batch<double, A> const& self, batch<double, A> const& other, requires<sse>) {
      return _mm_cmpneq_pd(self, other);
    }
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch_bool<T, A> neq(batch<T, A> const& self, batch<T, A> const& other, requires<sse>) {
        return ~(self == other);
    }

    template<class A> batch_bool<float, A> neq(batch_bool<float, A> const& self, batch_bool<float, A> const& other, requires<sse>) {
      return _mm_cmpneq_ps(self, other);
    }
    template<class A> batch_bool<double, A> neq(batch_bool<double, A> const& self, batch_bool<double, A> const& other, requires<sse>) {
      return _mm_cmpneq_pd(self, other);
    }
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch_bool<T, A> neq(batch_bool<T, A> const& self, batch_bool<T, A> const& other, requires<sse>) {
        return ~(self == other);
    }

    // sadd
    template<class A> batch<float, A> sadd(batch<float, A> const& self, batch<float, A> const& other, requires<sse>) {
      return _mm_add_ps(self, other); // no saturated arithmetic on floating point numbers
    }
    template<class A> batch<double, A> sadd(batch<double, A> const& self, batch<double, A> const& other, requires<sse>) {
      return _mm_add_pd(self, other); // no saturated arithmetic on floating point numbers
    }
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> sadd(batch<T, A> const& self, batch<T, A> const& other, requires<sse>) {
      if(std::is_signed<T>::value) {
        auto mask = (other >> (8 * sizeof(T) - 1));
        auto self_pos_branch = min(std::numeric_limits<T>::max() - other, self);
        auto self_neg_branch = max(std::numeric_limits<T>::min() - other, self);
        return other + select(batch_bool<T, A>(mask.data), self_neg_branch, self_pos_branch);
      }
      else {
        const auto diffmax = std::numeric_limits<T>::max() - self;
        const auto mindiff = min(diffmax, other);
        return self + mindiff;
      }
    }

    // set
    template<class A, class... Values>
    batch<float, A> set(batch<float, A> const&, requires<sse>, Values... values) {
      static_assert(sizeof...(Values) == batch<float, A>::size, "consistent init");
      return _mm_setr_ps(values...);
    }

    template<class A, class... Values>
    batch<double, A> set(batch<double, A> const&, requires<sse>, Values... values) {
      static_assert(sizeof...(Values) == batch<double, A>::size, "consistent init");
      return _mm_setr_pd(values...);
    }

    // select
    template<class A> batch<float, A> select(batch_bool<float, A> const& cond, batch<float, A> const& true_br, batch<float, A> const& false_br, requires<sse>) {
      return _mm_or_ps(_mm_and_ps(cond, true_br), _mm_andnot_ps(cond, false_br));
    }
    template<class A> batch<double, A> select(batch_bool<double, A> const& cond, batch<double, A> const& true_br, batch<double, A> const& false_br, requires<sse>) {
      return _mm_or_pd(_mm_and_pd(cond, true_br), _mm_andnot_pd(cond, false_br));
    }

    // sqrt
    template<class A> batch<float, A> sqrt(batch<float, A> const& val, requires<sse>) {
      return _mm_sqrt_ps(val);
    }
    template<class A> batch<double, A> sqrt(batch<double, A> const& val, requires<sse>) {
      return _mm_sqrt_pd(val);
    }

    // ssub
    template<class A> batch<float, A> ssub(batch<float, A> const& self, batch<float, A> const& other, requires<sse>) {
      return _mm_sub_ps(self, other); // no saturated arithmetic on floating point numbers
    }
    template<class A> batch<double, A> ssub(batch<double, A> const& self, batch<double, A> const& other, requires<sse>) {
      return _mm_sub_pd(self, other); // no saturated arithmetic on floating point numbers
    }
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> ssub(batch<T, A> const& self, batch<T, A> const& other, requires<sse>) {
      if(std::is_signed<T>::value) {
         return sadd(self, -other);
      }
      else {
        const auto diff = min(self, other);
        return self - diff;
      }
    }

    // store_aligned
    template<class A> void store_aligned(float *mem, batch<float, A> const& self, requires<sse>) {
      return _mm_store_ps(mem, self);
    }
    template<class A> void store_aligned(double *mem, batch<double, A> const& self, requires<sse>) {
      return _mm_store_pd(mem, self);
    }

    // store_unaligned
    template<class A> void store_unaligned(float *mem, batch<float, A> const& self, requires<sse>) {
      return _mm_storeu_ps(mem, self);
    }
    template<class A> void store_unaligned(double *mem, batch<double, A> const& self, requires<sse>) {
      return _mm_storeu_pd(mem, self);
    }

    // sub
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> sub(batch<T, A> const&, batch<T, A> const&, requires<sse>) {
      static_assert(std::is_same<A, sse>::value, "unsupported arch / op combination");
    }
    template<class A> batch<float, A> sub(batch<float, A> const& self, batch<float, A> const& other, requires<sse>) {
      return _mm_sub_ps(self, other);
    }
    template<class A> batch<double, A> sub(batch<double, A> const& self, batch<double, A> const& other, requires<sse>) {
      return _mm_sub_pd(self, other);
    }

    // zip_hi
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> zip_hi(batch<T, A> const&, batch<T, A> const&, requires<sse>) {
      static_assert(std::is_same<A, sse>::value, "unsupported arch / op combination");
    }
    template<class A> batch<float, A> zip_hi(batch<float, A> const& self, batch<float, A> const& other, requires<sse>) {
      return _mm_unpackhi_ps(self, other);
    }
    template<class A> batch<double, A> zip_hi(batch<double, A> const& self, batch<double, A> const& other, requires<sse>) {
      return _mm_unpackhi_pd(self, other);
    }

    // zip_lo
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> zip_lo(batch<T, A> const&, batch<T, A> const&, requires<sse>) {
      static_assert(std::is_same<A, sse>::value, "unsupported arch / op combination");
    }
    template<class A> batch<float, A> zip_lo(batch<float, A> const& self, batch<float, A> const& other, requires<sse>) {
      return _mm_unpacklo_ps(self, other);
    }
    template<class A> batch<double, A> zip_lo(batch<double, A> const& self, batch<double, A> const& other, requires<sse>) {
      return _mm_unpacklo_pd(self, other);
    }
  }

}

#endif
