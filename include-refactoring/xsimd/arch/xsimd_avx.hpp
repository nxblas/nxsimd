#ifndef XSIMD_AVX_HPP
#define XSIMD_AVX_HPP

#include "../types/xsimd_avx_register.hpp"


namespace xsimd {

  namespace kernel {
    using namespace types;

    namespace detail {
    inline void split_avx(__m256i val, __m128i& low, __m128i& high) {
        low =_mm256_castsi256_si128(val);
        high =_mm256_extractf128_si256(val, 1);
      }
      inline __m256i merge_sse(__m128i low, __m128i high) {
        return _mm256_insertf128_si256(_mm256_castsi128_si256(low), high, 1);
      }
      template<class F>
      __m256i fwd_to_sse(F f, __m256i self, __m256i other) {
        __m128i self_low, self_high, other_low, other_high;
        split_avx(self, self_low, self_high);
        split_avx(other, other_low, other_high);
        __m128i res_low = f(self_low, other_low);
        __m128i res_high = f(self_high, other_high);
        return merge_sse(res_low, res_high);
      }
      template<class F>
      __m256i fwd_to_sse(F f, __m256i self, int32_t other) {
        __m128i self_low, self_high;
        split_avx(self, self_low, self_high);
        __m128i res_low = f(self_low, other);
        __m128i res_high = f(self_high, other);
        return merge_sse(res_low, res_high);
      }
    }

    // abs
    template<class A> batch<float, A> abs(batch<float, A> const& self, requires<avx>) {
      __m256 sign_mask = _mm256_set1_ps(-0.f);  // -0.f = 1 << 31
      return _mm256_andnot_ps(sign_mask, self);
    }
    template<class A> batch<double, A> abs(batch<double, A> const& self, requires<avx>) {
      __m256d sign_mask = _mm256_set1_pd(-0.f);  // -0.f = 1 << 31
      return _mm256_andnot_pd(sign_mask, self);
    }

    // add
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> add(batch<T, A> const& self, batch<T, A> const& other, requires<avx>) {
      switch(sizeof(T)) {
        case 1: return _mm256_add_epi8(self, other);
        case 2: return _mm256_add_epi16(self, other);
        case 4: return detail::fwd_to_sse([](__m128i s, __m128i o) { return add(batch<T, sse4_2>(s), batch<T, sse4_2>(o)); }, self, other);
        case 8: return detail::fwd_to_sse([](__m128i s, __m128i o) { return add(batch<T, sse4_2>(s), batch<T, sse4_2>(o)); }, self, other);
        default: assert(false && "unsupported arch/op combination"); return {};
      }
    }
    template<class A> batch<float, A> add(batch<float, A> const& self, batch<float, A> const& other, requires<avx>) {
      return _mm256_add_ps(self, other);
    }
    template<class A> batch<double, A> add(batch<double, A> const& self, batch<double, A> const& other, requires<avx>) {
      return _mm256_add_pd(self, other);
    }

    // all
    template<class A> bool all(batch_bool<float, A> const& self, requires<avx>) {
      return _mm256_testc_ps(self, batch_bool<float, A>(true)) != 0;
    }
    template<class A> bool all(batch_bool<double, A> const& self, requires<avx>) {
      return _mm256_testc_pd(self, batch_bool<double, A>(true)) != 0;
    }
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    bool all(batch_bool<T, A> const& self, requires<avx>) {
                return _mm256_testc_si256(self, batch_bool<T, A>(true)) != 0;
    }

    // any
    template<class A> bool any(batch_bool<float, A> const& self, requires<avx>) {
      return !_mm256_testz_ps(self, self);
    }
    template<class A> bool any(batch_bool<double, A> const& self, requires<avx>) {
      return !_mm256_testz_pd(self, self);
    }
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    bool any(batch_bool<T, A> const& self, requires<avx>) {
                return !_mm256_testz_si256(self, self);
    }

    // bitwise_and
    template<class A> batch<float, A> bitwise_and(batch<float, A> const& self, batch<float, A> const& other, requires<avx>) {
      return _mm256_and_ps(self, other);
    }
    template<class A> batch<double, A> bitwise_and(batch<double, A> const& self, batch<double, A> const& other, requires<avx>) {
      return _mm256_and_pd(self, other);
    }

    template<class A> batch_bool<float, A> bitwise_and(batch_bool<float, A> const& self, batch_bool<float, A> const& other, requires<avx>) {
      return _mm256_and_ps(self, other);
    }
    template<class A> batch_bool<double, A> bitwise_and(batch_bool<double, A> const& self, batch_bool<double, A> const& other, requires<avx>) {
      return _mm256_and_pd(self, other);
    }

    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> bitwise_and(batch<T, A> const& self, batch<T, A> const& other, requires<avx>) {
      return detail::fwd_to_sse([](__m128i s, __m128i o) { return bitwise_and(batch<T, sse4_2>(s), batch<T, sse4_2>(o)); }, self, other);
    }
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch_bool<T, A> bitwise_and(batch_bool<T, A> const& self, batch_bool<T, A> const& other, requires<avx>) {
      return detail::fwd_to_sse([](__m128i s, __m128i o) { return bitwise_and(batch<T, sse4_2>(s), batch<T, sse4_2>(o)); }, self, other);
    }

    // bitwise_lshift
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> bitwise_lshift(batch<T, A> const& self, int32_t other, requires<avx>) {
      switch(sizeof(T)) {
        case 1: return detail::fwd_to_sse([](__m128i s, __m128i o) { return bitwise_and(batch<T, sse4_2>(s), batch<T, sse4_2>(o)); }, _mm256_set1_epi8(0xFF << other), _mm256_slli_epi32(self, other));

        case 2: return _mm256_slli_epi16(self, other);
        case 4: return detail::fwd_to_sse([](__m128i s, int32_t o) { return bitwise_lshift(batch<T, sse4_2>(s), o, sse4_2{}); },self, other);
        case 8: return _mm256_slli_epi64(self, other);
        default: assert(false && "unsupported arch/op combination"); return {};
      }
    }

    // bitwise_not
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> bitwise_not(batch<T, A> const& self, requires<avx>) {
      return _mm256_xor_si256(self, _mm256_set1_epi32(-1));
    }
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch_bool<T, A> bitwise_not(batch_bool<T, A> const& self, requires<avx>) {
      return _mm256_xor_si256(self, _mm256_set1_epi32(-1));
    }

    // bitwise_or
    template<class A> batch<float, A> bitwise_or(batch<float, A> const& self, batch<float, A> const& other, requires<avx>) {
      return _mm256_or_ps(self, other);
    }
    template<class A> batch<double, A> bitwise_or(batch<double, A> const& self, batch<double, A> const& other, requires<avx>) {
      return _mm256_or_pd(self, other);
    }
    template<class A> batch_bool<float, A> bitwise_or(batch_bool<float, A> const& self, batch_bool<float, A> const& other, requires<avx>) {
      return _mm256_or_ps(self, other);
    }
    template<class A> batch_bool<double, A> bitwise_or(batch_bool<double, A> const& self, batch_bool<double, A> const& other, requires<avx>) {
      return _mm256_or_pd(self, other);
    }
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> bitwise_or(batch<T, A> const& self, batch<T, A> const& other, requires<avx>) {
      return _mm256_or_si256(self, other);
    }
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch_bool<T, A> bitwise_or(batch_bool<T, A> const& self, batch_bool<T, A> const& other, requires<avx>) {
      return _mm256_or_si256(self, other);
    }

    // bitwise_rshift
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> bitwise_rshift(batch<T, A> const& self, int32_t other, requires<avx>) {
      if(std::is_signed<T>::value) {
        switch(sizeof(T)) {
          case 1: {
            __m256i sign_mask = _mm256_set1_epi16((0xFF00 >> other) & 0x00FF);
            __m256i cmp_is_negative = _mm256_cmpgt_epi8(_mm256_setzero_si256(), self);
            __m256i res = _mm256_srai_epi16(self, other);
            return _mm256_or_si256(
                detail::fwd_to_sse([](__m128i s, __m128i o) { return bitwise_and(batch<T, sse4_2>(s), batch<T, sse4_2>(o)); }, sign_mask, cmp_is_negative),
                _mm256_andnot_si256(sign_mask, res));
          }
          case 2: return _mm256_srai_epi16(self, other);
          case 4: return detail::fwd_to_sse([](__m128i s, int32_t o) { return bitwise_rshift(batch<T, sse4_2>(s), o, sse4_2{}); }, self, other);
          case 8: {
            // from https://github.com/samyvilar/vect/blob/master/vect_128.h
            return _mm256_or_si256(
                _mm256_srli_epi64(self, other),
                _mm256_slli_epi64(
                    _mm256_srai_epi32(_mm256_shuffle_epi32(self, _MM_SHUFFLE(3, 3, 1, 1)), 32),
                    64 - other));
          }
          default: assert(false && "unsupported arch/op combination"); return {};
        }
      }
      else {
        switch(sizeof(T)) {
          case 1: return detail::fwd_to_sse([](__m128i s, __m128i o) { return bitwise_and(batch<T, sse4_2>(s), batch<T, sse4_2>(o)); }, _mm256_set1_epi8(0xFF >> other), _mm256_srli_epi32(self, other));
          case 2: return _mm256_srli_epi16(self, other);
          case 4: return _mm256_srli_epi32(self, other);
          case 8: return _mm256_srli_epi64(self, other);
          default: assert(false && "unsupported arch/op combination"); return {};
        }
      }
    }

    // bitwise_xor
    template<class A> batch<float, A> bitwise_xor(batch<float, A> const& self, batch<float, A> const& other, requires<avx>) {
      return _mm256_xor_ps(self, other);
    }
    template<class A> batch<double, A> bitwise_xor(batch<double, A> const& self, batch<double, A> const& other, requires<avx>) {
      return _mm256_xor_pd(self, other);
    }
    template<class A> batch_bool<float, A> bitwise_xor(batch_bool<float, A> const& self, batch_bool<float, A> const& other, requires<avx>) {
      return _mm256_xor_ps(self, other);
    }
    template<class A> batch_bool<double, A> bitwise_xor(batch_bool<double, A> const& self, batch_bool<double, A> const& other, requires<avx>) {
      return _mm256_xor_pd(self, other);
    }
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> bitwise_xor(batch<T, A> const& self, batch<T, A> const& other, requires<avx>) {
      return _mm256_xor_si256(self, other);
    }
    // bitwise_cast
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<float, A> bitwise_cast(batch<T, A> const& self, batch<float, A> const &, requires<avx>) {
      return _mm256_castsi256_ps(self);
    }
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<double, A> bitwise_cast(batch<T, A> const& self, batch<double, A> const &, requires<avx>) {
      return _mm256_castsi256_pd(self);
    }
    template<class A, class T, class Tp, class=typename std::enable_if<std::is_integral<typename std::common_type<T, Tp>::type>::value, void>::type>
    batch<Tp, A> bitwise_cast(batch<T, A> const& self, batch<Tp, A> const &, requires<avx>) {
      return batch<Tp, A>(self.data);
    }
    template<class A>
    batch<double, A> bitwise_cast(batch<float, A> const& self, batch<double, A> const &, requires<avx>) {
      return _mm256_castps_pd(self);
    }
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> bitwise_cast(batch<float, A> const& self, batch<T, A> const &, requires<avx>) {
      return _mm256_castps_si256(self);
    }
    template<class A>
    batch<float, A> bitwise_cast(batch<double, A> const& self, batch<float, A> const &, requires<avx>) {
      return _mm256_castpd_ps(self);
    }
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> bitwise_cast(batch<double, A> const& self, batch<T, A> const &, requires<avx>) {
      return _mm256_castpd_si256(self);
    }

    // bitwise_not
    template<class A> batch<float, A> bitwise_not(batch<float, A> const& self, requires<avx>) {
      return _mm256_xor_ps(self, _mm256_castsi256_ps(_mm256_set1_epi32(-1)));
    }
    template <class A>
    batch<double, A> bitwise_not(batch<double, A> const &self, requires<avx>) {
      return _mm256_xor_pd(self, _mm256_castsi256_pd(_mm256_set1_epi32(-1)));
    }
    template<class A> batch_bool<float, A> bitwise_not(batch_bool<float, A> const& self, requires<avx>) {
      return _mm256_xor_ps(self, _mm256_castsi256_ps(_mm256_set1_epi32(-1)));
    }
    template <class A>
    batch_bool<double, A> bitwise_not(batch_bool<double, A> const &self, requires<avx>) {
      return _mm256_xor_pd(self, _mm256_castsi256_pd(_mm256_set1_epi32(-1)));
    }

    // bool_cast
    template<class A> batch_bool<int32_t, A> bool_cast(batch_bool<float, A> const& self, requires<avx>) {
        return _mm256_castps_si256(self);
    }
    template<class A> batch_bool<float, A> bool_cast(batch_bool<int32_t, A> const& self, requires<avx>) {
        return _mm256_castsi256_ps(self);
    }
    template<class A> batch_bool<int64_t, A> bool_cast(batch_bool<double, A> const& self, requires<avx>) {
        return _mm256_castpd_si256(self);
    }
    template<class A> batch_bool<double, A> bool_cast(batch_bool<int64_t, A> const& self, requires<avx>) {
        return _mm256_castsi256_pd(self);
    }

    // broadcast
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> broadcast(T val, requires<avx>) {
      switch(sizeof(T)) {
        case 1: return _mm256_set1_epi8(val);
        case 2: return _mm256_set1_epi16(val);
        case 4: return _mm256_set1_epi32(val);
        case 8: return _mm256_set1_epi64x(val);
        default: assert(false && "unsupported"); return {};
      }
    }
    template<class A> batch<float, A> broadcast(float val, requires<avx>) {
      return _mm256_set1_ps(val);
    }
    template<class A> batch<double, A> broadcast(double val, requires<avx>) {
      return _mm256_set1_pd(val);
    }

    // ceil
    template<class A> batch<float, A> ceil(batch<float, A> const& self, requires<avx>) {
      return _mm256_ceil_ps(self);
    }
    template<class A> batch<double, A> ceil(batch<double, A> const& self, requires<avx>) {
      return _mm256_ceil_pd(self);
    }


    // complex_low
    namespace detail {
        // On clang, _mm256_extractf128_ps is built upon build_shufflevector
        // which require index parameter to be a constant
        template <int index, class B>
        inline B get_half_complex_f(const B& real, const B& imag)
        {
            __m128 tmp0 = _mm256_extractf128_ps(real, index);
            __m128 tmp1 = _mm256_extractf128_ps(imag, index);
            __m128 tmp2 = _mm_unpackhi_ps(tmp0, tmp1);
            tmp0 = _mm_unpacklo_ps(tmp0, tmp1);
            __m256 res = real;
            res = _mm256_insertf128_ps(res, tmp0, 0);
            res = _mm256_insertf128_ps(res, tmp2, 1);
            return res;
        }
        template <int index, class B>
        inline B get_half_complex_d(const B& real, const B& imag)
        {
            __m128d tmp0 = _mm256_extractf128_pd(real, index);
            __m128d tmp1 = _mm256_extractf128_pd(imag, index);
            __m128d tmp2 = _mm_unpackhi_pd(tmp0, tmp1);
            tmp0 = _mm_unpacklo_pd(tmp0, tmp1);
            __m256d res = real;
            res = _mm256_insertf128_pd(res, tmp0, 0);
            res = _mm256_insertf128_pd(res, tmp2, 1);
            return res;
        }
    }
    template<class A> batch<float, A> complex_low(batch<std::complex<float>, A> const& self, requires<avx>) {
            return detail::get_half_complex_f<0>(self.real(), self.imag());
    }
    template<class A> batch<double, A> complex_low(batch<std::complex<double>, A> const& self, requires<avx>) {
            return detail::get_half_complex_d<0>(self.real(), self.imag());
    }

    // complex_high
    template<class A> batch<float, A> complex_high(batch<std::complex<float>, A> const& self, requires<avx>) {
            return detail::get_half_complex_f<1>(self.real(), self.imag());
    }
    template<class A> batch<double, A> complex_high(batch<std::complex<double>, A> const& self, requires<avx>) {
            return detail::get_half_complex_d<1>(self.real(), self.imag());
    }
    // convert
    namespace detail {
    template<class A> batch<float, A> fast_cast(batch<int32_t, A> const& self, batch<float, A> const&, requires<avx>) {
      return _mm256_cvtepi32_ps(self);
    }
    template<class A> batch<int32_t, A> fast_cast(batch<float, A> const& self, batch<int32_t, A> const&, requires<avx>) {
      return _mm256_cvttps_epi32(self);
    }
    }

    // div
    template<class A> batch<float, A> div(batch<float, A> const& self, batch<float, A> const& other, requires<avx>) {
      return _mm256_div_ps(self, other);
    }
    template<class A> batch<double, A> div(batch<double, A> const& self, batch<double, A> const& other, requires<avx>) {
      return _mm256_div_pd(self, other);
    }

    // eq
    template<class A> batch_bool<float, A> eq(batch<float, A> const& self, batch<float, A> const& other, requires<avx>) {
      return _mm256_cmp_ps(self, other, _CMP_EQ_OQ);
    }
    template<class A> batch_bool<double, A> eq(batch<double, A> const& self, batch<double, A> const& other, requires<avx>) {
      return _mm256_cmp_pd(self, other, _CMP_EQ_OQ);
    }
    template<class A> batch_bool<float, A> eq(batch_bool<float, A> const& self, batch_bool<float, A> const& other, requires<avx>) {
      return  _mm256_castsi256_ps(detail::fwd_to_sse([](__m128i s, __m128i o) { return eq(batch_bool<int32_t, sse4_2>(s), batch_bool<int32_t, sse4_2>(o), sse4_2{}); },
                                  _mm256_castps_si256(self), _mm256_castps_si256(other)));
    }
    template<class A> batch_bool<double, A> eq(batch_bool<double, A> const& self, batch_bool<double, A> const& other, requires<avx>) {
      return  _mm256_castsi256_pd(detail::fwd_to_sse([](__m128i s, __m128i o) { return eq(batch_bool<int32_t, sse4_2>(s), batch_bool<int32_t, sse4_2>(o), sse4_2{}); }, _mm256_castpd_si256(self), _mm256_castpd_si256(other)));
    }
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch_bool<T, A> eq(batch<T, A> const& self, batch<T, A> const& other, requires<avx>) {
      switch(sizeof(T)) {
        case 1: return _mm256_cmpeq_epi8(self, other);
        case 2: return _mm256_cmpeq_epi16(self, other);
        case 4: return detail::fwd_to_sse([](__m128i s, __m128i o) { return eq(batch<T, sse4_2>(s), batch<T, sse4_2>(o), sse4_2{}); },self, other);
        case 8: {
            __m256i tmp1 = detail::fwd_to_sse([](__m128i s, __m128i o) { return eq(batch<T, sse4_2>(s), batch<T, sse4_2>(o)); },self, other);
            __m256i tmp2 = _mm256_shuffle_epi32(tmp1, 0xB1);
            __m256i tmp3 = detail::fwd_to_sse([](__m128i s, __m128i o) { return bitwise_and(batch<T, sse4_2>(s), batch<T, sse4_2>(o)); }, tmp1, tmp2);
            __m256i tmp4 = _mm256_srai_epi32(tmp3, 31);
            return _mm256_shuffle_epi32(tmp4, 0xF5);
        }
        default: assert(false && "unsupported arch/op combination"); return {};
      }
    }
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch_bool<T, A> eq(batch_bool<T, A> const& self, batch_bool<T, A> const& other, requires<avx>) {
      return eq(batch<T, A>(self.data), batch<T, A>(other.data));
    }

    // floor
    template<class A> batch<float, A> floor(batch<float, A> const& self, requires<avx>) {
      return _mm256_floor_ps(self);
    }
    template<class A> batch<double, A> floor(batch<double, A> const& self, requires<avx>) {
      return _mm256_floor_pd(self);
    }

    // ge
    template<class A> batch_bool<float, A> ge(batch<float, A> const& self, batch<float, A> const& other, requires<avx>) {
      return _mm256_cmp_ps(self, other, _CMP_GE_OQ);
    }
    template<class A> batch_bool<double, A> ge(batch<double, A> const& self, batch<double, A> const& other, requires<avx>) {
      return _mm256_cmp_pd(self, other, _CMP_GE_OQ);
    }
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch_bool<T, A> ge(batch<T, A> const& self, batch<T, A> const& other, requires<avx>) {
      return detail::fwd_to_sse([](__m128i s, __m128i o) { return ge(batch<T, sse4_2>(s), batch<T, sse4_2>(o)); }, self, other);
    }

    // gt
    template<class A> batch_bool<float, A> gt(batch<float, A> const& self, batch<float, A> const& other, requires<avx>) {
      return _mm256_cmp_ps(self, other, _CMP_GT_OQ);
    }
    template<class A> batch_bool<double, A> gt(batch<double, A> const& self, batch<double, A> const& other, requires<avx>) {
      return _mm256_cmp_pd(self, other, _CMP_GT_OQ);
    }
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch_bool<T, A> gt(batch<T, A> const& self, batch<T, A> const& other, requires<avx>) {
      return detail::fwd_to_sse([](__m128i s, __m128i o) { return gt(batch<T, sse4_2>(s), batch<T, sse4_2>(o)); }, self, other);
    }


    // hadd
    template<class A> float hadd(batch<float, A> const& rhs, requires<avx>) {
                // Warning about _mm256_hadd_ps:
                // _mm256_hadd_ps(a,b) gives
                // (a0+a1,a2+a3,b0+b1,b2+b3,a4+a5,a6+a7,b4+b5,b6+b7). Hence we can't
                // rely on a naive use of this method
                // rhs = (x0, x1, x2, x3, x4, x5, x6, x7)
                // tmp = (x4, x5, x6, x7, x0, x1, x2, x3)
                __m256 tmp = _mm256_permute2f128_ps(rhs, rhs, 1);
                // tmp = (x4+x0, x5+x1, x6+x2, x7+x3, x0+x4, x1+x5, x2+x6, x3+x7)
                tmp = _mm256_add_ps(rhs, tmp);
                // tmp = (x4+x0+x5+x1, x6+x2+x7+x3, -, -, -, -, -, -)
                tmp = _mm256_hadd_ps(tmp, tmp);
                // tmp = (x4+x0+x5+x1+x6+x2+x7+x3, -, -, -, -, -, -, -)
                tmp = _mm256_hadd_ps(tmp, tmp);
                return _mm_cvtss_f32(_mm256_extractf128_ps(tmp, 0));

    }
    template <class A>
    double hadd(batch<double, A> const &rhs, requires<avx>) {
                // rhs = (x0, x1, x2, x3)
                // tmp = (x2, x3, x0, x1)
                __m256d tmp = _mm256_permute2f128_pd(rhs, rhs, 1);
                // tmp = (x2+x0, x3+x1, -, -)
                tmp = _mm256_add_pd(rhs, tmp);
                // tmp = (x2+x0+x3+x1, -, -, -)
                tmp = _mm256_hadd_pd(tmp, tmp);
                return _mm_cvtsd_f64(_mm256_extractf128_pd(tmp, 0));
    }
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    T hadd(batch<T, A> const& self, requires<avx>) {
      __m128i low, high;
      detail::split_avx(self, low, high);
      batch<T, sse4_2> blow(low), bhigh(high);
      return hadd(blow) + hadd(bhigh);
    }

    // haddp
    template<class A> batch<float, A> haddp(batch<float, A> const* row, requires<avx>) {
                // row = (a,b,c,d,e,f,g,h)
                // tmp0 = (a0+a1, a2+a3, b0+b1, b2+b3, a4+a5, a6+a7, b4+b5, b6+b7)
                __m256 tmp0 = _mm256_hadd_ps(row[0], row[1]);
                // tmp1 = (c0+c1, c2+c3, d1+d2, d2+d3, c4+c5, c6+c7, d4+d5, d6+d7)
                __m256 tmp1 = _mm256_hadd_ps(row[2], row[3]);
                // tmp1 = (a0+a1+a2+a3, b0+b1+b2+b3, c0+c1+c2+c3, d0+d1+d2+d3,
                // a4+a5+a6+a7, b4+b5+b6+b7, c4+c5+c6+c7, d4+d5+d6+d7)
                tmp1 = _mm256_hadd_ps(tmp0, tmp1);
                // tmp0 = (e0+e1, e2+e3, f0+f1, f2+f3, e4+e5, e6+e7, f4+f5, f6+f7)
                tmp0 = _mm256_hadd_ps(row[4], row[5]);
                // tmp2 = (g0+g1, g2+g3, h0+h1, h2+h3, g4+g5, g6+g7, h4+h5, h6+h7)
                __m256 tmp2 = _mm256_hadd_ps(row[6], row[7]);
                // tmp2 = (e0+e1+e2+e3, f0+f1+f2+f3, g0+g1+g2+g3, h0+h1+h2+h3,
                // e4+e5+e6+e7, f4+f5+f6+f7, g4+g5+g6+g7, h4+h5+h6+h7)
                tmp2 = _mm256_hadd_ps(tmp0, tmp2);
                // tmp0 = (a0+a1+a2+a3, b0+b1+b2+b3, c0+c1+c2+c3, d0+d1+d2+d3,
                // e4+e5+e6+e7, f4+f5+f6+f7, g4+g5+g6+g7, h4+h5+h6+h7)
                tmp0 = _mm256_blend_ps(tmp1, tmp2, 0b11110000);
                // tmp1 = (a4+a5+a6+a7, b4+b5+b6+b7, c4+c5+c6+c7, d4+d5+d6+d7,
                // e0+e1+e2+e3, f0+f1+f2+f3, g0+g1+g2+g3, h0+h1+h2+h3)
                tmp1 = _mm256_permute2f128_ps(tmp1, tmp2, 0x21);
                return _mm256_add_ps(tmp0, tmp1);
    }
    template <class A>
    batch<double, A> haddp(batch<double, A> const *row, requires<avx>) {
                // row = (a,b,c,d)
                // tmp0 = (a0+a1, b0+b1, a2+a3, b2+b3)
                __m256d tmp0 = _mm256_hadd_pd(row[0], row[1]);
                // tmp1 = (c0+c1, d0+d1, c2+c3, d2+d3)
                __m256d tmp1 = _mm256_hadd_pd(row[2], row[3]);
                // tmp2 = (a0+a1, b0+b1, c2+c3, d2+d3)
                __m256d tmp2 = _mm256_blend_pd(tmp0, tmp1, 0b1100);
                // tmp1 = (a2+a3, b2+b3, c2+c3, d2+d3)
                tmp1 = _mm256_permute2f128_pd(tmp0, tmp1, 0x21);
                return _mm256_add_pd(tmp1, tmp2);
    }

    // isnan
    template<class A> batch_bool<float, A> isnan(batch<float, A> const& self, requires<avx>) {
                return _mm256_cmp_ps(self, self, _CMP_UNORD_Q);
    }
    template<class A> batch_bool<double, A> isnan(batch<double, A> const& self, requires<avx>) {
                return _mm256_cmp_pd(self, self, _CMP_UNORD_Q);
    }

    // le
    template<class A> batch_bool<float, A> le(batch<float, A> const& self, batch<float, A> const& other, requires<avx>) {
      return _mm256_cmp_ps(self, other, _CMP_LE_OQ);
    }
    template<class A> batch_bool<double, A> le(batch<double, A> const& self, batch<double, A> const& other, requires<avx>) {
      return _mm256_cmp_pd(self, other, _CMP_LE_OQ);
    }

    // load_aligned
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> load_aligned(T const* mem, convert<T>, requires<avx>) {
      return _mm256_load_si256((__m256i const*)mem);
    }
    template<class A> batch<float, A> load_aligned(float const* mem, convert<float>, requires<avx>) {
      return _mm256_load_ps(mem);
    }
    template<class A> batch<double, A> load_aligned(double const* mem, convert<double>, requires<avx>) {
      return _mm256_load_pd(mem);
    }

    // load_complex
    template<class A> batch<std::complex<float>, A> load_complex(batch<float, A> const& hi, batch<float, A> const& lo, requires<avx>) {
            using batch_type = batch<float, A>;
            __m128 tmp0 = _mm256_extractf128_ps(hi, 0);
            __m128 tmp1 = _mm256_extractf128_ps(hi, 1);
            batch_type real, imag;
            __m128 tmp_real = _mm_shuffle_ps(tmp0, tmp1, _MM_SHUFFLE(2, 0, 2, 0));
            __m128 tmp_imag = _mm_shuffle_ps(tmp0, tmp1, _MM_SHUFFLE(3, 1, 3, 1));
            real = _mm256_insertf128_ps(real, tmp_real, 0);
            imag = _mm256_insertf128_ps(imag, tmp_imag, 0);

            tmp0 = _mm256_extractf128_ps(lo, 0);
            tmp1 = _mm256_extractf128_ps(lo, 1);
            tmp_real = _mm_shuffle_ps(tmp0, tmp1, _MM_SHUFFLE(2, 0, 2, 0));
            tmp_imag = _mm_shuffle_ps(tmp0, tmp1, _MM_SHUFFLE(3, 1, 3, 1));
            real = _mm256_insertf128_ps(real, tmp_real, 1);
            imag = _mm256_insertf128_ps(imag, tmp_imag, 1);
            return {real, imag};
    }
    template<class A> batch<std::complex<double>, A> load_complex(batch<double,A> const& hi, batch<double,A> const& lo, requires<avx>) {
            using batch_type = batch<double, A>;
            __m128d tmp0 = _mm256_extractf128_pd(hi, 0);
            __m128d tmp1 = _mm256_extractf128_pd(hi, 1);
            batch_type real, imag;
            __m256d re_tmp0 = _mm256_insertf128_pd(real, _mm_unpacklo_pd(tmp0, tmp1), 0);
            __m256d im_tmp0 = _mm256_insertf128_pd(imag, _mm_unpackhi_pd(tmp0, tmp1), 0);
            tmp0 = _mm256_extractf128_pd(lo, 0);
            tmp1 = _mm256_extractf128_pd(lo, 1);
            __m256d re_tmp1 = _mm256_insertf128_pd(real, _mm_unpacklo_pd(tmp0, tmp1), 1);
            __m256d im_tmp1 = _mm256_insertf128_pd(imag, _mm_unpackhi_pd(tmp0, tmp1), 1);
            real = _mm256_blend_pd(re_tmp0, re_tmp1, 12);
            imag = _mm256_blend_pd(im_tmp0, im_tmp1, 12);
            return {real, imag};
    }

    // load_unaligned
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> load_unaligned(T const* mem, convert<T>, requires<avx>) {
      return _mm256_loadu_si256((__m256i const*)mem);
    }
    template<class A> batch<float, A> load_unaligned(float const* mem, convert<float>, requires<avx>){
      return _mm256_loadu_ps(mem);
    }
    template<class A> batch<double, A> load_unaligned(double const* mem, convert<double>, requires<avx>){
      return _mm256_loadu_pd(mem);
    }

    // lt
    template<class A> batch_bool<float, A> lt(batch<float, A> const& self, batch<float, A> const& other, requires<avx>) {
      return _mm256_cmp_ps(self, other, _CMP_LT_OQ);
    }
    template<class A> batch_bool<double, A> lt(batch<double, A> const& self, batch<double, A> const& other, requires<avx>) {
      return _mm256_cmp_pd(self, other, _CMP_LT_OQ);
    }

    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch_bool<T, A> lt(batch<T, A> const& self, batch<T, A> const& other, requires<avx>) {
      return detail::fwd_to_sse([](__m128i s, __m128i o) { return lt(batch<T, sse4_2>(s), batch<T, sse4_2>(o)); }, self, other);
    }

    // max
    template<class A> batch<float, A> max(batch<float, A> const& self, batch<float, A> const& other, requires<avx>) {
      return _mm256_max_ps(self, other);
    }
    template<class A> batch<double, A> max(batch<double, A> const& self, batch<double, A> const& other, requires<avx>) {
      return _mm256_max_pd(self, other);
    }
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> max(batch<T, A> const& self, batch<T, A> const& other, requires<avx>) {
      return select(self > other, self, other);
    }

    // min
    template<class A> batch<float, A> min(batch<float, A> const& self, batch<float, A> const& other, requires<avx>) {
      return _mm256_min_ps(self, other);
    }
    template<class A> batch<double, A> min(batch<double, A> const& self, batch<double, A> const& other, requires<avx>) {
      return _mm256_min_pd(self, other);
    }
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> min(batch<T, A> const& self, batch<T, A> const& other, requires<avx>) {
      return select(self <= other, self, other);
    }

    // mul
    template<class A> batch<float, A> mul(batch<float, A> const& self, batch<float, A> const& other, requires<avx>) {
      return _mm256_mul_ps(self, other);
    }
    template<class A> batch<double, A> mul(batch<double, A> const& self, batch<double, A> const& other, requires<avx>) {
      return _mm256_mul_pd(self, other);
    }

    // nearbyint
    template<class A> batch<float, A> nearbyint(batch<float, A> const& self, requires<avx>) {
      return _mm256_round_ps(self, _MM_FROUND_TO_NEAREST_INT);
    }
    template<class A> batch<double, A> nearbyint(batch<double, A> const& self, requires<avx>) {
      return _mm256_round_pd(self, _MM_FROUND_TO_NEAREST_INT);
    }

    // neg
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> neg(batch<T, A> const& self, requires<avx>) {
      return 0 - self;
    }
    template<class A> batch<float, A> neg(batch<float, A> const& self, requires<avx>) {
      return _mm256_xor_ps(self, _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000)));
    }
    template <class A>
    batch<double, A> neg(batch<double, A> const &self, requires<avx>) {
      return _mm256_xor_pd(self, _mm256_castsi256_pd(_mm256_set1_epi64x(0x8000000000000000)));
    }

    // neq
    template<class A> batch_bool<float, A> neq(batch<float, A> const& self, batch<float, A> const& other, requires<avx>) {
      return _mm256_cmp_ps(self, other, _CMP_NEQ_OQ);
    }
    template<class A> batch_bool<double, A> neq(batch<double, A> const& self, batch<double, A> const& other, requires<avx>) {
      return _mm256_cmp_pd(self, other, _CMP_NEQ_OQ);
    }
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch_bool<T, A> neq(batch<T, A> const& self, batch<T, A> const& other, requires<avx>) {
        return ~(self == other);
    }


    template<class A> batch_bool<float, A> neq(batch_bool<float, A> const& self, batch_bool<float, A> const& other, requires<avx>) {
      return _mm256_xor_ps(self, other);
    }
    template<class A> batch_bool<double, A> neq(batch_bool<double, A> const& self, batch_bool<double, A> const& other, requires<avx>) {
      return _mm256_xor_pd(self, other);
    }
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch_bool<T, A> neq(batch_bool<T, A> const& self, batch_bool<T, A> const& other, requires<avx>) {
        return ~(self == other);
    }

    // sadd
    template<class A> batch<float, A> sadd(batch<float, A> const& self, batch<float, A> const& other, requires<avx>) {
      return add(self, other); // no saturated arithmetic on floating point numbers
    }
    template<class A> batch<double, A> sadd(batch<double, A> const& self, batch<double, A> const& other, requires<avx>) {
      return add(self, other); // no saturated arithmetic on floating point numbers
    }
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> sadd(batch<T, A> const& self, batch<T, A> const& other, requires<avx>) {
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

    // select
    template<class A> batch<float, A> select(batch_bool<float, A> const& cond, batch<float, A> const& true_br, batch<float, A> const& false_br, requires<avx>) {
                return _mm256_blendv_ps(false_br, true_br, cond);
    }
    template<class A> batch<double, A> select(batch_bool<double, A> const& cond, batch<double, A> const& true_br, batch<double, A> const& false_br, requires<avx>) {
                return _mm256_blendv_pd(false_br, true_br, cond);
    }
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> select(batch_bool<T, A> const& cond, batch<T, A> const& true_br, batch<T, A> const& false_br, requires<avx>) {
      __m128i cond_low, cond_hi;
      detail::split_avx(cond, cond_low, cond_hi);

      __m128i true_low, true_hi;
      detail::split_avx(true_br, true_low, true_hi);

      __m128i false_low, false_hi;
      detail::split_avx(false_br, false_low, false_hi);

      __m128i res_low = select(batch_bool<T, sse4_2>(cond_low), batch<T, sse4_2>(true_low), batch<T, sse4_2>(false_low), sse4_2{});
      __m128i res_hi = select(batch_bool<T, sse4_2>(cond_hi), batch<T, sse4_2>(true_hi), batch<T, sse4_2>(false_hi), sse4_2{});
      return detail::merge_sse(res_low, res_hi);
    }
    template<class A, class T, bool... Values, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> select(batch_bool_constant<batch<T, A>, Values...> const&, batch<T, A> const& true_br, batch<T, A> const& false_br, requires<avx>) {
      return select(batch_bool<T, A>{Values...}, true_br, false_br, avx2{});
    }


    // set
    template<class A, class... Values>
    batch<float, A> set(batch<float, A> const&, requires<avx>, Values... values) {
      static_assert(sizeof...(Values) == batch<float, A>::size, "consistent init");
      return _mm256_setr_ps(values...);
    }

    template<class A, class... Values>
    batch<double, A> set(batch<double, A> const&, requires<avx>, Values... values) {
      static_assert(sizeof...(Values) == batch<double, A>::size, "consistent init");
      return _mm256_setr_pd(values...);
    }
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> set(batch<T, A> const&, requires<avx>, T v0, T v1, T v2, T v3) {
      return _mm256_set_epi64x(v3, v2, v1, v0);
    }
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> set(batch<T, A> const&, requires<avx>, T v0, T v1, T v2, T v3, T v4, T v5, T v6, T v7) {
      return _mm256_setr_epi32(v0, v1, v2, v3, v4, v5, v6, v7);
    }
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> set(batch<T, A> const&, requires<avx>, T v0, T v1, T v2, T v3, T v4, T v5, T v6, T v7, T v8, T v9, T v10, T v11, T v12, T v13, T v14, T v15) {
      return _mm256_setr_epi16(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15);
    }
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> set(batch<T, A> const&, requires<avx>, T v0, T v1, T v2, T v3, T v4, T v5, T v6, T v7, T v8, T v9, T v10, T v11, T v12, T v13, T v14, T v15,
      T v16, T v17, T v18, T v19, T v20, T v21, T v22, T v23, T v24, T v25, T v26, T v27, T v28, T v29, T v30, T v31  ) {
      return _mm256_setr_epi8(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31);
    }

    template<class A, class T, class... Values, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch_bool<T, A> set(batch_bool<T, A> const&, requires<avx>, Values... values) {
      return set(batch<T, A>(), A{}, static_cast<T>(values ? -1LL : 0LL )...).data;
    }

    template<class A, class... Values>
    batch_bool<float, A> set(batch_bool<float, A> const&, requires<avx>, Values... values) {
      static_assert(sizeof...(Values) == batch_bool<float, A>::size, "consistent init");
      return _mm256_castsi256_ps(set(batch<int32_t, A>(), A{}, static_cast<int32_t>(values ? -1LL : 0LL )...).data);
    }

    template<class A, class... Values>
    batch_bool<double, A> set(batch_bool<double, A> const&, requires<avx>, Values... values) {
      static_assert(sizeof...(Values) == batch_bool<double, A>::size, "consistent init");
      return _mm256_castsi256_pd(set(batch<int64_t, A>(), A{},  static_cast<int64_t>(values ? -1LL : 0LL )...).data);
    }

    // sqrt
    template<class A> batch<float, A> sqrt(batch<float, A> const& val, requires<avx>) {
      return _mm256_sqrt_ps(val);
    }
    template<class A> batch<double, A> sqrt(batch<double, A> const& val, requires<avx>) {
      return _mm256_sqrt_pd(val);
    }

    // ssub
    template<class A> batch<float, A> ssub(batch<float, A> const& self, batch<float, A> const& other, requires<avx>) {
      return _mm256_sub_ps(self, other); // no saturated arithmetic on floating point numbers
    }
    template<class A> batch<double, A> ssub(batch<double, A> const& self, batch<double, A> const& other, requires<avx>) {
      return _mm256_sub_pd(self, other); // no saturated arithmetic on floating point numbers
    }
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> ssub(batch<T, A> const& self, batch<T, A> const& other, requires<avx>) {
      if(std::is_signed<T>::value) {
         return sadd(self, -other);
      }
      else {
        const auto diff = min(self, other);
        return self - diff;
      }
    }

    // store_aligned
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    void store_aligned(T *mem, batch<T, A> const& self, requires<avx>) {
      return _mm256_store_si256((__m256i *)mem, self);
    }
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    void store_aligned(T *mem, batch_bool<T, A> const& self, requires<avx>) {
      return _mm256_store_si256((__m256i *)mem, self);
    }
    template<class A> void store_aligned(float *mem, batch<float, A> const& self, requires<avx>) {
      return _mm256_store_ps(mem, self);
    }
    template<class A> void store_aligned(double *mem, batch<double, A> const& self, requires<avx>) {
      return _mm256_store_pd(mem, self);
    }

    // store_unaligned
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    void store_unaligned(T *mem, batch<T, A> const& self, requires<avx>) {
      return _mm256_storeu_si256((__m256i *)mem, self);
    }
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    void store_unaligned(T *mem, batch_bool<T, A> const& self, requires<avx>) {
      return _mm256_storeu_si256((__m256i *)mem, self);
    }
    template<class A> void store_unaligned(float *mem, batch<float, A> const& self, requires<avx>) {
      return _mm256_storeu_ps(mem, self);
    }
    template<class A> void store_unaligned(double *mem, batch<double, A> const& self, requires<avx>) {
      return _mm256_storeu_pd(mem, self);
    }

    // sub
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> sub(batch<T, A> const& self, batch<T, A> const& other, requires<avx>) {
      switch(sizeof(T)) {
        case 1: return _mm256_sub_epi8(self, other);
        case 2: return _mm256_sub_epi16(self, other);
        case 4: return detail::fwd_to_sse([](__m128i s, __m128i o) { return sub(batch<T, sse4_2>(s), batch<T, sse4_2>(o)); }, self, other);
        case 8: return _mm256_sub_epi64(self, other);
        default: assert(false && "unsupported arch/op combination"); return {};
      }
    }
    template<class A> batch<float, A> sub(batch<float, A> const& self, batch<float, A> const& other, requires<avx>) {
      return _mm256_sub_ps(self, other);
    }
    template<class A> batch<double, A> sub(batch<double, A> const& self, batch<double, A> const& other, requires<avx>) {
      return _mm256_sub_pd(self, other);
    }

    // to_float
    template<class A>
    batch<float, A> to_float(batch<int32_t, A> const& self, requires<avx>) {
      return _mm256_cvtepi32_ps(self);
    }
    template<class A>
    batch<double, A> to_float(batch<int64_t, A> const& self, requires<avx>) {
      // FIXME: call _mm_cvtepi64_pd
      alignas(A::alignment()) int64_t buffer[batch<int64_t, A>::size];
      self.store_aligned(&buffer[0]);
      return {(double)buffer[0], (double)buffer[1], (double)buffer[2], (double)buffer[3],};
    }

    // to_int
    template<class A>
    batch<int32_t, A> to_int(batch<float, A> const& self, requires<avx>) {
      return _mm256_cvttps_epi32(self);
    }

    template<class A>
    batch<int64_t, A> to_int(batch<double, A> const& self, requires<avx>) {
      // FIXME: call _mm_cvttpd_epi64
      alignas(A::alignment()) double buffer[batch<double, A>::size];
      self.store_aligned(&buffer[0]);
      return {(int64_t)buffer[0], (int64_t)buffer[1], (int64_t)buffer[2], (int64_t)buffer[3]};
    }

    // trunc
    template<class A> batch<float, A> trunc(batch<float, A> const& self, requires<avx>) {
      return _mm256_round_ps(self, _MM_FROUND_TO_ZERO);
    }
    template<class A> batch<double, A> trunc(batch<double, A> const& self, requires<avx>) {
      return _mm256_round_pd(self, _MM_FROUND_TO_ZERO);
    }

    // zip_hi
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> zip_hi(batch<T, A> const& self, batch<T, A> const& other, requires<avx>) {
      switch(sizeof(T)) {
        case 1: return _mm256_unpackhi_epi8(self, other);
        case 2: return _mm256_unpackhi_epi16(self, other);
        case 4: return _mm256_unpackhi_epi32(self, other);
        case 8: return _mm256_unpackhi_epi64(self, other);
        default: assert(false && "unsupported arch/op combination"); return {};
      }
    }
    template<class A> batch<float, A> zip_hi(batch<float, A> const& self, batch<float, A> const& other, requires<avx>) {
      return _mm256_unpackhi_ps(self, other);
    }
    template<class A> batch<double, A> zip_hi(batch<double, A> const& self, batch<double, A> const& other, requires<avx>) {
      return _mm256_unpackhi_pd(self, other);
    }

    // zip_lo
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> zip_lo(batch<T, A> const& self, batch<T, A> const& other, requires<avx>) {
      switch(sizeof(T)) {
        case 1: return _mm256_unpacklo_epi8(self, other);
        case 2: return _mm256_unpacklo_epi16(self, other);
        case 4: return _mm256_unpacklo_epi32(self, other);
        case 8: return _mm256_unpacklo_epi64(self, other);
        default: assert(false && "unsupported arch/op combination"); return {};
      }
    }
    template<class A> batch<float, A> zip_lo(batch<float, A> const& self, batch<float, A> const& other, requires<avx>) {
      return _mm256_unpacklo_ps(self, other);
    }
    template<class A> batch<double, A> zip_lo(batch<double, A> const& self, batch<double, A> const& other, requires<avx>) {
      return _mm256_unpacklo_pd(self, other);
    }

  }

}

#endif
