#ifndef XSIMD_AVX2_HPP
#define XSIMD_AVX2_HPP

#include "../types/xsimd_avx2_register.hpp"


namespace xsimd {

  namespace kernel {
    using namespace types;

    // abs
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> abs(batch<T, A> const& self, requires<avx2>) {
      if(std::is_signed<T>::value) {
        switch(sizeof(T)) {
          case 1: return _mm256_abs_epi8(self);
          case 2: return _mm256_abs_epi16(self);
          case 4: return _mm256_abs_epi32(self);
          default: return abs(self, avx{});
        }
      }
      return self;
    }

    // add
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> add(batch<T, A> const& self, batch<T, A> const& other, requires<avx2>) {
      switch(sizeof(T)) {
        case 1: return _mm256_add_epi8(self, other);
        case 2: return _mm256_add_epi16(self, other);
        case 4: return _mm256_add_epi32(self, other);
        case 8: return _mm256_add_epi64(self, other);
        default: return add(self, other, avx{});
      }
    }

    // bitwise_and
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> bitwise_and(batch<T, A> const& self, batch<T, A> const& other, requires<avx2>) {
      return _mm256_and_si256(self, other);
    }
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch_bool<T, A> bitwise_and(batch_bool<T, A> const& self, batch_bool<T, A> const& other, requires<avx2>) {
      return _mm256_and_si256(self, other);
    }

    // bitwise_lshift
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> bitwise_lshift(batch<T, A> const& self, int32_t other, requires<avx2>) {
      switch(sizeof(T)) {
        case 2: return _mm256_slli_epi16(self, other);
        case 4: return _mm256_slli_epi32(self, other);
        case 8: return _mm256_slli_epi64(self, other);
        default: return bitwise_lshift(self, other, avx{});
      }
    }

    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> bitwise_lshift(batch<T, A> const& self, batch<T, A> const& other, requires<avx2>) {
      switch(sizeof(T)) {
        case 4: return _mm256_sllv_epi32(self, other);
        case 8: return _mm256_sllv_epi64(self, other);
        default: return bitwise_lshift(self, other, avx{});
      }
    }

    // bitwise_rshift
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> bitwise_rshift(batch<T, A> const& self, int32_t other, requires<avx2>) {
      if(std::is_signed<T>::value) {
        switch(sizeof(T)) {
          case 2: return _mm256_srai_epi16(self, other);
          case 4: return _mm256_srai_epi32(self, other);
          default: return bitwise_rshift(self, other, avx{});
        }
      }
      else {
        switch(sizeof(T)) {
          case 2: return _mm256_srli_epi16(self, other);
          case 4: return _mm256_srli_epi32(self, other);
          case 8: return _mm256_srli_epi64(self, other);
          default: return bitwise_rshift(self, other, avx{});
        }
      }
    }

    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> bitwise_rshift(batch<T, A> const& self, batch<T, A> const& other, requires<avx2>) {
      if(std::is_signed<T>::value) {
        switch(sizeof(T)) {
          case 4: return _mm256_srav_epi32(self, other);
          default: return bitwise_rshift(self, other, avx{});
        }
      }
      else {
        switch(sizeof(T)) {
          case 4: return _mm256_srlv_epi32(self, other);
          case 8: return _mm256_srlv_epi64(self, other);
          default: return bitwise_rshift(self, other, avx{});
        }
      }
    }

    // complex_low
    template<class A> batch<double, A> complex_low(batch<std::complex<double>, A> const& self, requires<avx2>) {
            __m256d tmp0 = _mm256_permute4x64_pd(self.real(), _MM_SHUFFLE(3, 1, 1, 0));
            __m256d tmp1 = _mm256_permute4x64_pd(self.imag(), _MM_SHUFFLE(1, 2, 0, 0));
            return _mm256_blend_pd(tmp0, tmp1, 10);
    }

    // complex_high
    template<class A> batch<double, A> complex_high(batch<std::complex<double>, A> const& self, requires<avx2>) {
            __m256d tmp0 = _mm256_permute4x64_pd(self.real(), _MM_SHUFFLE(3, 3, 1, 2));
            __m256d tmp1 = _mm256_permute4x64_pd(self.imag(), _MM_SHUFFLE(3, 2, 2, 0));
            return _mm256_blend_pd(tmp0, tmp1, 10);
    }

    // eq
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch_bool<T, A> eq(batch<T, A> const& self, batch<T, A> const& other, requires<avx2>) {
      switch(sizeof(T)) {
        case 1: return _mm256_cmpeq_epi8(self, other);
        case 2: return _mm256_cmpeq_epi16(self, other);
        case 4: return _mm256_cmpeq_epi32(self, other);
        case 8: return _mm256_cmpeq_epi64(self, other);
        default: eq(self, other, avx{});
      }
    }

    // gt
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch_bool<T, A> gt(batch<T, A> const& self, batch<T, A> const& other, requires<avx2>) {
      if(std::is_signed<T>::value) {
        switch(sizeof(T)) {
          case 1: return _mm256_cmpgt_epi8(self, other);
          case 2: return _mm256_cmpgt_epi16(self, other);
          case 4: return _mm256_cmpgt_epi32(self, other);
          case 8: return _mm256_cmpgt_epi64(self, other);
          default: return gt(self, other, avx{});
        }
      }
      else {
          return gt(self, other, avx{});
      }
    }

    // hadd
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    T hadd(batch<T, A> const& self, requires<avx2>) {
      switch(sizeof(T)) {
        case 4:
          {
                __m256i tmp1 = _mm256_hadd_epi32(self, self);
                __m256i tmp2 = _mm256_hadd_epi32(tmp1, tmp1);
                __m128i tmp3 = _mm256_extracti128_si256(tmp2, 1);
                __m128i tmp4 = _mm_add_epi32(_mm256_castsi256_si128(tmp2), tmp3);
                return _mm_cvtsi128_si32(tmp4);
          }
        case 8:
          {
                __m256i tmp1 = _mm256_shuffle_epi32(self, 0x0E);
                __m256i tmp2 = _mm256_add_epi64(self, tmp1);
                __m128i tmp3 = _mm256_extracti128_si256(tmp2, 1);
                __m128i res = _mm_add_epi64(_mm256_castsi256_si128(tmp2), tmp3);
                return _mm_cvtsi128_si64(res);
          }
          default: return hadd(self, avx{});
      }
    }
    // load_complex
    template<class A> batch<std::complex<float>, A> load_complex(batch<float, A> const& hi, batch<float, A> const& lo, requires<avx2>) {
            using batch_type = batch<float, A>;
            batch_type real = _mm256_castpd_ps(
                         _mm256_permute4x64_pd(
                             _mm256_castps_pd(_mm256_shuffle_ps(hi, lo, _MM_SHUFFLE(2, 0, 2, 0))),
                             _MM_SHUFFLE(3, 1, 2, 0)));
            batch_type imag = _mm256_castpd_ps(
                         _mm256_permute4x64_pd(
                             _mm256_castps_pd(_mm256_shuffle_ps(hi, lo, _MM_SHUFFLE(3, 1, 3, 1))),
                             _MM_SHUFFLE(3, 1, 2, 0)));
            return {real, imag};
    }
    template<class A> batch<std::complex<double>, A> load_complex(batch<double,A> const& hi, batch<double,A> const& lo, requires<avx2>) {
            using batch_type = batch<double, A>;
            batch_type real = _mm256_permute4x64_pd(_mm256_unpacklo_pd(hi, lo), _MM_SHUFFLE(3, 1, 2, 0));
            batch_type imag = _mm256_permute4x64_pd(_mm256_unpackhi_pd(hi, lo), _MM_SHUFFLE(3, 1, 2, 0));
            return {real, imag};
    }

    // max
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> max(batch<T, A> const& self, batch<T, A> const& other, requires<avx2>) {
      if(std::is_signed<T>::value) {
        switch(sizeof(T)) {
          case 1: return _mm256_max_epi8(self, other);
          case 2: return _mm256_max_epi16(self, other);
          case 4: return _mm256_max_epi32(self, other);
          default: return max(self, other, avx{});
        }
      }
      else {
        switch(sizeof(T)) {
          case 1: return _mm256_max_epu8(self, other);
          case 2: return _mm256_max_epu16(self, other);
          case 4: return _mm256_max_epu32(self, other);
          default: return max(self, other, avx{});
        }
      }
    }

    // min
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> min(batch<T, A> const& self, batch<T, A> const& other, requires<avx2>) {
      if(std::is_signed<T>::value) {
        switch(sizeof(T)) {
          case 1:  return _mm256_min_epi8(self, other);
          case 2:  return _mm256_min_epi16(self, other);
          case 4:  return _mm256_min_epi32(self, other);
          default: return min(self, other, avx{});
        }
      }
      else {
        switch(sizeof(T)) {
          case 1:  return _mm256_min_epu8(self, other);
          case 2:  return _mm256_min_epu16(self, other);
          case 4:  return _mm256_min_epu32(self, other);
          default: return min(self, other, avx{});
        }
      }
    }

    // mul
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> mul(batch<T, A> const& self, batch<T, A> const& other, requires<avx2>) {
      switch(sizeof(T)) {
        case 2: return _mm256_mullo_epi16(self, other);
        case 4: return _mm256_mullo_epi32(self, other);
        default: return mul(self, other, avx{});
      }
    }

    // sadd
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> sadd(batch<T, A> const& self, batch<T, A> const& other, requires<avx2>) {
      if(std::is_signed<T>::value) {
        switch(sizeof(T)) {
          case 1: return _mm256_adds_epi8(self, other);
          case 2: return _mm256_adds_epi16(self, other);
          default: return sadd(self, other, avx{});
        }
      }
      else {
        switch(sizeof(T)) {
          case 1: return _mm256_adds_epu8(self, other);
          case 2: return _mm256_adds_epu16(self, other);
          default: return sadd(self, other, avx{});
        }
      }
    }

    // select
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> select(batch_bool<T, A> const& cond, batch<T, A> const& true_br, batch<T, A> const& false_br, requires<avx2>) {
      switch(sizeof(T)) {
        case 1: return _mm256_blendv_epi8(false_br, true_br, cond);
        case 2: return _mm256_blendv_epi8(false_br, true_br, cond);
        case 4: return _mm256_blendv_epi8(false_br, true_br, cond);
        case 8: return _mm256_blendv_epi8(false_br, true_br, cond);
        default: return select(cond, true_br, false_br, avx{});
      }
    }
    template<class A, class T, bool... Values, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> select(batch_bool_constant<batch<T, A>, Values...> const&, batch<T, A> const& true_br, batch<T, A> const& false_br, requires<avx2>) {
      constexpr int mask = batch_bool_constant<batch<T, A>, Values...>::mask();
      switch(sizeof(T)) {
        // FIXME: for some reason mask here is not considered as an immediate,
        // but it's okay for _mm256_blend_epi32
        //case 2: return _mm256_blend_epi16(false_br, true_br, mask);
        case 4: return _mm256_blend_epi32(false_br, true_br, mask);
        case 8: {
          constexpr int imask = detail::interleave(mask);
          return _mm256_blend_epi32(false_br, true_br, imask);
        }
        default: return select(batch_bool<T, A>{Values...}, true_br, false_br, avx2{});
      }
    }

    // ssub
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> ssub(batch<T, A> const& self, batch<T, A> const& other, requires<avx2>) {
      if(std::is_signed<T>::value) {
        switch(sizeof(T)) {
          case 1: return _mm256_subs_epi8(self, other);
          case 2: return _mm256_subs_epi16(self, other);
          default: return ssub(self, other, avx{});
        }
      }
      else {
        switch(sizeof(T)) {
          case 1: return _mm256_subs_epu8(self, other);
          case 2: return _mm256_subs_epu16(self, other);
          default: return ssub(self, other, avx{});
        }
      }
    }

    // sub
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> sub(batch<T, A> const& self, batch<T, A> const& other, requires<avx2>) {
      switch(sizeof(T)) {
        case 1: return _mm256_sub_epi8(self, other);
        case 2: return _mm256_sub_epi16(self, other);
        case 4: return _mm256_sub_epi32(self, other);
        case 8: return _mm256_sub_epi64(self, other);
        default: return sub(self, other, avx{});
      }
    }



  }

}

#endif
