#ifndef XSIMD_SSE2_HPP
#define XSIMD_SSE2_HPP

#include "../types/xsimd_sse2_register.hpp"

#include <emmintrin.h>


namespace xsimd {

  namespace kernel {
    using namespace types;

    // add
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> add(batch<T, A> const& self, batch<T, A> const& other, requires<sse2>) {
      switch(sizeof(T)) {
        case 1: return _mm_add_epi8(self, other);
        case 2: return _mm_add_epi16(self, other);
        case 4: return _mm_add_epi32(self, other);
        case 8: return _mm_add_epi64(self, other);
        default: assert(false && "unsupported arch/op combination"); return {};
      }
    }
    template<class A> batch<float, A> add(batch<float, A> const& self, batch<float, A> const& other, requires<sse2>) {
      return _mm_add_ps(self, other);
    }
    template<class A> batch<double, A> add(batch<double, A> const& self, batch<double, A> const& other, requires<sse2>) {
      return _mm_add_pd(self, other);
    }

    // all
    template<class A> bool all(batch_bool<double, A> const& self, requires<sse2>) {
      return _mm_movemask_pd(self) == 0x03;
    }
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    bool all(batch_bool<T, A> const& self, requires<sse2>) {
      return _mm_movemask_epi8(self) == 0xFFFF;
    }

    // any
    template<class A> bool any(batch_bool<double, A> const& self, requires<sse2>) {
      return _mm_movemask_pd(self) != 0;
    }
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    bool any(batch_bool<T, A> const& self, requires<sse2>) {
      return _mm_movemask_epi8(self) != 0;
    }

    // bitwise_and
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> bitwise_and(batch<T, A> const& self, batch<T, A> const& other, requires<sse2>) {
      return _mm_and_si128(self, other);
    }
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch_bool<T, A> bitwise_and(batch_bool<T, A> const& self, batch_bool<T, A> const& other, requires<sse2>) {
      return _mm_and_si128(self, other);
    }

    // bitwise_lshift
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> bitwise_lshift(batch<T, A> const& self, int32_t other, requires<sse2>) {
      switch(sizeof(T)) {
        case 1: return _mm_and_si128(_mm_set1_epi8(0xFF << other), _mm_slli_epi32(self, other));
        case 2: return _mm_slli_epi16(self, other);
        case 4: return _mm_slli_epi32(self, other);
        case 8: return _mm_slli_epi64(self, other);
        default: assert(false && "unsupported arch/op combination"); return {};
      }
    }

    // bitwise_not
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> bitwise_not(batch<T, A> const& self, requires<sse2>) {
      return _mm_xor_si128(self, _mm_set1_epi32(-1));
    }
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch_bool<T, A> bitwise_not(batch_bool<T, A> const& self, requires<sse2>) {
      return _mm_xor_si128(self, _mm_set1_epi32(-1));
    }

    // bitwise_or
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> bitwise_or(batch<T, A> const& self, batch<T, A> const& other, requires<sse2>) {
      return _mm_or_si128(self, other);
    }
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch_bool<T, A> bitwise_or(batch_bool<T, A> const& self, batch_bool<T, A> const& other, requires<sse2>) {
      return _mm_or_si128(self, other);
    }

    // bitwise_rshift
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> bitwise_rshift(batch<T, A> const& self, int32_t other, requires<sse2>) {
      if(std::is_signed<T>::value) {
        switch(sizeof(T)) {
          case 1: {
            __m128i sign_mask = _mm_set1_epi16((0xFF00 >> other) & 0x00FF);
            __m128i cmp_is_negative = _mm_cmpgt_epi8(_mm_setzero_si128(), self);
            __m128i res = _mm_srai_epi16(self, other);
            return _mm_or_si128(_mm_and_si128(sign_mask, cmp_is_negative), _mm_andnot_si128(sign_mask, res));
          }
          case 2: return _mm_srai_epi16(self, other);
          case 4: return _mm_srai_epi32(self, other);
          case 8: {
            // from https://github.com/samyvilar/vect/blob/master/vect_128.h
            return _mm_or_si128(
                _mm_srli_epi64(self, other),
                _mm_slli_epi64(
                    _mm_srai_epi32(_mm_shuffle_epi32(self, _MM_SHUFFLE(3, 3, 1, 1)), 32),
                    64 - other));
          }
          default: assert(false && "unsupported arch/op combination"); return {};
        }
      }
      else {
        switch(sizeof(T)) {
          case 1: return _mm_and_si128(_mm_set1_epi8(0xFF >> other), _mm_srli_epi32(self, other));
          case 2: return _mm_srli_epi16(self, other);
          case 4: return _mm_srli_epi32(self, other);
          case 8: return _mm_srli_epi64(self, other);
          default: assert(false && "unsupported arch/op combination"); return {};
        }
      }
    }

    // bitwise_xor
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> bitwise_xor(batch<T, A> const& self, batch<T, A> const& other, requires<sse2>) {
      return _mm_xor_si128(self, other);
    }

    // broadcast
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> broadcast(T val, requires<sse2>) {
      switch(sizeof(T)) {
        case 1: return _mm_set1_epi8(val);
        case 2: return _mm_set1_epi16(val);
        case 4: return _mm_set1_epi32(val);
        case 8: return _mm_set1_epi64x(val);
        default: assert(false && "unsupported arch/op combination"); return {};
      }
    }
    template<class A> batch<float, A> broadcast(float val, requires<sse2>) {
      return _mm_set1_ps(val);
    }
    template<class A> batch<double, A> broadcast(double val, requires<sse2>) {
      return _mm_set1_pd(val);
    }

    // convert
    namespace conversion {
    template<class A> batch<float, A> fast(batch<int32_t, A> const& self, batch<float, A> const&, requires<sse2>) {
      return _mm_cvtepi32_ps(self);
    }
    template<class A> batch<int32_t, A> fast(batch<float, A> const& self, batch<int32_t, A> const&, requires<sse2>) {
      return _mm_cvtps_epi32(self);
    }
    }

    // eq
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch_bool<T, A> eq(batch<T, A> const& self, batch<T, A> const& other, requires<sse2>) {
      switch(sizeof(T)) {
        case 1: return _mm_cmpeq_epi8(self, other);
        case 2: return _mm_cmpeq_epi16(self, other);
        case 4: return _mm_cmpeq_epi32(self, other);
        case 8: {
            __m128i tmp1 = _mm_cmpeq_epi32(self, other);
            __m128i tmp2 = _mm_shuffle_epi32(tmp1, 0xB1);
            __m128i tmp3 = _mm_and_si128(tmp1, tmp2);
            __m128i tmp4 = _mm_srai_epi32(tmp3, 31);
            return _mm_shuffle_epi32(tmp4, 0xF5);
        }
        default: assert(false && "unsupported arch/op combination"); return {};
      }
    }
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch_bool<T, A> eq(batch_bool<T, A> const& self, batch_bool<T, A> const& other, requires<sse2>) {
      return eq(batch<T, A>(self.data), batch<T, A>(other.data));
    }

    // gt
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch_bool<T, A> gt(batch<T, A> const& self, batch<T, A> const& other, requires<sse2>) {
      if(std::is_signed<T>::value) {
        switch(sizeof(T)) {
          case 1: return _mm_cmpgt_epi8(self, other);
          case 2: return _mm_cmpgt_epi16(self, other);
          case 4: return _mm_cmpgt_epi32(self, other);
          default: return gt(self, other, generic{});
        }
      }
      else {
        return gt(self, other, generic{});
      }
    }

    // hadd
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    T hadd(batch<T, A> const& self, requires<sse2>) {
      switch(sizeof(T)) {
        case 4: {
                __m128i tmp1 = _mm_shuffle_epi32(self, 0x0E);
                __m128i tmp2 = _mm_add_epi32(self, tmp1);
                __m128i tmp3 = _mm_shuffle_epi32(tmp2, 0x01);
                __m128i tmp4 = _mm_add_epi32(tmp2, tmp3);
                return _mm_cvtsi128_si32(tmp4);
                }
        case 8: {
                __m128i tmp1 = _mm_shuffle_epi32(self, 0x0E);
                __m128i tmp2 = _mm_add_epi64(self, tmp1);
#if defined(__x86_64__)
                return _mm_cvtsi128_si64(tmp2);
#else
                __m128i m;
                _mm_storel_epi64(&m, tmp2);
                int64_t i;
                std::memcpy(&i, &m, sizeof(i));
                return i;
#endif
        }
        default: return hadd(self, sse{});
      }
    }

    // load_aligned
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> load_aligned(T const* mem, convert<T>, requires<sse2>) {
      return _mm_load_si128((__m128i const*)mem);
    }

    // load_unaligned
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> load_unaligned(T const* mem, convert<T>, requires<sse2>) {
      return _mm_loadu_si128((__m128i const*)mem);
    }

    // lt
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch_bool<T, A> lt(batch<T, A> const& self, batch<T, A> const& other, requires<sse2>) {
      if(std::is_signed<T>::value) {
        switch(sizeof(T)) {
          case 1: return _mm_cmplt_epi8(self, other);
          case 2: return _mm_cmplt_epi16(self, other);
          case 4: return _mm_cmplt_epi32(self, other);
          case 8: {
              __m128i tmp1 = _mm_sub_epi64(self, other);
              __m128i tmp2 = _mm_xor_si128(self, other);
              __m128i tmp3 = _mm_andnot_si128(other, self);
              __m128i tmp4 = _mm_andnot_si128(tmp2, tmp1);
              __m128i tmp5 = _mm_or_si128(tmp3, tmp4);
              __m128i tmp6 = _mm_srai_epi32(tmp5, 31);
              return _mm_shuffle_epi32(tmp6, 0xF5);
          }
          default: assert(false && "unsupported arch/op combination"); return {};
        }
      }
      else {
        switch(sizeof(T)) {
          case 1: return _mm_cmplt_epi8(_mm_xor_si128(self, _mm_set1_epi8(std::numeric_limits<int8_t>::lowest())), _mm_xor_si128(other, _mm_set1_epi8(std::numeric_limits<int8_t>::lowest())));
          case 2: return _mm_cmplt_epi16(_mm_xor_si128(self, _mm_set1_epi16(std::numeric_limits<int16_t>::lowest())), _mm_xor_si128(other, _mm_set1_epi16(std::numeric_limits<int16_t>::lowest())));
          case 4: return _mm_cmplt_epi32(_mm_xor_si128(self, _mm_set1_epi32(std::numeric_limits<int32_t>::lowest())), _mm_xor_si128(other, _mm_set1_epi32(std::numeric_limits<int32_t>::lowest())));
          case 8: {
                auto xself = _mm_xor_si128(self, _mm_set1_epi64x(std::numeric_limits<int64_t>::lowest()));
                auto xother = _mm_xor_si128(other, _mm_set1_epi64x(std::numeric_limits<int64_t>::lowest()));
                __m128i tmp1 = _mm_sub_epi64(xself, xother);
                __m128i tmp2 = _mm_xor_si128(xself, xother);
                __m128i tmp3 = _mm_andnot_si128(xother, xself);
                __m128i tmp4 = _mm_andnot_si128(tmp2, tmp1);
                __m128i tmp5 = _mm_or_si128(tmp3, tmp4);
                __m128i tmp6 = _mm_srai_epi32(tmp5, 31);
                return _mm_shuffle_epi32(tmp6, 0xF5);
                  }
          default: assert(false && "unsupported arch/op combination"); return {};
        }
      }
    }
    // max
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> max(batch<T, A> const& self, batch<T, A> const& other, requires<sse2>) {
      return select(self > other, self, other);
    }

    // min
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> min(batch<T, A> const& self, batch<T, A> const& other, requires<sse2>) {
      return select(self <= other, self, other);
    }

    // select
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> select(batch_bool<T, A> const& cond, batch<T, A> const& true_br, batch<T, A> const& false_br, requires<sse2>) {
      return _mm_or_si128(_mm_and_si128(cond, true_br), _mm_andnot_si128(cond, false_br));
    }
    template<class A, class T, bool... Values, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> select(batch_bool_constant<batch<T, A>, Values...> const&, batch<T, A> const& true_br, batch<T, A> const& false_br, requires<sse2>) {
      return select(batch_bool<T, A>{Values...}, true_br, false_br, sse2{});
    }

    // sadd
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> sadd(batch<T, A> const& self, batch<T, A> const& other, requires<sse2>) {
      if(std::is_signed<T>::value) {
        switch(sizeof(T)) {
          case 1: return _mm_adds_epi8(self, other);
          case 2: return _mm_adds_epi16(self, other);
          default: return sadd(self, other, sse{});
        }
      }
      else {
        switch(sizeof(T)) {
          case 1: return _mm_adds_epu8(self, other);
          case 2: return _mm_adds_epu16(self, other);
          default: return sadd(self, other, sse{});
        }
      }
    }

    // set
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> set(batch<T, A> const&, requires<sse2>, T v0, T v1) {
      return _mm_set_epi64x(v1, v0);
    }
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> set(batch<T, A> const&, requires<sse2>, T v0, T v1, T v2, T v3) {
      return _mm_setr_epi32(v0, v1, v2, v3);
    }
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> set(batch<T, A> const&, requires<sse2>, T v0, T v1, T v2, T v3, T v4, T v5, T v6, T v7) {
      return _mm_setr_epi16(v0, v1, v2, v3, v4, v5, v6, v7);
    }
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> set(batch<T, A> const&, requires<sse2>, T v0, T v1, T v2, T v3, T v4, T v5, T v6, T v7, T v8, T v9, T v10, T v11, T v12, T v13, T v14, T v15) {
      return _mm_setr_epi8(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15);
    }

    template<class A, class T, class... Values, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch_bool<T, A> set(batch_bool<T, A> const&, requires<sse2>, Values... values) {
      return set(batch<T, A>(), A{}, static_cast<T>(values ? -1LL : 0LL )...).data;
    }

    template<class A, class... Values>
    batch_bool<float, A> set(batch_bool<float, A> const&, requires<sse2>, Values... values) {
      static_assert(sizeof...(Values) == batch_bool<float, A>::size, "consistent init");
      return _mm_castsi128_ps(set(batch<int32_t, A>(), A{}, static_cast<int32_t>(values ? -1LL : 0LL )...).data);
    }

    template<class A, class... Values>
    batch_bool<double, A> set(batch_bool<double, A> const&, requires<sse2>, Values... values) {
      static_assert(sizeof...(Values) == batch_bool<double, A>::size, "consistent init");
      return _mm_castsi128_pd(set(batch<int64_t, A>(), A{},  static_cast<int64_t>(values ? -1LL : 0LL )...).data);
    }

    // ssub
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> ssub(batch<T, A> const& self, batch<T, A> const& other, requires<sse2>) {
      if(std::is_signed<T>::value) {
        switch(sizeof(T)) {
          case 1: return _mm_subs_epi8(self, other);
          case 2: return _mm_subs_epi16(self, other);
          default: return ssub(self, other, sse{});
        }
      }
      else {
        switch(sizeof(T)) {
          case 1: return _mm_subs_epu8(self, other);
          case 2: return _mm_subs_epu16(self, other);
          default: return ssub(self, other, sse{});
        }
      }
    }

    // store_aligned
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    void store_aligned(T *mem, batch<T, A> const& self, requires<sse2>) {
      return _mm_store_si128((__m128i *)mem, self);
    }
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    void store_aligned(T *mem, batch_bool<T, A> const& self, requires<sse2>) {
      return _mm_store_si128((__m128i *)mem, self);
    }

    // store_unaligned
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    void store_unaligned(T *mem, batch<T, A> const& self, requires<sse2>) {
      return _mm_storeu_si128((__m128i *)mem, self);
    }
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    void store_unaligned(T *mem, batch_bool<T, A> const& self, requires<sse2>) {
      return _mm_storeu_si128((__m128i *)mem, self);
    }

    // sub
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> sub(batch<T, A> const& self, batch<T, A> const& other, requires<sse2>) {
      switch(sizeof(T)) {
        case 1: return _mm_sub_epi8(self, other);
        case 2: return _mm_sub_epi16(self, other);
        case 4: return _mm_sub_epi32(self, other);
        case 8: return _mm_sub_epi64(self, other);
        default: assert(false && "unsupported arch/op combination"); return {};
      }
    }
    template<class A> batch<float, A> sub(batch<float, A> const& self, batch<float, A> const& other, requires<sse2>) {
      return _mm_sub_ps(self, other);
    }
    template<class A> batch<double, A> sub(batch<double, A> const& self, batch<double, A> const& other, requires<sse2>) {
      return _mm_sub_pd(self, other);
    }

    // to_float
    template<class A>
    batch<float, A> to_float(batch<int32_t, A> const& self, requires<sse2>) {
      return _mm_cvtepi32_ps(self);
    }
    template<class A>
    batch<double, A> to_float(batch<int64_t, A> const& self, requires<sse2>) {
      // FIXME: call _mm_cvtepi64_pd
      alignas(A::alignment()) int64_t buffer[batch<int64_t, A>::size];
      self.store_aligned(&buffer[0]);
      return {(double)buffer[0], (double)buffer[1]};
    }

    // to_int
    template<class A>
    batch<int32_t, A> to_int(batch<float, A> const& self, requires<sse2>) {
      return _mm_cvttps_epi32(self);
    }

    template<class A>
    batch<int64_t, A> to_int(batch<double, A> const& self, requires<sse2>) {
      // FIXME: call _mm_cvttpd_epi64
      alignas(A::alignment()) double buffer[batch<double, A>::size];
      self.store_aligned(&buffer[0]);
      return {(int64_t)buffer[0], (int64_t)buffer[1]};
    }

    // zip_hi
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> zip_hi(batch<T, A> const& self, batch<T, A> const& other, requires<sse2>) {
      switch(sizeof(T)) {
        case 1: return _mm_unpackhi_epi8(self, other);
        case 2: return _mm_unpackhi_epi16(self, other);
        case 4: return _mm_unpackhi_epi32(self, other);
        case 8: return _mm_unpackhi_epi64(self, other);
        default: assert(false && "unsupported arch/op combination"); return {};
      }
    }

    // zip_lo
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> zip_lo(batch<T, A> const& self, batch<T, A> const& other, requires<sse2>) {
      switch(sizeof(T)) {
        case 1: return _mm_unpacklo_epi8(self, other);
        case 2: return _mm_unpacklo_epi16(self, other);
        case 4: return _mm_unpacklo_epi32(self, other);
        case 8: return _mm_unpacklo_epi64(self, other);
        default: assert(false && "unsupported arch/op combination"); return {};
      }
    }

  }

}

#endif


