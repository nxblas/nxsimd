#ifndef XSIMD_AVX512F_REGISTER_HPP
#define XSIMD_AVX512F_REGISTER_HPP

#include "./xsimd_generic_arch.hpp"

namespace xsimd {

  struct avx512f : generic {
    static constexpr bool supported() { return XSIMD_WITH_AVX512F; }
    static constexpr bool available() { return true; }
    static constexpr unsigned version() { return generic::version(3, 1, 0); }
    static constexpr std::size_t alignment() { return 64; }
  };

#if XSIMD_WITH_AVX512F

namespace types {
template <class T> struct simd_avx512_bool_register {
  using register_type = typename std::conditional<
      (sizeof(T) < 4), std::conditional<(sizeof(T) == 1), __mmask64, __mmask32>,
      std::conditional<(sizeof(T) == 4), __mmask16, __mmask8>>::type::type;
  register_type data;
  operator register_type() const { return data; }
};
template <class T> struct get_bool_simd_register<T, avx512f> {
  using type = simd_avx512_bool_register<T>;
};



    XSIMD_DECLARE_SIMD_REGISTER(bool, avx512f, __m512i);
    XSIMD_DECLARE_SIMD_REGISTER(signed char, avx512f, __m512i);
    XSIMD_DECLARE_SIMD_REGISTER(unsigned char, avx512f, __m512i);
    XSIMD_DECLARE_SIMD_REGISTER(char, avx512f, __m512i);
    XSIMD_DECLARE_SIMD_REGISTER(unsigned short, avx512f, __m512i);
    XSIMD_DECLARE_SIMD_REGISTER(short, avx512f, __m512i);
    XSIMD_DECLARE_SIMD_REGISTER(unsigned int, avx512f, __m512i);
    XSIMD_DECLARE_SIMD_REGISTER(int, avx512f, __m512i);
    XSIMD_DECLARE_SIMD_REGISTER(unsigned long int, avx512f, __m512i);
    XSIMD_DECLARE_SIMD_REGISTER(long int, avx512f, __m512i);
    XSIMD_DECLARE_SIMD_REGISTER(unsigned long long int, avx512f, __m512i);
    XSIMD_DECLARE_SIMD_REGISTER(long long int, avx512f, __m512i);
    XSIMD_DECLARE_SIMD_REGISTER(float, avx512f, __m512);
    XSIMD_DECLARE_SIMD_REGISTER(double, avx512f, __m512d);

  }
#endif
}
#endif

