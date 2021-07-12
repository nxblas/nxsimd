#ifndef XSIMD_REGISTER_HPP
#define XSIMD_REGISTER_HPP

namespace xsimd {

template<class T, class A>
struct batch;

template<class T, class A>
struct batch_bool;

namespace types {

  template<class T, class Arch>
  struct simd_register;

  template <class T, class Arch>
  struct get_bool_simd_register
  {
  using type = simd_register<T, Arch>;
  };

  template <class T, class Arch>
  using get_bool_simd_register_t = typename get_bool_simd_register<T, Arch>::type;


#define XSIMD_DECLARE_SIMD_REGISTER(SCALAR_TYPE, ISA, VECTOR_TYPE) \
    template<> \
    struct simd_register<SCALAR_TYPE, ISA> {\
      using register_type = VECTOR_TYPE;\
      register_type data;\
      operator register_type() const { return data; }\
    }

#define XSIMD_DECLARE_SIMD_REGISTER_ALIAS(ISA, ISA_BASE) \
    template<class T> \
    struct simd_register<T, ISA> : simd_register<T, ISA_BASE> {\
      using register_type = typename simd_register<T, ISA_BASE>::register_type; \
      simd_register(register_type reg) : simd_register<T, ISA_BASE>{reg} {} \
      simd_register() = default; \
    }
}

namespace kernel {
  template<class A>
  using requires = A const&;
  template<class T>
  struct convert {};
}

}

#endif
