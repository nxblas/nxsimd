#ifndef XSIMD_BATCH_HPP
#define XSIMD_BATCH_HPP

#include "../config/xsimd_arch.hpp"
#include "../memory/xsimd_alignment.hpp"
#include "./xsimd_utils.hpp"

#include <cassert>

namespace xsimd {

template<class T, class A=default_arch>
struct batch : types::simd_register<T, A> {

  static constexpr std::size_t size = sizeof(types::simd_register<T, A>) / sizeof(T);

  using value_type = T;
  using arch_type = A;
  using register_type = typename types::simd_register<T, A>::register_type;
  using batch_bool_type = batch_bool<T, A>;

  // constructors
  batch() = default;
  batch(T val);
  batch(std::initializer_list<T> data) : batch(data.begin(), detail::make_index_sequence<size>()) {}
  explicit batch(batch_bool_type b);
  batch(register_type reg) : types::simd_register<T, A>({reg}) {}

  template<class U>
  static XSIMD_NO_DISCARD batch broadcast(U val) { return batch(static_cast<T>(val)); }

  // memory operators
  template<class U>
  void store_aligned(U * mem) const;
  template<class U>
  void store_unaligned(U * mem) const;
  template<class U>
  void store(U * mem, aligned_mode) const { return store_aligned(mem); }
  template<class U>
  void store(U * mem, unaligned_mode) const { return store_unaligned(mem); }

  template<class U>
  static XSIMD_NO_DISCARD batch load_aligned(U const* mem) ;
  template<class U>
  static XSIMD_NO_DISCARD batch load_unaligned(U const* mem);
  template<class U>
  static XSIMD_NO_DISCARD batch load(U const* mem, aligned_mode) { return load_aligned(mem); }
  template<class U>
  static XSIMD_NO_DISCARD batch load(U const* mem, unaligned_mode) { return load_unaligned(mem); }

  T get(std::size_t i) const {
    alignas(A::alignment()) T buffer[size];
    store_aligned(&buffer[0]);
    return buffer[i];
  }

  // unary operators
  batch_bool_type operator!() const;
  batch operator~() const;
  batch operator-() const;
  batch operator+() const { return *this; }

  // comparison operators
  batch_bool_type operator==(batch const& other) const;
  batch_bool_type operator!=(batch const& other) const;
  batch_bool_type operator>=(batch const& other) const;
  batch_bool_type operator<=(batch const& other) const;
  batch_bool_type operator>(batch const& other) const;
  batch_bool_type operator<(batch const& other) const;

  // arithmetic operators. They are defined as friend to enable automatic
  // conversion of parameters from scalar to batch
  friend batch operator+(batch const& self, batch const& other) {
    return batch(self) += other;
  }
  friend batch operator-(batch const& self, batch const& other) {
    return batch(self) -= other;
  }
  friend batch operator*(batch const& self, batch const& other) {
    return batch(self) *= other;
  }
  friend batch operator/(batch const& self, batch const& other) {
    return batch(self) /= other;
  }
  friend batch operator%(batch const& self, batch const& other) {
    return batch(self) %= other;
  }

  friend batch operator&(batch const& self, batch const& other) {
    return batch(self) &= other;
  }

  friend batch operator|(batch const& self, batch const& other) {
    return batch(self) |= other;
  }

  friend batch operator^(batch const& self, batch const& other) {
    return batch(self) ^= other;
  }

  friend batch operator>>(batch const& self, batch const& other) {
    return batch(self) >>= other;
  }

  friend batch operator<<(batch const& self, batch const& other) {
    return batch(self) <<= other;
  }

  friend batch operator>>(batch const& self, int32_t other) {
    return batch(self) >>= other;
  }

  friend batch operator<<(batch const& self, int32_t other) {
    return batch(self) <<= other;
  }

  // Update operators
  batch& operator+=(batch const& other);
  batch& operator-=(batch const& other);
  batch& operator*=(batch const& other);
  batch& operator/=(batch const& other);
  batch& operator%=(batch const& other);
  batch& operator&=(batch const& other);
  batch& operator|=(batch const& other);
  batch& operator^=(batch const& other);
  batch& operator>>=(int32_t other);
  batch& operator>>=(batch const& other);
  batch& operator<<=(int32_t other);
  batch& operator<<=(batch const& other);

  // incr/decr
  batch& operator++() { return operator+=(1);}
  batch& operator--() { return operator-=(1);}
  batch operator++(int) { batch copy(*this); operator+=(1); return copy;}
  batch operator--(int) { batch copy(*this); operator-=(1); return copy;}

  private:
  template<size_t... Is>
  batch(T const* data, detail::index_sequence<Is...>);
};

template<class T, class A=default_arch>
struct batch_bool : types::get_bool_simd_register_t<T, A> {
  static constexpr std::size_t size = sizeof(types::simd_register<T, A>) / sizeof(T);

  using value_type = bool;
  using register_type = typename types::get_bool_simd_register_t<T, A>::register_type;
  using batch_type = batch<T, A>;

  batch_bool() = default;
  batch_bool(bool val);
  batch_bool(register_type reg) : types::get_bool_simd_register_t<T, A>({reg}) {}
  batch_bool(std::initializer_list<bool> data) : batch_bool(data.begin(), detail::make_index_sequence<size>()) {}

  void store_aligned(bool * mem) const;
  void store_unaligned(bool * mem) const;
  static XSIMD_NO_DISCARD batch_bool load_aligned(bool const * mem);
  static XSIMD_NO_DISCARD batch_bool load_unaligned(bool const * mem);

  batch_bool operator~() const;
  batch_bool operator!() const { return operator==(batch_bool(false)); }
  batch_bool operator==(batch_bool const& other) const;
  batch_bool operator!=(batch_bool const& other) const;
  batch_bool operator&(batch_bool const& other) const;
  batch_bool operator|(batch_bool const& other) const;
  batch_bool operator&&(batch_bool const& other) const {
    return operator&(other);
  }
  batch_bool operator||(batch_bool const& other) const {
    return operator|(other);
  }

  bool get(std::size_t i) const {
    alignas(A::alignment()) bool buffer[size];
    store_aligned(&buffer[0]);
    return buffer[i];
  }

  private:
  template<size_t... Is>
  batch_bool(bool const* data, detail::index_sequence<Is...>);


};

template<class T, class A>
struct batch<std::complex<T>, A> {
  using value_type = std::complex<T>;
  using real_batch = batch<T, A>;
  using arch_type = A;
  static constexpr std::size_t size = real_batch::size;

  batch() = default;
  batch(value_type const& val) : m_real(val.real()), m_imag(val.imag()) {}
  batch(real_batch const& real, real_batch const& imag) : m_real(real), m_imag(imag) {}
  batch(real_batch const& real) : m_real(real), m_imag(0) {}
  batch(T val) : m_real(val), m_imag(0) {}
  batch(std::initializer_list<std::complex<T>> data) { *this = load_unaligned(data.begin()); }

        static XSIMD_NO_DISCARD batch load_aligned(const T* real_src, const T* imag_src=nullptr);
        static XSIMD_NO_DISCARD batch load_unaligned(const T* real_src, const T* imag_src=nullptr);
        void store_aligned(T* real_dst, T* imag_dst) const;
        void store_unaligned(T* real_dst, T* imag_dst) const;

        static XSIMD_NO_DISCARD batch load_aligned(const value_type* src);
        static XSIMD_NO_DISCARD batch load_unaligned(const value_type* src);
        void store_aligned(value_type* dst) const;
        void store_unaligned(value_type* dst) const;

  template<class U>
  static XSIMD_NO_DISCARD batch load(U const* mem, aligned_mode) { return load_aligned(mem); }
  template<class U>
  static XSIMD_NO_DISCARD batch load(U const* mem, unaligned_mode) { return load_unaligned(mem); }
  template<class U>
  void store(U * mem, aligned_mode) const { return store_aligned(mem); }
  template<class U>
  void store(U * mem, unaligned_mode) const { return store_unaligned(mem); }

    real_batch real() const { return m_real; }
    real_batch imag() const { return m_imag; }

  value_type get(std::size_t i) const {
    alignas(A::alignment()) value_type buffer[size];
    store_aligned(&buffer[0]);
    return buffer[i];
  }

  // unary operators
  batch operator~() const { return {~m_real, ~m_imag}; }
  batch operator-() const { return {-m_real, -m_imag};}
  batch operator+() const { return {+m_real, +m_imag}; }

  // comparison operators
  batch_bool<T, A> operator==(batch const& other) const { return m_real == other.m_real && m_imag == other.m_imag; }
  batch_bool<T, A> operator!=(batch const& other) const { return m_real != other.m_real || m_imag != other.m_imag; }

  // arithmetic operators. They are defined as friend to enable automatic
  // conversion of parameters from scalar to batch
  friend batch operator+(batch const& self, batch const& other) {
    return batch(self) += other;
  }
  friend batch operator-(batch const& self, batch const& other) {
    return batch(self) -= other;
  }
  friend batch operator*(batch const& self, batch const& other) {
    return batch(self) *= other;
  }
  friend batch operator/(batch const& self, batch const& other) {
    return batch(self) /= other;
  }


  // Update operators
  batch& operator+=(batch const& other) {
    m_real += other.m_real;
    m_imag += other.m_imag;
    return *this;
  }
  batch& operator-=(batch const& other){
    m_real -= other.m_real;
    m_imag -= other.m_imag;
    return *this;
  }
  batch& operator*=(batch const& other);

  batch& operator/=(batch const& other);

  // incr/decr
  batch& operator++() { return operator+=(1);}
  batch& operator--() { return operator-=(1);}
  batch operator++(int) { batch copy(*this); operator+=(1); return copy;}
  batch operator--(int) { batch copy(*this); operator-=(1); return copy;}


    protected:

        real_batch m_real;
        real_batch m_imag;
};

}

#include "../types/xsimd_batch_constant.hpp"
#include "../arch/xsimd_isa.hpp"

namespace xsimd {

// batch implementation

template<class T, class A>
batch<T, A>::batch(T val) : types::simd_register<T, A>(kernel::broadcast<A>(val, A{})) {}

template<class T, class A>
batch<T, A>::batch(batch_bool<T, A> b) : batch(batch(b.data) & batch(1)) {}

template<class T, class A>
template<size_t... Is>
batch<T, A>::batch(T const*data, detail::index_sequence<Is...>) : batch(kernel::set<A>(batch{}, A{}, data[Is]...)) {}

template<class T, class A>
template<class U>
void batch<T, A>::store_aligned(U* mem) const {
  kernel::store_aligned<A>(mem, *this, A{});
}

template<class T, class A>
template<class U>
void batch<T, A>::store_unaligned(U* mem) const {
  kernel::store_unaligned<A>(mem, *this, A{});
}

template<class T, class A>
template<class U>
batch<T, A> batch<T, A>::load_aligned(U const* mem) {
  return kernel::load_aligned<A>(mem, kernel::convert<T>{}, A{});
}

template<class T, class A>
template<class U>
batch<T, A> batch<T, A>::load_unaligned(U const* mem) {
  return kernel::load_unaligned<A>(mem, kernel::convert<T>{}, A{});
}

template<class T, class A>
batch_bool<T, A> batch<T, A>::operator!() const { return kernel::eq<A>(*this, batch(0), A{}); }

template<class T, class A>
batch<T, A> batch<T, A>::operator~() const { return kernel::bitwise_not<A>(*this, A{}); }

template<class T, class A>
batch<T, A> batch<T, A>::operator-() const { return kernel::neg<A>(*this, A{}); }

template<class T, class A>
batch_bool<T, A> batch<T, A>::operator==(batch<T, A> const& other) const { return kernel::eq<A>(*this, other, A{}); }

template<class T, class A>
batch_bool<T, A> batch<T, A>::operator!=(batch<T, A> const& other) const { return kernel::neq<A>(*this, other, A{}); }

template<class T, class A>
batch_bool<T, A> batch<T, A>::operator>=(batch<T, A> const& other) const { return kernel::ge<A>(*this, other, A{}); }

template<class T, class A>
batch_bool<T, A> batch<T, A>::operator<=(batch<T, A> const& other) const { return kernel::le<A>(*this, other, A{}); }

template<class T, class A>
batch_bool<T, A> batch<T, A>::operator>(batch<T, A> const& other) const { return kernel::gt<A>(*this, other, A{}); }

template<class T, class A>
batch_bool<T, A> batch<T, A>::operator<(batch<T, A> const& other) const { return kernel::lt<A>(*this, other, A{}); }

template<class T, class A>
batch<T, A>& batch<T, A>::operator+=(batch<T, A> const& other) { return *this = kernel::add<A>(*this, other, A{}); }

template<class T, class A>
batch<T, A>& batch<T, A>::operator-=(batch<T, A> const& other) { return *this = kernel::sub<A>(*this, other, A{}); }

template<class T, class A>
batch<T, A>& batch<T, A>::operator*=(batch<T, A> const& other) { return *this = kernel::mul<A>(*this, other, A{}); }

template<class T, class A>
batch<T, A>& batch<T, A>::operator/=(batch<T, A> const& other) { return *this = kernel::div<A>(*this, other, A{}); }

template<class T, class A>
batch<T, A>& batch<T, A>::operator%=(batch<T, A> const& other) { return *this = kernel::mod<A>(*this, other, A{}); }

template<class T, class A>
batch<T, A>& batch<T, A>::operator&=(batch<T, A> const& other) { return *this = kernel::bitwise_and<A>(*this, other, A{}); }

template<class T, class A>
batch<T, A>& batch<T, A>::operator|=(batch<T, A> const& other) { return *this = kernel::bitwise_or<A>(*this, other, A{}); }

template<class T, class A>
batch<T, A>& batch<T, A>::operator^=(batch<T, A> const& other) { return *this = kernel::bitwise_xor<A>(*this, other, A{}); }

template<class T, class A>
batch<T, A>& batch<T, A>::operator>>=(batch<T, A> const& other) { return *this = kernel::bitwise_rshift<A>(*this, other, A{}); }

template<class T, class A>
batch<T, A>& batch<T, A>::operator<<=(batch<T, A> const& other) { return *this = kernel::bitwise_lshift<A>(*this, other, A{}); }

template<class T, class A>
batch<T, A>& batch<T, A>::operator>>=(int32_t other) { return *this = kernel::bitwise_rshift<A>(*this, other, A{}); }

template<class T, class A>
batch<T, A>& batch<T, A>::operator<<=(int32_t other) { return *this = kernel::bitwise_lshift<A>(*this, other, A{}); }

// batch_bool implementation
template<class T, class A>
template<size_t... Is>
batch_bool<T, A>::batch_bool(bool const*data, detail::index_sequence<Is...>) : batch_bool(kernel::set<A>(batch_bool{}, A{}, data[Is]...)) {}

template<class T, class A>
batch_bool<T, A> batch_bool<T, A>::operator~() const {
  return kernel::bitwise_not<A>(*this, A{}).data;
}

template<class T, class A>
batch_bool<T, A> batch_bool<T, A>::operator==(batch_bool<T, A> const& other) const {
  return kernel::eq<A>(*this, other, A{}).data;
}

template<class T, class A>
batch_bool<T, A> batch_bool<T, A>::operator!=(batch_bool<T, A> const& other) const {
  return kernel::neq<A>(*this, other, A{}).data;
}

template<class T, class A>
batch_bool<T, A> batch_bool<T, A>::operator&(batch_bool<T, A> const& other) const {
  return kernel::bitwise_and<A>(*this, other, A{}).data;
}

template<class T, class A>
batch_bool<T, A> batch_bool<T, A>::operator|(batch_bool<T, A> const& other) const {
  return kernel::bitwise_or<A>(*this, other, A{}).data;
}

template<class T, class A>
batch_bool<T, A>::batch_bool(bool val) : types::get_bool_simd_register_t<T, A>(val?
    (batch_type(0) == batch_type(0)):
    (batch_type(0) != batch_type(0))) {
}


template<class T, class A>
void batch_bool<T, A>::store_aligned(bool* mem) const {
  alignas(A::alignment()) T buffer[size];
  kernel::store_aligned<A>(&buffer[0], batch_type(*this), A{});
  for(std::size_t i = 0; i < size; ++i)
    mem[i] = bool(buffer[i]);
}

template<class T, class A>
void batch_bool<T, A>::store_unaligned(bool* mem) const {
  store_aligned(mem);
}

template<class T, class A>
batch_bool<T, A> batch_bool<T, A>::load_aligned(bool const* mem) {
  batch_type ref(0);
  alignas(A::alignment()) T buffer[size];
  for(std::size_t i = 0; i < size; ++i)
    buffer[i] = mem[i] ? 1 : 0;
  return ref != batch_type::load_aligned(&buffer[0]);
}

template<class T, class A>
batch_bool<T, A> batch_bool<T, A>::load_unaligned(bool const* mem) {
  return load_aligned(mem);
}

// batch complex implementation
//
template <class T, class A>
batch<std::complex<T>, A> batch<std::complex<T>, A>::load_aligned(const T* real_src, const T* imag_src)
{
    return {batch<T, A>::load_aligned(real_src), imag_src?batch<T, A>::load_aligned(imag_src): batch<T, A>(0)};
}
template <class T, class A>
batch<std::complex<T>, A> batch<std::complex<T>, A>::load_unaligned(const T* real_src, const T* imag_src)
{
    return {batch<T, A>::load_unaligned(real_src), imag_src?batch<T, A>::load_unaligned(imag_src):batch<T, A>(0)};
}

template<class T, class A>
batch<std::complex<T>, A>
batch<std::complex<T>, A>::load_aligned(const value_type* src)
{
  T const *buffer = reinterpret_cast<T const *>(src);
  real_batch hi = real_batch::load_aligned(buffer),
             lo = real_batch::load_aligned(buffer + real_batch::size);
  return kernel::load_complex<A>(hi, lo, A{});
}

template<class T, class A>
batch<std::complex<T>, A>
batch<std::complex<T>, A>::load_unaligned(const value_type* src)
{
  T const *buffer = reinterpret_cast<T const *>(src);
  real_batch hi = real_batch::load_unaligned(buffer),
             lo = real_batch::load_unaligned(buffer + real_batch::size);
  return kernel::load_complex<A>(hi, lo, A{});
}

template<class T, class A>
void batch<std::complex<T>, A>::store_aligned(value_type* dst) const {
          real_batch hi = kernel::complex_high(*this, A{});
          real_batch lo = kernel::complex_low(*this, A{});
          T * buffer = reinterpret_cast<T*>(dst);
          lo.store_aligned(buffer);
          hi.store_aligned(buffer + size);
        }

template<class T, class A>
void batch<std::complex<T>, A>::store_unaligned(value_type* dst) const {
          real_batch hi = kernel::complex_high(*this, A{});
          real_batch lo = kernel::complex_low(*this, A{});
          T * buffer = reinterpret_cast<T *>(dst);
          lo.store_unaligned(buffer);
          hi.store_unaligned(buffer + size);
        }

template<class T, class A>
void batch<std::complex<T>, A>::store_aligned(T* real_dst, T* imag_dst) const {
  m_real.store_aligned(real_dst);
  m_imag.store_aligned(imag_dst);
}

template<class T, class A>
void batch<std::complex<T>, A>::store_unaligned(T* real_dst, T* imag_dst) const {
  m_real.store_unaligned(real_dst);
  m_imag.store_unaligned(imag_dst);
}

template<class T, class A>
batch<std::complex<T>, A>& batch<std::complex<T>, A>::operator*=(batch const& other) {
  real_batch new_real = real() * other.real() - imag() * other.imag();
  real_batch new_imag = real() * other.imag() + imag() * other.real();
  m_real = new_real;
  m_imag = new_imag;
  return *this;
}

template<class T, class A>
batch<std::complex<T>, A>& batch<std::complex<T>, A>::operator/=(batch const& other) {
  real_batch a = real();
  real_batch b = imag();
  real_batch c = other.real();
  real_batch d = other.imag();
  real_batch e = c*c + d*d;
  m_real = (c*a + d*b) / e;
  m_imag = (c*b - d*a) / e;
  return *this;
}

}

#endif
