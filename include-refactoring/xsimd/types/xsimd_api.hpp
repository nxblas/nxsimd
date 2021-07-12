#ifndef XSIMD_API_HPP
#define XSIMD_API_HPP

#include "../types/xsimd_batch.hpp"
#include "../arch/xsimd_isa.hpp"

#include <limits>

namespace xsimd {

// high level free functions
//

template<class T, class A>
batch<T, A> abs(batch<T, A> const& self) {
  return kernel::abs<A>(self, A{});
}

template<class T, class A>
batch<T, A> abs(batch<std::complex<T>, A> const& self) {
  return kernel::abs<A>(self, A{});
}

template<class T, class Tp>
auto add(T const& self, Tp const& other) -> decltype(self + other){
  return self + other;
}

template<class T, class A>
bool all(batch<T, A> const& self) {
  return kernel::all<A>(self, A{});
}

template<class T, class A>
bool any(batch<T, A> const& self) {
  return kernel::any<A>(self, A{});
}

template<class T, class A>
batch<T, A> acos(batch<T, A> const& self) {
  return kernel::acos<A>(self, A{});
}

template<class T, class A>
batch<T, A> acosh(batch<T, A> const& self) {
  return kernel::acosh<A>(self, A{});
}

template<class T, class A>
batch<T, A> arg(batch<std::complex<T>, A> const& self) {
  return kernel::arg<A>(self, A{});
}

template<class T, class A>
batch<T, A> asin(batch<T, A> const& self) {
  return kernel::asin<A>(self, A{});
}

template<class T, class A>
batch<T, A> asinh(batch<T, A> const& self) {
  return kernel::asinh<A>(self, A{});
}

template<class T, class A>
batch<T, A> atan(batch<T, A> const& self) {
  return kernel::atan<A>(self, A{});
}

template<class T, class A>
batch<T, A> atan2(batch<T, A> const& self, batch<T, A> const& other) {
  return kernel::atan2<A>(self, other, A{});
}

template<class T, class A>
batch<T, A> atanh(batch<T, A> const& self) {
  return kernel::atanh<A>(self, A{});
}

template<class T_out, class T_in, class A>
batch<T_out, A> batch_cast(batch<T_in, A> const & self) {
  return kernel::batch_cast<A>(self, batch<T_out, A>{}, A{});
}

template<class T, class A>
batch<T, A> bitofsign(batch<T, A> const& self) {
  return kernel::bitofsign<A>(self, A{});
}

template<class T, class Tp>
auto bitwise_and(T const& self, Tp const& other) -> decltype(self & other){
  return self & other;
}

template<class B, class T, class A>
B bitwise_cast(batch<T, A> const& self) {
  return kernel::bitwise_cast<A>(self, B{}, A{});
}

template<class T, class A>
batch<T, A> bitwise_not(batch<T, A> const& self) {
  return kernel::bitwise_not<A>(self, A{});
}

template<class T, class Tp>
auto bitwise_or(T const& self, Tp const& other) -> decltype(self | other){
  return self | other;
}

template<class T, class Tp>
auto bitwise_xor(T const& self, Tp const& other) -> decltype(self ^ other){
  return self ^ other;
}

template<class A>
batch_bool<float, A> bool_cast(batch_bool<int32_t, A> const& self) {
  return kernel::bool_cast<A>(self, A{});
}
template<class A>
batch_bool<int32_t, A> bool_cast(batch_bool<float, A> const& self) {
  return kernel::bool_cast<A>(self, A{});
}
template<class A>
batch_bool<double, A> bool_cast(batch_bool<int64_t, A> const& self) {
  return kernel::bool_cast<A>(self, A{});
}
template<class A>
batch_bool<int64_t, A> bool_cast(batch_bool<double, A> const& self) {
  return kernel::bool_cast<A>(self, A{});
}

template<class T, class A=default_arch>
batch<T, A> broadcast(T val) {
  return kernel::broadcast<A>(val, A{});
}

template<class T, class A>
batch<T, A> cbrt(batch<T, A> const& self) {
  return kernel::cbrt<A>(self, A{});
}

template<class T, class A>
batch<T, A> ceil(batch<T, A> const& self) {
  return kernel::ceil<A>(self, A{});
}

template<class A, class T>
batch<T, A> clip(batch<T, A> const& self, batch<T, A> const& lo, batch<T, A> const& hi) {
  return kernel::clip(self, lo, hi, A{});
}

template<class A, class T>
batch<std::complex<T>, A> conj(batch<std::complex<T>, A> const& self) {
  return kernel::conj(self, A{});
}

template<class A, class T>
batch<T, A> copysign(batch<T, A> const& self, batch<T, A> const& other) {
  return kernel::copysign<A>(self, other, A{});
}

template<class T, class A>
batch<T, A> cos(batch<T, A> const& self) {
  return kernel::cos<A>(self, A{});
}

template<class T, class A>
batch<T, A> cosh(batch<T, A> const& self) {
  return kernel::cosh<A>(self, A{});
}

template<class T, class Tp>
auto div(T const& self, Tp const& other) -> decltype(self / other){
  return self / other;
}

template<class T, class A>
batch_bool<T, A> eq(batch<T, A> const& self, batch<T, A> const& other) {
  return self == other;
}

template<class T, class A>
batch<T, A> exp(batch<T, A> const& self) {
  return kernel::exp<A>(self, A{});
}

template<class T, class A>
batch<T, A> exp10(batch<T, A> const& self) {
  return kernel::exp10<A>(self, A{});
}

template<class T, class A>
batch<T, A> exp2(batch<T, A> const& self) {
  return kernel::exp2<A>(self, A{});
}

template<class T, class A>
batch<T, A> expm1(batch<T, A> const& self) {
  return kernel::expm1<A>(self, A{});
}

template<class T, class A>
batch<T, A> erf(batch<T, A> const& self) {
  return kernel::erf<A>(self, A{});
}

template<class T, class A>
batch<T, A> erfc(batch<T, A> const& self) {
  return kernel::erfc<A>(self, A{});
}

template <class T, class A, uint64_t... Coefs>
batch<T, A> estrin(const batch<T, A>& self) {
  return kernel::estrin<T, A, Coefs...>(self);
}

template<class T, class A>
batch<T, A> fabs(batch<T, A> const& self) {
  return kernel::abs<A>(self, A{});
}

template<class T, class A>
batch<T, A> fdim(batch<T, A> const& self, batch<T, A> const& other) {
  return kernel::fdim<A>(self, other, A{});
}

template<class T, class A>
batch<T, A> floor(batch<T, A> const& self) {
  return kernel::floor<A>(self, A{});
}

template<class T, class A>
batch<T, A> fma(batch<T, A> const& x, batch<T, A> const& y, batch<T, A> const& z) {
  return kernel::fma<A>(x, y, z, A{});
}

template<class T, class A>
batch<T, A> fmax(batch<T, A> const& self, batch<T, A> const& other) {
  return kernel::max<A>(self, other, A{});
}

template<class T, class A>
batch<T, A> fmin(batch<T, A> const& self, batch<T, A> const& other) {
  return kernel::min<A>(self, other, A{});
}

template<class T, class A>
batch<T, A> fmod(batch<T, A> const& self, batch<T, A> const& other) {
  return kernel::fmod<A>(self, other, A{});
}

template<class T, class A>
batch<T, A> fms(batch<T, A> const& x, batch<T, A> const& y, batch<T, A> const& z) {
  return kernel::fms<A>(x, y, z, A{});
}

template<class T, class A>
batch<T, A> fnma(batch<T, A> const& x, batch<T, A> const& y, batch<T, A> const& z) {
  return kernel::fnma<A>(x, y, z, A{});
}

template<class T, class A>
batch<T, A> fnms(batch<T, A> const& x, batch<T, A> const& y, batch<T, A> const& z) {
  return kernel::fnms<A>(x, y, z, A{});
}

template <class T, class A>
batch<T, A> frexp(const batch<T, A>& self, batch<as_integer_t<T>, A>& other) {
  return kernel::frexp<A>(self, other, A{});
}

template<class T, class A>
batch_bool<T, A> ge(batch<T, A> const& self, batch<T, A> const& other) {
  return self >= other;
}

template<class T, class A>
batch_bool<T, A> gt(batch<T, A> const& self, batch<T, A> const& other) {
  return self > other;
}

template<class T, class A>
T hadd(batch<T, A> const& self) {
  return kernel::hadd<A>(self, A{});
}

template<class T, class A>
batch<T, A> haddp(batch<T, A> const* self) {
  return kernel::haddp<A>(self, A{});
}

template <class T, class A, uint64_t... Coefs>
batch<T, A> horner(const batch<T, A>& self) {
  return kernel::horner<T, A, Coefs...>(self);
}

template<class T, class A>
batch<T, A> hypot(batch<T, A> const& self, batch<T, A> const& other) {
  return kernel::hypot<A>(self, other, A{});
}


template<class B>
B infinity() {
  using T = typename B::value_type;
  return B(std::numeric_limits<T>::infinity());
}

template<class T, class A>
batch_bool<T, A> is_even(batch<T, A> const& self) {
  return kernel::is_even<A>(self, A{});
}

template<class T, class A>
batch_bool<T, A> is_flint(batch<T, A> const& self) {
  return kernel::is_flint<A>(self, A{});
}

template<class T, class A>
batch_bool<T, A> is_odd(batch<T, A> const& self) {
  return kernel::is_odd<A>(self, A{});
}

template<class T, class A>
batch_bool<T, A> isinf(batch<T, A> const& self) {
  return kernel::isinf<A>(self, A{});
}

template<class T, class A>
batch_bool<T, A> isfinite(batch<T, A> const& self) {
  return kernel::isfinite<A>(self, A{});
}

template<class T, class A>
batch_bool<T, A> isnan(batch<T, A> const& self) {
  return kernel::isnan<A>(self, A{});
}

template <class T, class A>
batch<T, A> ldexp(const batch<T, A>& self, const batch<as_integer_t<T>, A>& other) {
  return kernel::ldexp<A>(self, other, A{});
}

template<class T, class A>
batch_bool<T, A> le(batch<T, A> const& self, batch<T, A> const& other) {
  return self <= other;
}

template<class T, class A>
batch<T, A> lgamma(batch<T, A> const& self) {
  return kernel::lgamma<A>(self, A{});
}

template<class A=default_arch, class From>
batch<From, A> load(From const* ptr, aligned_mode= {}) {
  return kernel::load_aligned<A>(ptr, kernel::convert<From>{}, A{});
}

template<class A=default_arch, class From>
batch<From, A> load(From const* ptr, unaligned_mode) {
  return kernel::load_unaligned<A>(ptr, kernel::convert<From>{}, A{});
}

template<class A/*=default_arch*/, class From>
batch<From, A> load_aligned(From const* ptr) {
  return kernel::load_aligned<A>(ptr, kernel::convert<From>{}, A{});
}

template<class A/*=default_arch*/, class From>
batch<From, A> load_unaligned(From const* ptr) {
  return kernel::load_unaligned<A>(ptr, kernel::convert<From>{}, A{});
}

template<class T, class A>
batch<T, A> log(batch<T, A> const& self) {
  return kernel::log<A>(self, A{});
}

template<class T, class A>
batch<T, A> log2(batch<T, A> const& self) {
  return kernel::log2<A>(self, A{});
}

template<class T, class A>
batch<T, A> log10(batch<T, A> const& self) {
  return kernel::log10<A>(self, A{});
}

template<class T, class A>
batch<T, A> log1p(batch<T, A> const& self) {
  return kernel::log1p<A>(self, A{});
}

template<class T, class A>
batch_bool<T, A> lt(batch<T, A> const& self, batch<T, A> const& other) {
  return self < other;
}

template<class T, class A>
batch<T, A> max(batch<T, A> const& self, batch<T, A> const& other) {
  return kernel::max<A>(self, other, A{});
}

template<class T, class A>
batch<T, A> min(batch<T, A> const& self, batch<T, A> const& other) {
  return kernel::min<A>(self, other, A{});
}

template<class B>
B minusinfinity() {
  using T = typename B::value_type;
  return B(-std::numeric_limits<T>::infinity());
}

template<class T, class Tp>
auto mod(T const& self, Tp const& other) -> decltype(self % other){
  return self % other;
}

template<class T, class Tp>
auto mul(T const& self, Tp const& other) -> decltype(self * other){
  return self * other;
}

template<class T, class A>
batch<T, A> nearbyint(batch<T, A> const& self) {
  return kernel::nearbyint<A>(self, A{});
}

template<class T, class A>
batch_bool<T, A> neq(batch<T, A> const& self, batch<T, A> const& other) {
  return self != other;
}

template<class T, class A>
batch<T, A> neg(batch<T, A> const& self) {
  return -self;
}

template<class T, class A>
batch<T, A> nextafter(batch<T, A> const& from, batch<T, A> const& to) {
  return kernel::nextafter<A>(from, to, A{});
}

template<class A, class T>
batch<T, A> norm(batch<std::complex<T>, A> const& self) {
  return kernel::norm(self, A{});
}

template<class T, class A>
batch<T, A> pos(batch<T, A> const& self) {
  return +self;
}

template<class T, class A>
batch<T, A> pow(batch<T, A> const& self, batch<T, A> const& other) {
  return kernel::pow<A>(self, other, A{});
}

template<class T, class ITy, class A, class=typename std::enable_if<std::is_integral<ITy>::value, void>::type>
batch<T, A> pow(batch<T, A> const& self, ITy other) {
  return kernel::ipow<A>(self, other, A{});
}

template<class A, class T>
batch<std::complex<T>, A> proj(batch<std::complex<T>, A> const& self) {
  return kernel::proj(self, A{});
}

template<class T, class A>
batch<T, A> remainder(batch<T, A> const& self, batch<T, A> const& other) {
  return kernel::remainder<A>(self, other, A{});
}

template<class T, class A>
batch<T, A> rint(batch<T, A> const& self) {
  return nearbyint(self);
}

template<class T, class A>
batch<T, A> round(batch<T, A> const& self) {
  return kernel::round<A>(self, A{});
}

template<class T, class Tp>
auto sadd(T const& self, Tp const& other) -> decltype(self + other) {
  using B = decltype(self + other);
  using A = typename B::arch_type;
  return kernel::sadd<A>(B(self), B(other), A{});
}

template<class T, class A>
batch<T, A> select(batch_bool<T, A> const& cond, batch<T, A> const& true_br, batch<T, A> const& false_br) {
  return kernel::select<A>(cond, true_br, false_br, A{});
}

template<class T, class A>
batch<std::complex<T>, A> select(batch_bool<T, A> const& cond, batch<std::complex<T>, A> const& true_br, batch<std::complex<T>, A> const& false_br) {
  return kernel::select<A>(cond, true_br, false_br, A{});
}

template<class T, class A, bool... Values>
batch<T, A> select(batch_bool_constant<batch<T, A>, Values...> const& cond, batch<T, A> const& true_br, batch<T, A> const& false_br) {
  return kernel::select<A>(cond, true_br, false_br, A{});
}

template<class T, class A>
batch<T, A> sign(batch<T, A> const& self) {
  return kernel::sign<A>(self, A{});
}

template<class T, class A>
batch<T, A> signnz(batch<T, A> const& self) {
  return kernel::signnz<A>(self, A{});
}

template<class T, class A>
batch<T, A> sin(batch<T, A> const& self) {
  return kernel::sin<A>(self, A{});
}

template<class T, class A>
batch<T, A> sinh(batch<T, A> const& self) {
  return kernel::sinh<A>(self, A{});
}

template<class T, class A>
std::pair<batch<T, A>, batch<T, A>> sincos(batch<T, A> const& self) {
  return kernel::sincos<A>(self, A{});
}

template<class T, class A>
batch<T, A> sqrt(batch<T, A> const& self) {
  return kernel::sqrt<A>(self, A{});
}

template<class T, class Tp>
auto ssub(T const& self, Tp const& other) -> decltype(self - other) {
  using B = decltype(self + other);
  using A = typename B::arch_type;
  return kernel::ssub<A>(B(self), B(other), A{});
}

template<class To, class A, class From>
void store(From* mem, batch<To, A> const& val, aligned_mode={}) {
  return kernel::store_aligned<A>(mem, val, A{});
}

template<class To, class A, class From>
void store(To* mem, batch<From, A> const& val, unaligned_mode) {
  return kernel::store_unaligned<A>(mem, val, A{});
}

template<class To, class A, class From>
void store_aligned(To* mem, batch<From, A> const& val) {
  return kernel::store_aligned<A>(mem, val, A{});
}

template<class To, class A, class From>
void store_unaligned(To* mem, batch<From, A> const& val) {
  return kernel::store_unaligned<A>(mem, val, A{});
}

template<class T, class Tp>
auto sub(T const& self, Tp const& other) -> decltype(self - other){
  return self - other;
}

template<class T, class A>
batch<T, A> tan(batch<T, A> const& self) {
  return kernel::tan<A>(self, A{});
}

template<class T, class A>
batch<T, A> tanh(batch<T, A> const& self) {
  return kernel::tanh<A>(self, A{});
}

template<class T, class A>
batch<T, A> tgamma(batch<T, A> const& self) {
  return kernel::tgamma<A>(self, A{});
}

template<class T, class A>
batch<as_float_t<T>, A> to_float(batch<T, A> const& self) {
  return kernel::to_float<A>(self, A{});
}

template<class T, class A>
batch<as_integer_t<T>, A> to_int(batch<T, A> const& self) {
  return kernel::to_int<A>(self, A{});
}

template<class T, class A>
batch<T, A> trunc(batch<T, A> const& self) {
  return kernel::trunc<A>(self, A{});
}

template<class T, class A>
batch<T, A> zip_hi(batch<T, A> const& self, batch<T, A> const& other) {
  return kernel::zip_hi<A>(self, other, A{});
}

template<class T, class A>
batch<T, A> zip_lo(batch<T, A> const& self, batch<T, A> const& other) {
  return kernel::zip_lo<A>(self, other, A{});
}

// high level functions - batch_bool
//

template<class T, class A>
bool all(batch_bool<T, A> const& self) {
  return kernel::all<A>(self, A{});
}

template<class T, class A>
bool any(batch_bool<T, A> const& self) {
  return kernel::any<A>(self, A{});
}

}

#endif

