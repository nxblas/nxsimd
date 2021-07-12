#ifndef XSIMD_ARCH_HPP
#define XSIMD_ARCH_HPP

#include "./xsimd_config.hpp"
#include "../types/xsimd_all_registers.hpp"
#include "./xsimd_cpuid.hpp"

#include <type_traits>

namespace xsimd {

  namespace detail {
  // Checks whether T appears in Tys.
  template <class T, class... Tys> struct contains;

  template <class T> struct contains<T> : std::false_type {};

  template <class T, class Ty, class... Tys>
  struct contains<T, Ty, Tys...>
      : std::conditional<std::is_same<Ty, T>::value, std::true_type,
                         contains<T, Tys...>>::type {};

  template <class... Archs> struct is_sorted;

  template <> struct is_sorted<> : std::true_type {};

  template <class Arch> struct is_sorted<Arch> : std::true_type {};

  template <class A0, class A1, class... Archs>
  struct is_sorted<A0, A1, Archs...>
      : std::conditional<(A0::version() >= A1::version()), is_sorted<Archs...>,
                         std::false_type>::type {};

  } // namespace detail

  // An arch_list is a list of architectures, sorted by version number.
  template <class... Archs> struct arch_list {
#ifndef NDEBUG
    static_assert(detail::is_sorted<Archs...>::value,
                  "architecture list must be sorted by version");
#endif

    template <class Arch> using add = arch_list<Archs..., Arch>;

    template <class... OtherArchs>
    using extend = arch_list<Archs..., OtherArchs...>;

    template <class Arch> static constexpr bool contains() {
      return detail::contains<Arch, Archs...>::value;
    }

    template <class F> static void for_each(F &&f) {
      (void)std::initializer_list<bool>{(f(Archs{}), true)...};
    }
  };

  struct unavailable {};

  namespace detail {
    // Pick the best architecture in arch_list L, which is the last
    // because architectures are sorted by version.
    template <class L> struct best;

    template <> struct best<arch_list<>> { using type = unavailable; };

    template <class Arch, class... Archs> struct best<arch_list<Arch, Archs...>> {
      using type = Arch;
    };

    // Filter archlists Archs, picking only supported archs and adding
    // them to L.
    template <class L, class... Archs> struct supported_helper;

    template <class L> struct supported_helper<L, arch_list<>> { using type = L; };

    template <class L, class Arch, class... Archs>
    struct supported_helper<L, arch_list<Arch, Archs...>>
        : supported_helper<
              typename std::conditional<Arch::supported(),
                                        typename L::template add<Arch>, L>::type,
              arch_list<Archs...>> {};

    template <class... Archs>
    struct supported : supported_helper<arch_list<>, Archs...> {};

    // Joins all arch_list Archs in a single arch_list.
    template <class... Archs> struct join;

    template <class Arch> struct join<Arch> { using type = Arch; };

    template <class Arch, class... Archs, class... Args>
    struct join<Arch, arch_list<Archs...>, Args...>
        : join<typename Arch::template extend<Archs...>, Args...> {};
  } // namespace detail

  struct unsupported {};
  using all_x86_architectures = arch_list<avx512f, avx2, /*fma3, xop, fma4,*/ avx, sse4_2, sse4_1, /*sse4a, ssse3,*/ sse3, sse2, sse>;
  using all_arm_architectures = arch_list<arm8_64, arm8_32, arm7>;
  using all_architectures = typename detail::join<all_arm_architectures, all_x86_architectures>::type;

  using supported_architectures = typename detail::supported<all_architectures>::type;

  using x86_arch = typename detail::best<typename detail::supported<all_x86_architectures>::type>::type;
  using arm_arch = typename detail::best<typename detail::supported<all_arm_architectures>::type>::type;
  //using default_arch = typename detail::best<typename detail::supported<arch_list</*arm_arch,*/ x86_arch>>::type>::type;
  using default_arch = typename std::conditional<std::is_same<x86_arch, unavailable>::value,
                                                 arm_arch,
                                                 x86_arch>::type;


    namespace detail
    {
        template<class F, class ArchList>
        class dispatcher
        {

            const unsigned best_arch;
            F functor;

            template<class Arch, class... Tys>
            auto walk_archs(arch_list<Arch>, Tys&&... args) -> decltype(functor(Arch{}, std::forward<Tys>(args)...))
            {
                static_assert(Arch::supported(), "dispatching on supported architecture");
                assert(Arch::available() && "At least one arch must be supported during dispatch");
                return functor(Arch{}, std::forward<Tys>(args)...);
            }

            template<class Arch, class ArchNext, class... Archs, class... Tys>
            auto walk_archs(arch_list<Arch, ArchNext, Archs...>, Tys&&... args) -> decltype(functor(Arch{}, std::forward<Tys>(args)...))
            {
                static_assert(Arch::supported(), "dispatching on supported architecture");
                if(Arch::version() == best_arch)
                  return functor(Arch{}, std::forward<Tys>(args)...);
                else
                  return walk_archs(arch_list<ArchNext, Archs...>{}, std::forward<Tys>(args)...);
            }

            public:

            dispatcher(F f) : best_arch(available_architectures().best), functor(f)
            {
            }

            template<class... Tys>
            auto operator()(Tys&&... args) -> decltype(functor(default_arch{}, std::forward<Tys>(args)...))
            {
                return walk_archs(ArchList{}, std::forward<Tys>(args)...);
            }
        };
    }

    // Generic function dispatch, à la ifunc
    template<class F, class ArchList=supported_architectures>
    inline detail::dispatcher<F, ArchList> dispatch(F&& f)
    {
        return {std::forward<F>(f)};
    }

} // namespace xsimd

#endif

