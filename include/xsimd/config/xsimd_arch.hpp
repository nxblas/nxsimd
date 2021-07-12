/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay, Wolf Vollprecht and         *
* Martin Renou                                                             *
* Copyright (c) QuantStack                                                 *
* Copyright (c) Serge Guelton                                              *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_ARCH_HPP
#define XSIMD_ARCH_HPP

#include "xsimd_instruction_set.hpp"

#include <algorithm>

// header for runtime architecture detection {
#if defined(__ARM_NEON)
#include <sys/auxv.h>
#include <asm/hwcap.h>
#endif

#ifdef _MSC_VER
    #include <intrin.h>
#elif defined(__INTEL_COMPILER)
    #include <immintrin.h>
#endif


// }

#include <tuple>


namespace xsimd
{

    // forward declaration
    template<class T, size_t N>
    class batch;

    namespace arch
    {
        namespace detail {

          struct supported_arch
          {
              unsigned sse : 1;
              unsigned sse2 : 1;
              unsigned sse3 : 1;
              unsigned ssse3 : 1;
              unsigned sse4_1 : 1;
              unsigned sse4_2 : 1;
              unsigned sse4a : 1;
              unsigned fma3 : 1;
              unsigned fma4 : 1;
              unsigned xop : 1;
              unsigned avx : 1;
              unsigned avx2 : 1;
              unsigned avx512 : 1;
              unsigned neon : 1;
              unsigned neon64 : 1;

              // version number of the best arch available
              unsigned best_version;

              supported_arch()
              {
                   memset(this, 0, sizeof(supported_arch));

#if defined(__aarch64__) || defined(_M_ARM64)
                   // neon is required on AArch64
                   neon = 1;
                   neon64 = 1;
                   best_version = XSIMD_ARM8_64_NEON_VERSION;

#elif defined(__ARM_NEON) || defined(_M_ARM)
                   neon = bool(getauxval(AT_HWCAP) & HWCAP_NEON);
                   best_version = XSIMD_ARM7_NEON_VERSION * neon;

#elif defined(__x86_64__) || defined(__i386__) || defined(_M_AMD64) || defined(_M_IX86)
                   auto get_cpuid = [](int reg[4], int func_id)
                   {

#if defined(_MSC_VER)
                       __cpuidex(reg, func_id, 0);

#elif defined(__INTEL_COMPILER)
                       __cpuid(reg, func_id);

#elif defined(__GNUC__) || defined(__clang__)

#if defined( __i386__ ) && defined(__PIC__)
                       // %ebx may be the PIC register
                       __asm__("xchg{l}\t{%%}ebx, %1\n\t"
                               "cpuid\n\t"
                               "xchg{l}\t{%%}ebx, %1\n\t"
                               : "=a" (reg[0]), "=r" (reg[1]), "=c" (reg[2]),
                                 "=d" (reg[3])
                               : "a" (func_id), "c" (0)
                       );

#else
                       __asm__("cpuid\n\t"
                               : "=a" (reg[0]), "=b" (reg[1]), "=c" (reg[2]),
                                 "=d" (reg[3])
                               : "a" (func_id), "c" (0)
                       );
#endif

#else
#error "Unsupported configuration"
#endif
                   };

                   int regs[4];

                   get_cpuid(regs, 0x1);
                   sse = regs[3]>> 25 & 1;
                   best_version = std::max(best_version, XSIMD_X86_SSE_VERSION * sse);

                   sse2 = regs[2] >> 26 & 1;
                   best_version = std::max(best_version, XSIMD_X86_SSE2_VERSION * sse2);

                   sse3 = regs[2] >> 0 & 1;
                   best_version = std::max(best_version, XSIMD_X86_SSE3_VERSION * sse3);

                   ssse3 = regs[2] >> 9 & 1;
                   best_version = std::max(best_version, XSIMD_X86_SSSE3_VERSION * ssse3);

                   sse4_1 = regs[2] >> 19 & 1;
                   best_version = std::max(best_version, XSIMD_X86_SSE4_1_VERSION * sse4_1);

                   sse4_2 = regs[2] >> 20 & 1;
                   best_version = std::max(best_version, XSIMD_X86_SSE4_2_VERSION * sse4_2);

                   sse4a = regs[2] >> 6 & 1;
                   best_version = std::max(best_version, XSIMD_X86_AMD_SSE4A_VERSION * sse4a);

                   xop = regs[2] >> 11 & 1;
                   best_version = std::max(best_version, XSIMD_X86_AMD_XOP_VERSION * xop);

                   avx = regs[2] >> 28 & 1;
                   best_version = std::max(best_version, XSIMD_X86_AVX_VERSION * avx);

                   fma3 = regs[2] >> 12 & 1;
                   best_version = std::max(best_version, XSIMD_X86_FMA3_VERSION * fma3);

                   get_cpuid(regs, 0x7);
                   avx2 = regs[1] >> 5 & 1;
                   best_version = std::max(best_version, XSIMD_X86_AVX2_VERSION * avx2);

                   avx512 = regs[1] >> 16 & 1; // actually avx512f
                   best_version = std::max(best_version, XSIMD_X86_AVX512_VERSION * avx512);

                   get_cpuid(regs, 0x80000001);
                   fma4 = regs[2] >> 16 & 1;
                   best_version = std::max(best_version, XSIMD_X86_AMD_FMA4_VERSION * fma4);
#endif
              }
          };

          inline supported_arch available()
          {
              static supported_arch supported;
              return supported;
          }
        }

        struct sse
        {
            static const unsigned version = XSIMD_X86_SSE_VERSION;
            static constexpr bool supported = XSIMD_X86_INSTR_SET >= version;
            static bool available() { return detail::available().sse; }

            template<class T>
            using batch = xsimd::batch<T, 128 / ( 8 * sizeof(T))>;
            static constexpr size_t alignment = 8;
        };

        struct sse2 : sse
        {
            static const unsigned version = XSIMD_X86_SSE2_VERSION;
            static constexpr bool supported = XSIMD_X86_INSTR_SET >= version;
            static bool available() { return detail::available().sse2; }

            static constexpr size_t alignment = 16;
        };

        struct sse3 : sse2
        {
            static const unsigned version = XSIMD_X86_SSE3_VERSION;
            static constexpr bool supported = XSIMD_X86_INSTR_SET >= version;
            static bool available() { return detail::available().sse3; }
        };

        struct ssse3 : sse2
        {
            static const unsigned version = XSIMD_X86_SSSE3_VERSION;
            static constexpr bool supported = XSIMD_X86_INSTR_SET >= version;
            static bool available() { return detail::available().ssse3; }
        };

        // Intel specific
        struct sse4_1 : sse3
        {
            static const unsigned version = XSIMD_X86_SSE4_1_VERSION;
            static constexpr bool supported = XSIMD_X86_INSTR_SET >= version;
            static bool available() { return detail::available().sse4_1; }
        };

        // Intel specific
        struct sse4_2 : sse4_1
        {
            static const unsigned version = XSIMD_X86_SSE4_2_VERSION;
            static constexpr bool supported = XSIMD_X86_INSTR_SET >= version;
            static bool available() { return detail::available().sse4_2; }
        };

        // AMD specific
        struct sse4a : sse3
        {
            static const unsigned version = XSIMD_X86_AMD_SSE4A_VERSION;
            static constexpr bool supported = XSIMD_X86_AMD_INSTR_SET >= version;
            static bool available() { return detail::available().sse4a; }
        };

        struct avx
        {
            static const unsigned version = XSIMD_X86_AVX_VERSION;
            static constexpr bool supported = XSIMD_X86_INSTR_SET >= version;
            static bool available() { return detail::available().avx; }

            template<class T>
            using batch = xsimd::batch<T, 256 / ( 8 * sizeof(T))>;
            static constexpr size_t alignment = 32;
        };

        struct fma3 : avx
        {
            static const unsigned version = XSIMD_X86_FMA3_VERSION;
            static constexpr bool supported = XSIMD_X86_INSTR_SET >= version;
            static bool available() { return detail::available().fma3; }
        };

        // AMD specific (very few old processors)
        struct fma4 : avx
        {
            static const unsigned version = XSIMD_X86_AMD_FMA4_VERSION;
            static constexpr bool supported = XSIMD_X86_AMD_INSTR_SET >= version;
            static bool available() { return detail::available().fma4; }
        };

        // AMD specific (very few old processors)
        struct xop : fma4
        {
            static const unsigned version = XSIMD_X86_AMD_XOP_VERSION;
            static constexpr bool supported = XSIMD_X86_AMD_INSTR_SET >= version;
            static bool available() { return detail::available().xop; }
        };

        struct avx2 : avx
        {
            static const unsigned version = XSIMD_X86_AVX2_VERSION;
            static constexpr bool supported = XSIMD_X86_INSTR_SET >= version;
            static bool available() { return detail::available().avx2; }
        };

        struct avx512
        {
            static const unsigned version = XSIMD_X86_AVX512_VERSION;
            static constexpr bool supported = XSIMD_X86_INSTR_SET >= version;
            static bool available() { return detail::available().avx512; }

            template<class T>
            using batch = xsimd::batch<T, 512 / ( 8 * sizeof(T))>;
            static constexpr size_t alignment = 64;
        };

        struct neon
        {
            static const unsigned version = XSIMD_ARM7_NEON_VERSION;
            static constexpr bool supported = XSIMD_ARM_INSTR_SET >= version;
            static bool available() { return detail::available().neon; }

            template<class T>
            using batch = xsimd::batch<T, 128 / ( 8 * sizeof(T))>;
            static constexpr size_t alignment = 16;
        };

        struct neon64
        {
            static const unsigned version = XSIMD_ARM8_64_NEON_VERSION;
            static constexpr bool supported = XSIMD_ARM_INSTR_SET >= version;
            static bool available() { return detail::available().neon64; }

            template<class T>
            using batch = xsimd::batch<T, 128 / ( 8 * sizeof(T))>;
            static constexpr size_t alignment = 16;
        };

        struct scalar
        {
            static const unsigned version = XSIMD_VERSION_NUMBER_AVAILABLE;
#ifdef XSIMD_ENABLE_FALLBACK
            static constexpr bool supported = true;
#else
            static constexpr bool supported = false;
#endif
            static constexpr bool available() { return true; }

            template<class T>
            using batch = xsimd::batch<T, 1>;
            static constexpr size_t alignment = sizeof(void*);
        };

        struct unavailable
        {
            static const unsigned version = XSIMD_VERSION_NUMBER_NOT_AVAILABLE;
            static constexpr bool supported = false;
            static constexpr bool available() { return false; }

            template<class T>
            using batch = xsimd::batch<T, 1>;
            static constexpr size_t alignment = sizeof(void*);
        };

        namespace detail
        {
            // Checks whether T appears in Tys.
            template<class T, class... Tys>
            struct contains;

            template<class T>
            struct contains<T> : std::false_type
            {
            };

            template<class T, class Ty, class... Tys>
            struct contains<T, Ty, Tys...> : std::conditional<std::is_same<Ty, T>::value, std::true_type, contains<T, Tys...>>::type
            {
            };

            template<class... Archs>
            struct is_sorted;

            template<>
            struct is_sorted<> : std::true_type
            {
            };

            template<class Arch>
            struct is_sorted<Arch> : std::true_type
            {
            };

            template<class A0, class A1, class... Archs>
            struct is_sorted<A0, A1, Archs...> : std::conditional<(A0::version >= A1::version), is_sorted<Archs...>, std::false_type>::type
            {
            };

        }


        // An arch_list is a list of architectures, sorted by version number.
        template<class... Archs>
        struct arch_list {
#ifndef NDEBUG
          static_assert(detail::is_sorted<Archs...>::value, "architecture list must be sorted by version");
#endif

          template<class Arch>
          using add = arch_list<Archs..., Arch>;

          template<class... OtherArchs>
          using extend = arch_list<Archs..., OtherArchs...>;

          template<class Arch>
          static constexpr bool contains()
          {
              return detail::contains<Arch, Archs...>::value;
          }

          template<class F>
          static void for_each(F&& f)
          {
              (void)std::initializer_list<bool>{(f(Archs{}), true)...};
          }
        };


        namespace detail
        {
            // Pick the best architecture in arch_list L, which is the last
            // because architectures are sorted by version.
            template<class L>
            struct best;

            template<>
            struct best<arch_list<>>
            {
                using type = unavailable;
            };

            template<class Arch, class... Archs>
            struct best<arch_list<Arch, Archs...>>
            {
                using type = Arch;
            };

            // Filter archlists Archs, picking only supported archs and adding
            // them to L.
            template<class L, class... Archs>
            struct supported_helper;

            template<class L>
            struct supported_helper<L, arch_list<>>
            {
                using type = L;
            };

            template<class L, class Arch, class... Archs>
            struct supported_helper<L, arch_list<Arch, Archs...>> : supported_helper<typename std::conditional<Arch::supported, typename L::template add<Arch>, L >::type, arch_list<Archs...>>
            {
            };

            template<class... Archs>
            struct supported : supported_helper<arch_list<>, Archs...>
            {
            };

            // Joins all arch_list Archs in a single arch_list.
            template<class... Archs>
            struct join;

            template<class Arch>
            struct join<Arch>
            {
                using type = Arch;
            };

            template<class Arch, class... Archs, class... Args>
            struct join<Arch, arch_list<Archs...>, Args...> : join<typename Arch::template extend<Archs...>, Args...>
            {
            };
        }

        using all_x86 = arch_list<avx512, avx2, fma3, xop, fma4, avx, sse4_2, sse4_1, sse4a, ssse3, sse3, sse2, sse>;
        using all_arm = arch_list<neon64, neon>;
        using all = typename detail::join<all_arm, all_x86, arch_list<scalar>>::type;

        using supported = typename detail::supported<all>::type;

        using x86 = typename detail::best<typename detail::supported<all_x86>::type>::type;
        using arm = typename detail::best<typename detail::supported<all_arm>::type>::type;

        using default_ = typename detail::best<typename detail::supported<arch_list<arm, x86, scalar>>::type>::type;

        // Generic, arch-dependent batch type.
        template <class T, class InstructionSet>
        using batch = typename InstructionSet::template batch<T>;

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
                    static_assert(Arch::supported, "dispatching on supported architecture");
                    assert(Arch::available() && "At least one arch must be supported during dispatch");
                    return functor(Arch{}, std::forward<Tys>(args)...);
                }

                template<class Arch, class ArchNext, class... Archs, class... Tys>
                auto walk_archs(arch_list<Arch, ArchNext, Archs...>, Tys&&... args) -> decltype(functor(Arch{}, std::forward<Tys>(args)...))
                {
                    static_assert(Arch::supported, "dispatching on supported architecture");
                    if(Arch::version == best_arch)
                      return functor(Arch{}, std::forward<Tys>(args)...);
                    else
                      return walk_archs(arch_list<ArchNext, Archs...>{}, std::forward<Tys>(args)...);
                }

                public:

                dispatcher(F f) : best_arch(::xsimd::arch::detail::available().best_version), functor(f)
                {
                }

                template<class... Tys>
                auto operator()(Tys&&... args) -> decltype(functor(default_{}, std::forward<Tys>(args)...))
                {
                    return walk_archs(ArchList{}, std::forward<Tys>(args)...);
                }
            };
        }

        // Generic function dispatch, à la ifunc
        template<class F, class ArchList=::xsimd::arch::supported>
        inline detail::dispatcher<F, ArchList> dispatch(F&& f)
        {
            return {std::forward<F>(f)};
        }

    }
}

#endif
