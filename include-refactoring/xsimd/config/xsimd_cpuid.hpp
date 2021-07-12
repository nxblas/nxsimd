#ifndef XSIMD_CPUID_HPP
#define XSIMD_CPUID_HPP

#include <cstring>
#include <algorithm>
#include <array>

#include "../types/xsimd_all_registers.hpp"

namespace xsimd
{
    namespace detail
    {
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
            unsigned arm7 : 1;
            unsigned arm8_32 : 1;
            unsigned arm8_64 : 1;

            // version number of the best arch available
            unsigned best;

            supported_arch()
            {
                 memset(this, 0, sizeof(supported_arch));

#if defined(__aarch64__) || defined(_M_ARM64)
                 arm7 = 1;
                 arm8_32 = 1;
                 arm8_64 = 1;
                 best = arm8_64::version();
#elif defined(__ARM_NEON)
#if __ARM_ARCH >= 8
                 arm7 = 1;
                 arm8_32 = 1;
                 arm8_64 = 0;
                 best = arm8_32::version();
#else
                 arm7 = 1;
                 arm8_32 = 0;
                 arm8_64 = 0;
                 best = arm7::version();
#endif
//#elif defined(__ARM_NEON) || defined(_M_ARM)
//                 neon = bool(getauxval(AT_HWCAP) & HWCAP_NEON);
//                 best = neon::version() * neon;

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
                 best = std::max(best, sse::version() * sse);

                 sse2 = regs[2] >> 26 & 1;
                 best = std::max(best, sse2::version() * sse2);

                 sse3 = regs[2] >> 0 & 1;
                 best = std::max(best, sse3::version() * sse3);

                 //ssse3 = regs[2] >> 9 & 1;
                 //best = std::max(best, ssse3::version() * ssse3);

                 sse4_1 = regs[2] >> 19 & 1;
                 best = std::max(best, sse4_1::version() * sse4_1);

                 sse4_2 = regs[2] >> 20 & 1;
                 best = std::max(best, sse4_2::version() * sse4_2);

                 //sse4a = regs[2] >> 6 & 1;
                 //best = std::max(best, XSIMD_X86_AMD_SSE4A_VERSION * sse4a);

                 //xop = regs[2] >> 11 & 1;
                 //best = std::max(best, XSIMD_X86_AMD_XOP_VERSION * xop);

                 avx = regs[2] >> 28 & 1;
                 best = std::max(best, avx::version() * avx);

                 //fma3 = regs[2] >> 12 & 1;
                 //best = std::max(best, XSIMD_X86_FMA3_VERSION * fma3);

                 get_cpuid(regs, 0x7);
                 avx2 = regs[1] >> 5 & 1;
                 best = std::max(best, avx2::version() * avx2);

                 avx512 = regs[1] >> 16 & 1; // actually avx512f
                 best = std::max(best, avx512f::version() * avx512);

                 //get_cpuid(regs, 0x80000001);
                 //fma4 = regs[2] >> 16 & 1;
                 //best = std::max(best, XSIMD_X86_AMD_FMA4_VERSION * fma4);
#endif
            }
        };
    }

    inline detail::supported_arch available_architectures()
    {
        static detail::supported_arch supported;
        return supported;
    }
}

#endif
