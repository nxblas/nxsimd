/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay, Wolf Vollprecht and         *
* Martin Renou                                                             *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/
#define DOCTEST_CONFIG_IMPLEMENT

#if defined(XSIMD_NO_EXCEPTIONS)
    #define DOCTEST_CONFIG_NO_EXCEPTIONS
#endif
#include "doctest/doctest.h"

#include <fstream>
#include <iostream>
#include <map>
#include <string>

#include "xsimd/config/xsimd_instruction_set.hpp"

using info_map_type = std::map<int, std::string>;

info_map_type init_instruction_map()
{
    info_map_type res;
#ifdef XSIMD_X86_INSTR_SET_AVAILABLE
    res[XSIMD_X86_SSE_VERSION] = "Intel SSE";
    res[XSIMD_X86_SSE2_VERSION] = "Intel SSE2"; 
    res[XSIMD_X86_SSE3_VERSION] = "Intel SSE3";
    res[XSIMD_X86_SSSE3_VERSION] = "Intel SSSE3";
    res[XSIMD_X86_SSE4_1_VERSION] = "Intel SSE4.1";
    res[XSIMD_X86_SSE4_2_VERSION] = "Intel SSE4.2";
    res[XSIMD_X86_AVX_VERSION] = "Intel AVX";
    res[XSIMD_X86_AVX512_VERSION] = "Intel AVX 512";
    res[XSIMD_X86_FMA3_VERSION] = "Intel FMA3";
    res[XSIMD_X86_AVX2_VERSION] = "Intel AVX2";
    res[XSIMD_X86_MIC_VERSION] = "Intel MIC";
    res[XSIMD_X86_AMD_SSE4A_VERSION] = "AMD SSE4A";
    res[XSIMD_X86_AMD_FMA4_VERSION] = "AMD FMA4";
    res[XSIMD_X86_AMD_XOP_VERSION] = "AMD XOP";
#elif defined(XSIMD_PPC_INSTR_SET_AVAILABLE) 
    res[XSIMD_PPC_VMX_VERSION] = "PowerPC VM";
    res[XSIMD_PPC_VSX_VERSION] = "PowerPC VSX";
    res[XSIMD_PPC_QPX_VERSION] = "PowerPC QPX";
#else
    res[XSIMD_ARM7_NEON_VERSION] = "ARMv7 Neon";
    res[XSIMD_ARM8_32_NEON_VERSION] = "ARMv8 32bit Neon";
    res[XSIMD_ARM8_64_NEON_VERSION] = "ARMv8 64bit Neon";
    res[XSIMD_VERSION_NUMBER_NOT_AVAILABLE] = "No SIMD available";
#endif
    return res;
}

std::string get_instruction_set_name()
{
    static info_map_type info_map(init_instruction_map());
    return info_map[XSIMD_INSTR_SET];
}

int main(int , char** )
{
    std::ofstream out("log/xsimd_info.log", std::ios_base::out);
    std::string instruction_set = get_instruction_set_name();
    out << "Instruction set: " << instruction_set << std::endl;
    std::cout << "Instruction set: " << instruction_set << std::endl;
    doctest::Context context;
    int res = context.run(); // run
    return res;
}
