############################################################################
# Copyright (c) Johan Mabille, Sylvain Corlay, Wolf Vollprecht and         #
# Martin Renou                                                             #
# Copyright (c) QuantStack                                                 #
#                                                                          #
# Distributed under the terms of the BSD 3-Clause License.                 #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

cmake_minimum_required(VERSION 3.1)
project(xsimd)
option(XSIMD_REFACTORING ON)

if(XSIMD_REFACTORING)
set(XSIMD_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include-refactoring)
else()
set(XSIMD_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
endif()

# Versionning
# ===========

file(STRINGS "${XSIMD_INCLUDE_DIR}/xsimd/config/xsimd_config.hpp" xsimd_version_defines
     REGEX "#define XSIMD_VERSION_(MAJOR|MINOR|PATCH)")
foreach(ver ${xsimd_version_defines})
    if(ver MATCHES "#define XSIMD_VERSION_(MAJOR|MINOR|PATCH) +([^ ]+)$")
        set(XSIMD_VERSION_${CMAKE_MATCH_1} "${CMAKE_MATCH_2}" CACHE INTERNAL "")
    endif()
endforeach()
set(${PROJECT_NAME}_VERSION
    ${XSIMD_VERSION_MAJOR}.${XSIMD_VERSION_MINOR}.${XSIMD_VERSION_PATCH})
message(STATUS "xsimd v${${PROJECT_NAME}_VERSION}")

# Build
# =====

if(XSIMD_REFACTORING)
set(XSIMD_HEADERS
${XSIMD_INCLUDE_DIR}/xsimd/arch/xsimd_constants.hpp
${XSIMD_INCLUDE_DIR}/xsimd/arch/xsimd_fma3.hpp
${XSIMD_INCLUDE_DIR}/xsimd/arch/xsimd_fma4.hpp
${XSIMD_INCLUDE_DIR}/xsimd/arch/xsimd_generic.hpp
${XSIMD_INCLUDE_DIR}/xsimd/arch/xsimd_isa.hpp
${XSIMD_INCLUDE_DIR}/xsimd/arch/xsimd_scalar.hpp
${XSIMD_INCLUDE_DIR}/xsimd/arch/xsimd_sse.hpp
${XSIMD_INCLUDE_DIR}/xsimd/arch/xsimd_sse2.hpp
${XSIMD_INCLUDE_DIR}/xsimd/arch/xsimd_sse3.hpp
${XSIMD_INCLUDE_DIR}/xsimd/arch/xsimd_sse4_1.hpp
${XSIMD_INCLUDE_DIR}/xsimd/arch/xsimd_sse4_2.hpp
${XSIMD_INCLUDE_DIR}/xsimd/arch/xsimd_ssse3.hpp
${XSIMD_INCLUDE_DIR}/xsimd/config/xsimd_arch.hpp
${XSIMD_INCLUDE_DIR}/xsimd/config/xsimd_config.hpp
${XSIMD_INCLUDE_DIR}/xsimd/config/xsimd_cpuid.hpp
${XSIMD_INCLUDE_DIR}/xsimd/memory/xsimd_aligned_allocator.hpp
${XSIMD_INCLUDE_DIR}/xsimd/memory/xsimd_alignment.hpp
${XSIMD_INCLUDE_DIR}/xsimd/stl/algorithms.hpp
${XSIMD_INCLUDE_DIR}/xsimd/types/xsimd_all_registers.hpp
${XSIMD_INCLUDE_DIR}/xsimd/types/xsimd_api.hpp
${XSIMD_INCLUDE_DIR}/xsimd/types/xsimd_arm7_register.hpp
${XSIMD_INCLUDE_DIR}/xsimd/types/xsimd_arm8_32_register.hpp
${XSIMD_INCLUDE_DIR}/xsimd/types/xsimd_arm8_64_register.hpp
${XSIMD_INCLUDE_DIR}/xsimd/types/xsimd_avx2_register.hpp
${XSIMD_INCLUDE_DIR}/xsimd/types/xsimd_avx512f_register.hpp
${XSIMD_INCLUDE_DIR}/xsimd/types/xsimd_avx_register.hpp
${XSIMD_INCLUDE_DIR}/xsimd/types/xsimd_batch.hpp
${XSIMD_INCLUDE_DIR}/xsimd/types/xsimd_batch_constant.hpp
${XSIMD_INCLUDE_DIR}/xsimd/types/xsimd_generic_arch.hpp
${XSIMD_INCLUDE_DIR}/xsimd/types/xsimd_register.hpp
${XSIMD_INCLUDE_DIR}/xsimd/types/xsimd_sse2_register.hpp
${XSIMD_INCLUDE_DIR}/xsimd/types/xsimd_sse3_register.hpp
${XSIMD_INCLUDE_DIR}/xsimd/types/xsimd_sse4_1_register.hpp
${XSIMD_INCLUDE_DIR}/xsimd/types/xsimd_sse4_2_register.hpp
${XSIMD_INCLUDE_DIR}/xsimd/types/xsimd_sse_register.hpp
${XSIMD_INCLUDE_DIR}/xsimd/types/xsimd_ssse3_register.hpp
${XSIMD_INCLUDE_DIR}/xsimd/types/xsimd_utils.hpp
${XSIMD_INCLUDE_DIR}/xsimd/xsimd.hpp
)

else()
set(XSIMD_HEADERS
    ${XSIMD_INCLUDE_DIR}/xsimd/xsimd.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/config/xsimd_align.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/config/xsimd_config.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/config/xsimd_include.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/config/xsimd_instruction_set.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/math/xsimd_basic_math.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/math/xsimd_error.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/math/xsimd_exp_reduction.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/math/xsimd_exponential.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/math/xsimd_fp_manipulation.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/math/xsimd_fp_sign.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/math/xsimd_gamma.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/math/xsimd_horner.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/math/xsimd_hyperbolic.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/math/xsimd_invtrigo.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/math/xsimd_logarithm.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/math/xsimd_math.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/math/xsimd_math_complex.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/math/xsimd_numerical_constant.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/math/xsimd_power.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/math/xsimd_rem_pio2.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/math/xsimd_rounding.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/math/xsimd_trigo_reduction.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/math/xsimd_trigonometric.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/memory/xsimd_aligned_allocator.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/memory/xsimd_aligned_stack_buffer.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/memory/xsimd_alignment.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/types/xsimd_avx_complex.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/types/xsimd_avx_conversion.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/types/xsimd_avx_double.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/types/xsimd_avx_float.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/types/xsimd_avx_int8.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/types/xsimd_avx_int16.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/types/xsimd_avx_int32.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/types/xsimd_avx_int64.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/types/xsimd_avx_int_base.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/types/xsimd_avx512_bool.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/types/xsimd_avx512_complex.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/types/xsimd_avx512_conversion.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/types/xsimd_avx512_double.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/types/xsimd_avx512_float.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/types/xsimd_avx512_int8.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/types/xsimd_avx512_int16.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/types/xsimd_avx512_int32.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/types/xsimd_avx512_int64.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/types/xsimd_avx512_int_base.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/types/xsimd_neon_bool.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/types/xsimd_neon_complex.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/types/xsimd_neon_conversion.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/types/xsimd_neon_double.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/types/xsimd_neon_float.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/types/xsimd_neon_int8.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/types/xsimd_neon_int16.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/types/xsimd_neon_int32.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/types/xsimd_neon_int64.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/types/xsimd_neon_uint8.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/types/xsimd_neon_uint16.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/types/xsimd_neon_uint32.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/types/xsimd_neon_uint64.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/types/xsimd_neon_utils.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/types/xsimd_base.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/types/xsimd_base_bool.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/types/xsimd_complex_base.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/types/xsimd_fallback.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/types/xsimd_int_conversion.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/types/xsimd_sse_complex.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/types/xsimd_sse_conversion.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/types/xsimd_sse_double.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/types/xsimd_sse_float.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/types/xsimd_sse_int8.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/types/xsimd_sse_int16.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/types/xsimd_sse_int32.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/types/xsimd_sse_int64.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/types/xsimd_sse_int_base.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/types/xsimd_traits.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/types/xsimd_types_include.hpp
    ${XSIMD_INCLUDE_DIR}/xsimd/types/xsimd_utils.hpp
)
endif()

add_library(xsimd INTERFACE)

target_include_directories(xsimd INTERFACE
    $<BUILD_INTERFACE:${XSIMD_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:include>)

target_compile_features(xsimd INTERFACE cxx_std_11)

OPTION(ENABLE_FALLBACK "build tests/benchmarks with fallback implementation" OFF)
OPTION(ENABLE_XTL_COMPLEX "enables support for xcomplex defined in xtl" OFF)
OPTION(BUILD_TESTS "xsimd test suite" OFF)
OPTION(DOWNLOAD_GTEST "build gtest from downloaded sources" OFF)

if(DOWNLOAD_GTEST OR GTEST_SRC_DIR)
    set(BUILD_TESTS ON)
endif()

if(ENABLE_FALLBACK)
    add_definitions(-DXSIMD_ENABLE_FALLBACK=1)
endif()

if(ENABLE_XTL_COMPLEX)
    add_definitions(-DXSIMD_ENABLE_XTL_COMPLEX=1)
    find_package(xtl 0.4.11 REQUIRED)
    target_link_libraries(xsimd INTERFACE xtl)
endif()

if(BUILD_TESTS)
    enable_testing()
    if(XSIMD_REFACTORING)
        add_subdirectory(test-refactoring)
    else()
        add_subdirectory(test)
    endif()
endif()

OPTION(BUILD_BENCHMARK "xsimd benchmarks" OFF)
if(BUILD_BENCHMARK)
    add_subdirectory(benchmark)
endif()

OPTION(BUILD_EXAMPLES "xsimd examples" OFF)
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Installation
# ============

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS xsimd
        EXPORT ${PROJECT_NAME}-targets)

# Makes the project importable from the build directory
export(EXPORT ${PROJECT_NAME}-targets
       FILE "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Targets.cmake")

install(DIRECTORY ${XSIMD_INCLUDE_DIR}/xsimd
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# GNUInstallDirs "DATADIR" wrong here; CMake search path wants "share".
set(XSIMD_CMAKECONFIG_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}" CACHE STRING "install path for xsimdConfig.cmake")

configure_package_config_file(${PROJECT_NAME}Config.cmake.in
                              "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
                              INSTALL_DESTINATION ${XSIMD_CMAKECONFIG_INSTALL_DIR})

# xsimd is header-only and does not depend on the architecture.
# Remove CMAKE_SIZEOF_VOID_P from xtensorConfigVersion.cmake so that an xtensorConfig.cmake
# generated for a 64 bit target can be used for 32 bit targets and vice versa.
set(_XTENSOR_CMAKE_SIZEOF_VOID_P ${CMAKE_SIZEOF_VOID_P})
unset(CMAKE_SIZEOF_VOID_P)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
                                 VERSION ${${PROJECT_NAME}_VERSION}
                                 COMPATIBILITY SameMajorVersion)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
        DESTINATION ${XSIMD_CMAKECONFIG_INSTALL_DIR})
install(EXPORT ${PROJECT_NAME}-targets
        FILE ${PROJECT_NAME}Targets.cmake
        DESTINATION ${XSIMD_CMAKECONFIG_INSTALL_DIR})

configure_file(${PROJECT_NAME}.pc.in
               "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.pc"
                @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.pc"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig/")
