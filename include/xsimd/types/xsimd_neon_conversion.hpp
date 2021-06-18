/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay, Wolf Vollprecht and         *
* Martin Renou                                                             *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_NEON_CONVERSION_HPP
#define XSIMD_NEON_CONVERSION_HPP

#include "xsimd_neon_bool.hpp"
#include "xsimd_neon_float.hpp"
#include "xsimd_neon_int8.hpp"
#include "xsimd_neon_int16.hpp"
#include "xsimd_neon_int32.hpp"
#include "xsimd_neon_int64.hpp"
#if XSIMD_ARM_INSTR_SET >= XSIMD_ARM8_64_NEON_VERSION
    #include "xsimd_neon_double.hpp"
#endif
#include "xsimd_neon_uint32.hpp"
#include "xsimd_neon_uint64.hpp"
#include "xsimd_neon_uint16.hpp"
#include "xsimd_neon_uint8.hpp"

namespace xsimd
{

    /************************
     * conversion functions *
     ************************/

    batch<int32_t, 4> to_int(const batch<float, 4>& x);
    batch<float, 4> to_float(const batch<int32_t, 4>& x);

    batch<uint16_t, 8> u8_to_u16(const batch<uint8_t, 16>& x);
    batch<uint8_t, 16> u16_to_u8(const batch<uint16_t, 8>& x);
    batch<uint32_t, 4> u8_to_u32(const batch<uint8_t, 16>& x);
    batch<uint8_t, 16> u32_to_u8(const batch<uint32_t, 4>& x);
    batch<uint64_t, 2> u8_to_u64(const batch<uint8_t, 16>& x);
    batch<uint8_t, 16> u64_to_u8(const batch<uint64_t, 2>& x);

#if XSIMD_ARM_INSTR_SET >= XSIMD_ARM8_64_NEON_VERSION
    batch<int64_t, 2> to_int(const batch<double, 2>& x);
    batch<double, 2> to_float(const batch<int64_t, 2>& x);
#endif

    /**************************
     * boolean cast functions *
     **************************/

    batch_bool<int32_t, 4> bool_cast(const batch_bool<float, 4>& x);
    batch_bool<float, 4> bool_cast(const batch_bool<int32_t, 4>& x);

#if XSIMD_ARM_INSTR_SET >= XSIMD_ARM8_64_NEON_VERSION
    batch_bool<int64_t, 2> bool_cast(const batch_bool<double, 2>& x);
    batch_bool<double, 2> bool_cast(const batch_bool<int64_t, 2>& x);
#endif

    /******************************************
     *  Convert Bytes, Shorts, Words, Doubles *
     *  to batch functions                    *
     ******************************************/

   void bytes_to_batch_vector(batch<uint8_t, 16>& vec,
                        int8_t b15, int8_t b14, int8_t b13, int8_t b12,
                        int8_t b11, int8_t b10, int8_t b9, int8_t b8,
                        int8_t b7, int8_t b6, int8_t b5, int8_t b4,
                        int8_t b3, int8_t b2, int8_t b1, int8_t b0);

    void shorts_to_batch_vector(batch<uint8_t, 16>& vec,
                          int16_t s7, int16_t s6, int16_t s5, int16_t s4,
                          int16_t s3, int16_t s2, int16_t s1, int16_t s0);

    void words_to_batch_vector(batch<uint8_t, 16>& vec,
                         int32_t i3, int32_t i2, int32_t i1, int32_t i0);

    void longs_to_batch_vector(batch<uint8_t, 16>& vec, int64_t d1, int64_t d0);

    /*******************************
     * bitwise_cast implementation *
     *******************************/

#if XSIMD_ARM_INSTR_SET >= XSIMD_ARM8_64_NEON_VERSION
    XSIMD_DEFINE_BITWISE_CAST_FLOAT(double, 2)
#endif
    XSIMD_DEFINE_BITWISE_CAST_FLOAT(float, 4)
    XSIMD_DEFINE_BITWISE_CAST(int64_t, 2)
    XSIMD_DEFINE_BITWISE_CAST(uint64_t, 2)
    XSIMD_DEFINE_BITWISE_CAST(int32_t, 4)
    XSIMD_DEFINE_BITWISE_CAST(uint32_t, 4)
    XSIMD_DEFINE_BITWISE_CAST(int16_t, 8)
    XSIMD_DEFINE_BITWISE_CAST(uint16_t, 8)
    XSIMD_DEFINE_BITWISE_CAST(int8_t, 16)
    XSIMD_DEFINE_BITWISE_CAST(uint8_t, 16)
    
    /***************************************
     * conversion functions implementation *
     ***************************************/

    inline batch<int32_t, 4> to_int(const batch<float, 4>& x)
    {
        return vcvtq_s32_f32(x);
    }

    inline batch<float, 4> to_float(const batch<int32_t, 4>& x)
    {
        return vcvtq_f32_s32(x);
    }

    inline batch<uint16_t, 8> u8_to_u16(const batch<uint8_t, 16>& x)
    {
        return vreinterpretq_u16_u8(x);
    }

    inline batch<uint8_t, 16> u16_to_u8(const batch<uint16_t, 8>& x)
    {
        return vreinterpretq_u8_u16(x);
    }

    inline batch<uint32_t, 4> u8_to_u32(const batch<uint8_t, 16>& x)
    {
        return vreinterpretq_u32_u8(x);
    }

    inline batch<uint8_t, 16> u32_to_u8(const batch<uint32_t, 4>& x)
    {
        return vreinterpretq_u8_u32(x);
    }

    inline batch<uint64_t, 2> u8_to_u64(const batch<uint8_t, 16>& x)
    {
        return vreinterpretq_u64_u8(x);
    }

    inline batch<uint8_t, 16> u64_to_u8(const batch<uint64_t, 2>& x)
    {
        return vreinterpretq_u8_u64(x);
    }

#if XSIMD_ARM_INSTR_SET >= XSIMD_ARM8_64_NEON_VERSION
    inline batch<int64_t, 2> to_int(const batch<double, 2>& x)
    {
        return vcvtq_s64_f64(x);
    }

    inline batch<double, 2> to_float(const batch<int64_t, 2>& x)
    {
        return vcvtq_f64_s64(x);
    }
#endif

    /*****************************************
     * batch cast functions implementation *
     *****************************************/

    XSIMD_BATCH_CAST_INTRINSIC(int8_t, uint8_t, 16, vreinterpretq_u8_s8);
    XSIMD_BATCH_CAST_INTRINSIC(uint8_t, int8_t, 16, vreinterpretq_s8_u8);
    XSIMD_BATCH_CAST_INTRINSIC(int16_t, uint16_t, 8, vreinterpretq_u16_s16);
    XSIMD_BATCH_CAST_INTRINSIC(uint16_t, int16_t, 8, vreinterpretq_s16_u16);
    XSIMD_BATCH_CAST_INTRINSIC(int32_t, uint32_t, 4, vreinterpretq_u32_s32);
    XSIMD_BATCH_CAST_INTRINSIC(int32_t, float, 4, vcvtq_f32_s32);
    XSIMD_BATCH_CAST_INTRINSIC(uint32_t, int32_t, 4, vreinterpretq_s32_u32);
    XSIMD_BATCH_CAST_INTRINSIC(uint32_t, float, 4, vcvtq_f32_u32);
    XSIMD_BATCH_CAST_INTRINSIC(float, int32_t, 4, vcvtq_s32_f32);
    XSIMD_BATCH_CAST_INTRINSIC(float, uint32_t, 4, vcvtq_u32_f32);
#if XSIMD_ARM_INSTR_SET >= XSIMD_ARM8_64_NEON_VERSION
    XSIMD_BATCH_CAST_INTRINSIC(int64_t, uint64_t, 2, vreinterpretq_u64_s64);
    XSIMD_BATCH_CAST_INTRINSIC(int64_t, double, 2, vcvtq_f64_s64);
    XSIMD_BATCH_CAST_INTRINSIC(uint64_t, int64_t, 2, vreinterpretq_s64_u64);
    XSIMD_BATCH_CAST_INTRINSIC(uint64_t, double, 2, vcvtq_f64_u64);
    XSIMD_BATCH_CAST_INTRINSIC(double, int64_t, 2, vcvtq_s64_f64);
    XSIMD_BATCH_CAST_INTRINSIC(double, uint64_t, 2, vcvtq_u64_f64);
#endif

    /**************************
     * boolean cast functions *
     **************************/

    inline batch_bool<int32_t, 4> bool_cast(const batch_bool<float, 4>& x)
    {
        return x;
    }

    inline batch_bool<float, 4> bool_cast(const batch_bool<int32_t, 4>& x)
    {
        return x;
    }

#if XSIMD_ARM_INSTR_SET >= XSIMD_ARM8_64_NEON_VERSION
    inline batch_bool<int64_t, 2> bool_cast(const batch_bool<double, 2>& x)
    {
        return x;
    }

    inline batch_bool<double, 2> bool_cast(const batch_bool<int64_t, 2>& x)
    {
        return x;
    }
#endif

    /*****************************************
     * bitwise cast functions implementation *
     *****************************************/

    XSIMD_BITWISE_CAST_INTRINSIC(float, 4,
                                 int32_t, 4,
                                 vreinterpretq_s32_f32)

    XSIMD_BITWISE_CAST_INTRINSIC(float, 4,
                                 int64_t, 2,
                                 vreinterpretq_s64_f32)

    XSIMD_BITWISE_CAST_INTRINSIC(int32_t, 4,
                                 float, 4,
                                 vreinterpretq_f32_s32)

    XSIMD_BITWISE_CAST_INTRINSIC(int64_t, 2,
                                 float, 4,
                                 vreinterpretq_f32_s64)

#if XSIMD_ARM_INSTR_SET >= XSIMD_ARM8_64_NEON_VERSION
    XSIMD_BITWISE_CAST_INTRINSIC(double, 2,
                                 float, 4,
                                 vreinterpretq_f32_f64)

    XSIMD_BITWISE_CAST_INTRINSIC(double, 2,
                                 int32_t, 4,
                                 vreinterpretq_s32_f64)

    XSIMD_BITWISE_CAST_INTRINSIC(double, 2,
                                 int64_t, 2,
                                 vreinterpretq_s64_f64)

    XSIMD_BITWISE_CAST_INTRINSIC(int32_t, 4,
                                 double, 2,
                                 vreinterpretq_f64_s32)

    XSIMD_BITWISE_CAST_INTRINSIC(int64_t, 2,
                                 double, 2,
                                 vreinterpretq_f64_s64)

    XSIMD_BITWISE_CAST_INTRINSIC(float, 4,
                                 double, 2,
                                 vreinterpretq_f64_f32)
#endif

    /*****************************************
     * vector cast functions implementation *
     *****************************************/

    inline void bytes_to_batch_vector(batch<uint8_t, 16>& vec,
                                int8_t b15, int8_t b14, int8_t b13, int8_t b12,
                                int8_t b11, int8_t b10, int8_t b9, int8_t b8,
                                int8_t b7, int8_t b6, int8_t b5, int8_t b4,
                                int8_t b3, int8_t b2, int8_t b1, int8_t b0)
    {
        int8_t bytes_buf[16] = {
            b0, b1, b2, b3, b4, b5, b6, b7,
            b8, b9, b10, b11, b12, b13, b14, b15};

        vec = vreinterpretq_u8_s8(vld1q_s8(bytes_buf));
    }

    inline void shorts_to_batch_vector(batch<uint8_t, 16>& vec,
                                 int16_t s7, int16_t s6, int16_t s5, int16_t s4,
                                 int16_t s3, int16_t s2, int16_t s1, int16_t s0)
    {
        int16_t shorts_buf[8] = {
            s0, s1, s2, s3, s4, s5, s6, s7};
        vec = vreinterpretq_u8_s16(vld1q_s16(shorts_buf));
    }

    inline void words_to_batch_vector(batch<uint8_t, 16>& vec,
                                int32_t i3, int32_t i2, int32_t i1, int32_t i0)
    {
        int32_t words_buf[4] = {i0, i1, i2, i3};
        vec = vreinterpretq_u8_s32(vld1q_s32(words_buf));
    }

    inline void longs_to_batch_vector(batch<uint8_t, 16>& vec, int64_t d1, int64_t d0)
    {
        vec = vreinterpretq_u8_s64(
            vcombine_s64(vcreate_s64(d0), vcreate_s64(d1)));
    }

}

#endif
