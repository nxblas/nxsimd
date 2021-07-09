#ifndef XSIMD_ARM8_64_HPP
#define XSIMD_ARM8_64_HPP

#include "../types/xsimd_arm8_64_register.hpp"
#include "../types/xsimd_utils.hpp"

namespace xsimd
{
    namespace kernel
    {
        using namespace types;

        /*************
         * broadcast *
         *************/

        // Required to avoid ambiguous call
        template <class A, class T>
        batch<T, A> broadcast(T val, requires<arm8_64>)
        {
            return broadcast<arm8_64>(val, arm7{});
        }

        template <class A>
        batch<double, A> broadcast(double val, requires<arm8_64>)
        {
            return vdupq_n_f64(val);
        }

        /*******
         * set *
         *******/

        template <class A>
        batch<double, A> set(batch<double, A> const&, requires<arm8_64>, double d0, double d1)
        {
            return float64x2_t{d0, d1};
        }

        /********
         * load *
         ********/

        template <class A>
        batch<double, A> load_aligned(double const* src, convert<double>, requires<arm8_64>)
        {
            return vld1q_f64(src);
        }

        template <class A>
        batch<double, A> load_unaligned(double const* src, convert<double>, requires<arm8_64>)
        {
            return load_aligned<A>(src, convert<double>(), A{});
        }

        /*********
         * store *
         *********/

        template <class A>
        void store_aligned(double* dst, batch<double, A> const& src, requires<arm8_64>)
        {
            vst1q_f64(dst, src);
        }

        template <class A>
        void store_unaligned(double* dst, batch<double, A> const& src, requires<arm8_64>)
        {
            return store_aligned<A>(dst, src, A{});
        }

        /*******
         * neg *
         *******/

        template <class A, class T,  detail::enable_sized_unsigned_t<T, 8> = 0>
        batch<T, A> neg(batch<T, A> const& rhs, requires<arm8_64>)
        {
            return vreinterpretq_u64_s64(vnegq_s64(vreinterpretq_s64_u64(rhs)));
        }

        template <class A, class T,  detail::enable_sized_signed_t<T, 8> = 0>
        batch<T, A> neg(batch<T, A> const& rhs, requires<arm8_64>)
        {
            return vnegq_s64(rhs);
        }

        template <class A>
        batch<double, A> neg(batch<double, A> const& rhs, requires<arm8_64>)
        {
            return vnegq_f64(rhs);
        }

        /*******
         * add *
         *******/

        template <class A>
        batch<double, A> add(batch<double, A> const& lhs, batch<double, A> const& rhs, requires<arm8_64>)
        {
            return vaddq_f64(lhs, rhs);
        }

        /********
         * sadd *
         ********/

        template <class A>
        batch<double, A> sadd(batch<double, A> const& lhs, batch<double, A> const& rhs, requires<arm8_64>)
        {
            return add(lhs, rhs, arm8_64{});
        }

        /*******
         * sub *
         *******/

        template <class A>
        batch<double, A> sub(batch<double, A> const& lhs, batch<double, A> const& rhs, requires<arm8_64>)
        {
            return vsubq_f64(lhs, rhs);
        }

        /********
         * ssub *
         ********/

        template <class A>
        batch<double, A> ssub(batch<double, A> const& lhs, batch<double, A> const& rhs, requires<arm8_64>)
        {
            return sub(lhs, rhs, arm8_64{});
        }

        /*******
         * mul *
         *******/

        template <class A>
        batch<double, A> mul(batch<double, A> const& lhs, batch<double, A> const& rhs, requires<arm8_64>)
        {
            return vmulq_f64(lhs, rhs);
        }

        /*******
         * div *
         *******/

#if defined(XSIMD_FAST_INTEGER_DIVISION)
        template <class A, class T,  detail::enable_sized_unsigned_t<T, 8> = 0>
        batch<T, A> div(batch<T, A> const& lhs, batch<T, A> const& rhs, requires<arm8_64>)
        {
            return vcvtq_u64_f64(vcvtq_f64_u64(lhs) / vcvtq_f64_u64(rhs));
        }

        template <class A, class T,  detail::enable_sized_signed_t<T, 8> = 0>
        batch<T, A> div(batch<T, A> const& lhs, batch<T, A> const& rhs, requires<arm8_64>)
        {
            return vcvtq_s64_f64(vcvtq_f64_s64(lhs) / vcvtq_f64_s64(rhs));
        }
#endif
        template <class A>
        batch<double, A> div(batch<double, A> const& lhs, batch<double, A> const& rhs, requires<arm8_64>)
        {
            return vdivq_f64(lhs, rhs);
        }

        /******
         * eq *
         ******/

        template <class A, class T,  detail::enable_sized_unsigned_t<T, 8> = 0>
        batch_bool<T, A> eq(batch<T, A> const& lhs, batch<T, A> const& rhs, requires<arm8_64>)
        {
            return vceqq_u64(lhs, rhs);
        }

        template <class A, class T,  detail::enable_sized_signed_t<T, 8> = 0>
        batch_bool<T, A> eq(batch<T, A> const& lhs, batch<T, A> const& rhs, requires<arm8_64>)
        {
            return vceqq_s64(lhs, rhs);
        }

        template <class A>
        batch_bool<double, A> eq(batch<double, A> const& lhs, batch<double, A> const& rhs, requires<arm8_64>)
        {
            return vceqq_f64(lhs, rhs);
        }

        template <class A, class T,  detail::enable_sized_unsigned_t<T, 8> = 0>
        batch_bool<T, A> eq(batch_bool<T, A> const& lhs, batch_bool<T, A> const& rhs, requires<arm8_64>)
        {
            return vceqq_u64(lhs, rhs);
        }

        template <class A, class T,  detail::enable_sized_signed_t<T, 8> = 0>
        batch_bool<T, A> eq(batch_bool<T, A> const& lhs, batch_bool<T, A> const& rhs, requires<arm8_64>)
        {
            return vceqq_u64(lhs, rhs);
        }

        template <class A>
        batch_bool<double, A> eq(batch_bool<double, A> const& lhs, batch_bool<double, A> const& rhs, requires<arm8_64>)
        {
            return vceqq_u64(lhs, rhs);
        }

        /******
         * lt *
         ******/

        template <class A, class T,  detail::enable_sized_unsigned_t<T, 8> = 0>
        batch_bool<T, A> lt(batch<T, A> const& lhs, batch<T, A> const& rhs, requires<arm8_64>)
        {
            return vcltq_u64(lhs, rhs);
        }

        template <class A, class T,  detail::enable_sized_signed_t<T, 8> = 0>
        batch_bool<T, A> lt(batch<T, A> const& lhs, batch<T, A> const& rhs, requires<arm8_64>)
        {
            return vcltq_s64(lhs, rhs);
        }

        template <class A>
        batch_bool<double, A> lt(batch<double, A> const& lhs, batch<double, A> const& rhs, requires<arm8_64>)
        {
            return vcltq_f64(lhs, rhs);
        }

        /******
         * le *
         ******/
        
        template <class A, class T,  detail::enable_sized_unsigned_t<T, 8> = 0>
        batch_bool<T, A> le(batch<T, A> const& lhs, batch<T, A> const& rhs, requires<arm8_64>)
        {
            return vcleq_u64(lhs, rhs);
        }

        template <class A, class T,  detail::enable_sized_signed_t<T, 8> = 0>
        batch_bool<T, A> le(batch<T, A> const& lhs, batch<T, A> const& rhs, requires<arm8_64>)
        {
            return vcleq_s64(lhs, rhs);
        }

        template <class A>
        batch_bool<double, A> le(batch<double, A> const& lhs, batch<double, A> const& rhs, requires<arm8_64>)
        {
            return vcleq_f64(lhs, rhs);
        }

        /******
         * gt *
         ******/

        template <class A, class T,  detail::enable_sized_unsigned_t<T, 8> = 0>
        batch_bool<T, A> gt(batch<T, A> const& lhs, batch<T, A> const& rhs, requires<arm8_64>)
        {
            return vcgtq_u64(lhs, rhs);
        }

        template <class A, class T,  detail::enable_sized_signed_t<T, 8> = 0>
        batch_bool<T, A> gt(batch<T, A> const& lhs, batch<T, A> const& rhs, requires<arm8_64>)
        {
            return vcgtq_s64(lhs, rhs);
        }

        template <class A>
        batch_bool<double, A> gt(batch<double, A> const& lhs, batch<double, A> const& rhs, requires<arm8_64>)
        {
            return vcgtq_f64(lhs, rhs);
        }

        /******
         * ge *
         ******/

        template <class A, class T,  detail::enable_sized_unsigned_t<T, 8> = 0>
        batch_bool<T, A> ge(batch<T, A> const& lhs, batch<T, A> const& rhs, requires<arm8_64>)
        {
            return vcgtq_u64(lhs, rhs);
        }

        template <class A, class T,  detail::enable_sized_signed_t<T, 8> = 0>
        batch_bool<T, A> ge(batch<T, A> const& lhs, batch<T, A> const& rhs, requires<arm8_64>)
        {
            return vcgtq_s64(lhs, rhs);
        }

        template <class A>
        batch_bool<double, A> ge(batch<double, A> const& lhs, batch<double, A> const& rhs, requires<arm8_64>)
        {
            return vcgtq_f64(lhs, rhs);
        }

        /***************
         * bitwise_and *
         ***************/

        template <class A>
        batch<double, A> bitwise_and(batch<double, A> const& lhs, batch<double, A> const& rhs, requires<arm8_64>)
        {
            return vreinterpretq_f64_u64(vandq_u64(vreinterpretq_u64_f64(lhs),
                                                   vreinterpretq_u64_f64(rhs)));
        }

        /**************
         * bitwise_or *
         **************/

        template <class A>
        batch<double, A> bitwise_or(batch<double, A> const& lhs, batch<double, A> const& rhs, requires<arm8_64>)
        {
            return vreinterpretq_f64_u64(vorrq_u64(vreinterpretq_u64_f64(lhs),
                                                   vreinterpretq_u64_f64(rhs)));
        }

        /***************
         * bitwise_xor *
         ***************/

        template <class A>
        batch<double, A> bitwise_xor(batch<double, A> const& lhs, batch<double, A> const& rhs, requires<arm8_64>)
        {
            return vreinterpretq_f64_u64(veorq_u64(vreinterpretq_u64_f64(lhs),
                                                   vreinterpretq_u64_f64(rhs)));
        }

        /***************
         * bitwise_not *
         ***************/

        template <class A>
        batch<double, A> bitwise_not(batch<double, A> const& rhs, requires<arm8_64>)
        {
            return vreinterpretq_f64_u32(vmvnq_u32(vreinterpretq_u32_f64(rhs)));
        }

        /******************
         * bitwise_andnot *
         ******************/

        template <class A>
        batch<double, A> bitwise_andnot(batch<double, A> const& lhs, batch<double, A> const& rhs, requires<arm8_64>)
        {
            return vreinterpretq_f64_u64(vbicq_u64(vreinterpretq_u64_f64(lhs),
                                                   vreinterpretq_u64_f64(rhs)));
        }
        
        /*******
         * min *
         *******/

        template <class A>
        batch<double, A> min(batch<double, A> const& lhs, batch<double, A> const& rhs, requires<arm8_64>)
        {
            return vminq_f64(lhs, rhs);
        }

        /*******
         * max *
         *******/

        template <class A>
        batch<double, A> max(batch<double, A> const& lhs, batch<double, A> const& rhs, requires<arm8_64>)
        {
            return vmaxq_f64(lhs, rhs);
        }

        /*******
         * abs *
         *******/

        template <class A, class T,  detail::enable_sized_unsigned_t<T, 8> = 0>
        batch<T, A> abs(batch<T, A> const& rhs, requires<arm8_64>)
        {
            return rhs;
        }

        template <class A, class T,  detail::enable_sized_signed_t<T, 8> = 0>
        batch<T, A> abs(batch<T, A> const& rhs, requires<arm8_64>)
        {
            return vabsq_s64(rhs);
        }

        template <class A>
        batch<double, A> abs(batch<double, A> const& rhs, requires<arm8_64>)
        {
            return vabsq_f64(rhs);
        }

        /********
         * sqrt *
         ********/

        template <class A>
        batch<double, A> sqrt(batch<double, A> const& rhs, requires<arm8_64>)
        {
            return vsqrtq_f64(rhs);
        }

        /********************
         * Fused operations *
         ********************/
        
#ifdef __ARM_FEATURE_FMA
        template <class A>
        batch<double, A> fma(batch<double, A> const& x, batch<double, A> const& y, batch<double, A> const& z, requires<arm8_64>)
        {
            return vfmaq_f64(z, x, y);
        }

        template <class A>
        batch<double, A> fms(batch<double, A> const& x, batch<double, A> const& y, batch<double, A> const& z, requires<arm8_64>)
        {
            return vfmaq_f64(-z, x, y);
        }
#endif

        /********
         * hadd *
         ********/

        template <class A, class T, detail::enable_sized_unsigned_t<T, 1> = 0>
        typename batch<T, A>::value_type hadd(batch<T, A> const& arg, requires<arm8_64>)
        {
            return vaddvq_u8(arg);
        }

        template <class A, class T, detail::enable_sized_signed_t<T, 1> = 0>
        typename batch<T, A>::value_type hadd(batch<T, A> const& arg, requires<arm8_64>)
        {
            return vaddvq_s8(arg);
        }

        template <class A, class T, detail::enable_sized_unsigned_t<T, 2> = 0>
        typename batch<T, A>::value_type hadd(batch<T, A> const& arg, requires<arm8_64>)
        {
            return vaddvq_u16(arg);
        }

        template <class A, class T, detail::enable_sized_signed_t<T, 2> = 0>
        typename batch<T, A>::value_type hadd(batch<T, A> const& arg, requires<arm8_64>)
        {
            return vaddvq_s16(arg);
        }

        template <class A, class T, detail::enable_sized_unsigned_t<T, 4> = 0>
        typename batch<T, A>::value_type hadd(batch<T, A> const& arg, requires<arm8_64>)
        {
            return vaddvq_u32(arg);
        }

        template <class A, class T, detail::enable_sized_signed_t<T, 4> = 0>
        typename batch<T, A>::value_type hadd(batch<T, A> const& arg, requires<arm8_64>)
        {
            return vaddvq_s32(arg);
        }

        template <class A, class T, detail::enable_sized_unsigned_t<T, 8> = 0>
        typename batch<T, A>::value_type hadd(batch<T, A> const& arg, requires<arm8_64>)
        {
            return vaddvq_u64(arg);
        }

        template <class A, class T, detail::enable_sized_signed_t<T, 8> = 0>
        typename batch<T, A>::value_type hadd(batch<T, A> const& arg, requires<arm8_64>)
        {
            return vaddvq_s64(arg);
        }

        template <class A>
        double hadd(batch<double, A> const& arg, requires<arm8_64>)
        {
            return vaddvq_f64(arg);
        }

        /*********
         * haddp *
         *********/

        template <class A>
        batch<double, A> haddp(const batch<double, A>* row, requires<arm8_64>)
        {
            return vpaddq_f64(row[0](), row[1]());
        }

        /**********
         * select *
         **********/

        template <class A>
        batch<double, A> select(batch_bool<double, A> const& cond, batch<double, A> const& a, batch<double, A> const& b, requires<arm8_64>)
        {
            return vbslq_f64(cond, a, b);
        }

        /**********
         * zip_lo *
         **********/

        template <class A, class T, detail::enable_sized_unsigned_t<T, 8> = 0>
        batch<T, A> zip_lo(batch<T, A> const& lhs, batch<T, A> const& rhs, requires<arm8_64>)
        {
            return vzip1q_u64(lhs, rhs);
        }

        template <class A, class T, detail::enable_sized_signed_t<T, 8> = 0>
        batch<T, A> zip_lo(batch<T, A> const& lhs, batch<T, A> const& rhs, requires<arm8_64>)
        {
            return vzip1q_s64(lhs, rhs);
        }

        template <class A>
        batch<double, A> zip_lo(batch<double, A> const& lhs, batch<double, A> const& rhs, requires<arm8_64>)
        {
            return vzip1q_f64(lhs, rhs);
        }

        /**********
         * zip_hi *
         **********/

        template <class A, class T, detail::enable_sized_unsigned_t<T, 8> = 0>
        batch<T, A> zip_hi(batch<T, A> const& lhs, batch<T, A> const& rhs, requires<arm8_64>)
        {
            return vzip2q_u64(lhs, rhs);
        }

        template <class A, class T, detail::enable_sized_signed_t<T, 8> = 0>
        batch<T, A> zip_hi(batch<T, A> const& lhs, batch<T, A> const& rhs, requires<arm8_64>)
        {
            return vzip2q_s64(lhs, rhs);
        }

        template <class A>
        batch<double, A> zip_hi(batch<double, A> const& lhs, batch<double, A> const& rhs, requires<arm8_64>)
        {
            return vzip2q_f64(lhs, rhs);
        }

        /****************
         * extract_pair *
         ****************/

        template <class A>
        batch<double, A> extract_pair(batch<double, A> const& lhs, batch<double, A> const& rhs, const int n, requires<arm8_64>)
        {
            using batch_type = batch<double, A>;
            switch(n)
            {
                case 0: return lhs;
                XSIMD_REPEAT_2(vextq_f64);
                default: break;
            }
            return batch_type(double(0));
        }

        /****************
         * bitwise_cast *
         ****************/

        template <class A, class T>
        batch<double, A> bitwise_cast(batch<T, A> const& arg, batch<double, A> const&, requires<arm8_64>)
        {
            using caster_type = detail::bitwise_caster_impl<float64x2_t,
                                                            uint8x16_t, int8x16_t,
                                                            uint16x8_t, int16x8_t,
                                                            uint32x4_t, int32x4_t,
                                                            uint64x2_t, int64x2_t,
                                                            float32x4_t>;
            constexpr caster_type caster = {
                std::make_tuple(vreinterpretq_f64_u8,  vreinterpretq_f64_s8,  vreinterpretq_f64_u16, vreinterpretq_f64_s16,
                                vreinterpretq_f64_u32, vreinterpretq_f64_s32, vreinterpretq_f64_u64, vreinterpretq_f64_s64,
                                vreinterpretq_f64_f32)
            };
            using register_type = typename batch<T, A>::register_type;
            return caster.run(register_type(arg));
        }

        namespace detail
        {
            template <class S, class... R>
            struct bitwise_caster_arm8
            {
                using container_type = std::tuple<R (*)(S)...>;
                container_type m_func;

                template <class V>
                V run(float64x2_t rhs) const
                {
                    using func_type = V (*)(float64x2_t);
                    auto func = xsimd::detail::get<func_type>(m_func);
                    return func(rhs);
                }
            };
        }

        template <class A, class R>
        batch<R, A> bitwise_cast(batch<double, A> const& arg, batch<R, A> const&, requires<arm8_64>)
        {
            using caster_type = detail::bitwise_caster_arm8<float64x2_t,
                                                            uint8x16_t, int8x16_t,
                                                            uint16x8_t, int16x8_t,
                                                            uint32x4_t, int32x4_t,
                                                            uint64x2_t, int64x2_t,
                                                            float32x4_t>;
            constexpr caster_type caster = {
                std::make_tuple(vreinterpretq_u8_f64,  vreinterpretq_s8_f64,  vreinterpretq_u16_f64, vreinterpretq_s16_f64,
                                vreinterpretq_u32_f64, vreinterpretq_s32_f64, vreinterpretq_u64_f64, vreinterpretq_s64_f64,
                                vreinterpretq_f32_f64)
            };
            using src_register_type = typename batch<double, A>::register_type;
            using dst_register_type = typename batch<R, A>::register_type;
            return caster.run<dst_register_type>(src_register_type(arg));
        }

        template <class A>
        batch<double, A> bitwise_cast(batch<double, A> const& arg, batch<double, A> const&, requires<arm8_64>)
        {
            return arg;
        }

        /*************
         * bool_cast *
         *************/

        template <class A>
        batch_bool<double, A> bool_cast(batch_bool<int64_t, A> const& arg, requires<arm8_64>)
        {
            return arg;
        }

        template <class A>
        batch_bool<int64_t, A> bool_cast(batch_bool<double, A> const& arg, requires<arm8_64>)
        {
            return arg;
        }

        /**********
         * to_int *
         **********/

        template <class A>
        batch<int64_t, A> to_int(const batch<double, A>& x, requires<arm8_64>)
        {
            return vcvtq_s64_f64(x);
        }

        /************
         * to_float *
         ************/

        template <class A>
        batch<double, A> to_float(const batch<int64_t, A>& x, requires<arm8_64>)
        {
            return vcvtq_f64_s64(x);
        }
    }
}

#endif

