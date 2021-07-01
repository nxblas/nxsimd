#ifndef XSIMD_ARM7_HPP
#define XSIMD_ARM7_HPP

#include <tuple>

#include "../types/xsimd_arm7_register.hpp"
#include "../types/xsimd_utils.hpp"

namespace xsimd
{
    namespace kernel
    {
        using namespace types;

        namespace detail
        {
            template <template <class> class return_type, class... T>
            struct arm_dispatcher_base
            {
                struct unary
                {
                    using container_type = std::tuple<return_type<T> (*)(T)...>;
                    const container_type m_func;

                    template <class U>
                    return_type<U> run(U rhs) const
                    {
                        using func_type = return_type<U> (*)(U);
                        auto func = xsimd::detail::get<func_type>(m_func);
                        return func(rhs);
                    }
                };

                struct binary
                {
                    using container_type = std::tuple<return_type<T> (*)(T, T) ...>;
                    const container_type m_func;

                    template <class U>
                    return_type<U> run(U lhs, U rhs) const
                    {
                        using func_type = return_type<U> (*)(U, U);
                        auto func = xsimd::detail::get<func_type>(m_func);
                        return func(lhs, rhs);
                    }
                };
            };

            /***************************
             *  arithmetic dispatchers *
             ***************************/

            template <class T>
            using identity_return_type = T;
            
            template <class... T>
            struct arm_dispatcher_impl : arm_dispatcher_base<identity_return_type, T...>
            {
            };


            using arm_dispatcher = arm_dispatcher_impl<uint8x16_t, int8x16_t,
                                                       uint16x8_t, int16x8_t,
                                                       uint32x4_t, int32x4_t,
                                                       uint64x2_t, int64x2_t>;

            using excluding_int64_dispatcher = arm_dispatcher_impl<uint8x16_t, int8x16_t,
                                                                   uint16x8_t, int16x8_t,
                                                                   uint32x4_t, int32x4_t>;

            /**************************
             * comparison dispatchers *
             **************************/

            template <class T>
            struct comp_return_type_impl;

            template <>
            struct comp_return_type_impl<uint8x16_t>
            {
                using type = uint8x16_t;
            };

            template <>
            struct comp_return_type_impl<int8x16_t>
            {
                using type = uint8x16_t;
            };

            template <>
            struct comp_return_type_impl<uint16x8_t>
            {
                using type = uint16x8_t;
            };

            template <>
            struct comp_return_type_impl<int16x8_t>
            {
                using type = uint16x8_t;
            };

            template <>
            struct comp_return_type_impl<uint32x4_t>
            {
                using type = uint32x4_t;
            };

            template <>
            struct comp_return_type_impl<int32x4_t>
            {
                using type = uint32x4_t;
            };

            template <class T>
            using comp_return_type = typename comp_return_type_impl<T>::type;

            template <class... T>
            struct arm_comp_dispatcher_impl : arm_dispatcher_base<comp_return_type, T...>
            {
            };

            using excluding_int64_comp_dispatcher = arm_comp_dispatcher_impl<uint8x16_t, int8x16_t,
                                                                             uint16x8_t, int16x8_t,
                                                                             uint32x4_t, int32x4_t>;

            /**************************************
             * enabling / disabling metafunctions *
             **************************************/

            template <class T>
            using enable_integral_t = typename std::enable_if<std::is_integral<T>::value, int>::type;

            template <class T, size_t S>
            using enable_sized_signed_t = typename std::enable_if<std::is_integral<T>::value &&
                                                                  std::is_signed<T>::value &&
                                                                  sizeof(T) == S, int>::type;

            template <class T, size_t S>
            using enable_sized_unsigned_t = typename std::enable_if<std::is_integral<T>::value &&
                                                                    !std::is_signed<T>::value &&
                                                                    sizeof(T) == S, int>::type;

            template <class T>
            using exclude_int64_t = typename std::enable_if<std::is_integral<T>::value && sizeof(T) != 8, int>::type;
        }

        /*************
         * broadcast *
         *************/

        template <class T, class A, detail::enable_sized_unsigned_t<T, 1> = 0>
        batch<T, A> broadcast(T val)
        {
            return vdupq_n_u8(uint8_t(val));
        }

        template <class T, class A, detail::enable_sized_signed_t<T, 1> = 0>
        batch<T, A> broadcast(T val)
        {
            return vdupq_n_s8(int8_t(val));
        }

        template <class T, class A, detail::enable_sized_unsigned_t<T, 2> = 0>
        batch<T, A> broadcast(T val)
        {
            return vdupq_n_u16(uint16_t(val));
        }

        template <class T, class A, detail::enable_sized_signed_t<T, 2> = 0>
        batch<T, A> broadcast(T val)
        {
            return vdupq_n_s16(int16_t(val));
        }

        template <class T, class A, detail::enable_sized_unsigned_t<T, 4> = 0>
        batch<T, A> broadcast(T val)
        {
            return vdupq_n_u32(uint32_t(val));
        }

        template <class T, class A, detail::enable_sized_signed_t<T, 4> = 0>
        batch<T, A> broadcast(T val)
        {
            return vdupq_n_s32(int32_t(val));
        }

        template <class T, class A, detail::enable_sized_unsigned_t<T, 8> = 0>
        batch<T, A> broadcast(T val)
        {
            return vdupq_n_u64(uint64_t(val));
        }

        template <class T, class A, detail::enable_sized_signed_t<T, 8> = 0>
        batch<T, A> broadcast(T val)
        {
            return vdupq_n_s64(int64_t(val));
        }

        template <class A>
        batch<float, A> broadcast(float val)
        {
            return vdupq_n_f32(val);
        }

        /*******
         * set *
         *******/

        template <class T, class A, class... Args, detail::enable_integral_t<T> = 0>
        batch<T, A> set(batch<T, A> const&, requires<arm7>, Args... args)
        {
            return xsimd::types::detail::arm_vector_type<T>{args...};
        }

        template <class A>
        batch<float, A> set(batch<float, A> const &, requires<arm7>, float f0, float f1, float f2, float f3)
        {
            return float32x4_t{f0, f1, f2, f3};
        }

        /*******
         * neg *
         *******/

        template <class T, class A, detail::enable_sized_unsigned_t<T, 1> = 0>
        batch<T, A> neg(batch<T, A> const& rhs, requires<arm7>)
        {
            return vreinterpretq_u8_s8(vnegq_s8(vreinterpretq_s8_u8(rhs)));
        }

        template <class T, class A, detail::enable_sized_signed_t<T, 1> = 0>
        batch<T, A> neg(batch<T, A> const& rhs, requires<arm7>)
        {
            return vnegq_s8(rhs);
        }

        template <class T, class A, detail::enable_sized_unsigned_t<T, 2> = 0>
        batch<T, A> neg(batch<T, A> const& rhs, requires<arm7>)
        {
            return vreinterpretq_u16_s16(vnegq_s16(vreinterpretq_s16_u16(rhs)));
        }

        template <class T, class A, detail::enable_sized_signed_t<T, 2> = 0>
        batch<T, A> neg(batch<T, A> const& rhs, requires<arm7>)
        {
            return vnegq_s16(rhs);
        }

        template <class T, class A, detail::enable_sized_unsigned_t<T, 4> = 0>
        batch<T, A> neg(batch<T, A> const& rhs, requires<arm7>)
        {
            return vreinterpretq_u32_s32(vnegq_s32(vreinterpretq_s32_u32(rhs)));
        }

        template <class T, class A, detail::enable_sized_signed_t<T, 4> = 0>
        batch<T, A> neg(batch<T, A> const& rhs, requires<arm7>)
        {
            return vnegq_s32(rhs);
        }

        template <class A>
        batch<float, A> neg(batch<float, A> const& rhs)
        {
            return vnegq_f32(rhs);
        }

        /*******
         * add *
         *******/

        template <class T, class A, detail::enable_integral_t<T> = 0>
        batch<T, A> add(batch<T, A> const& lhs, batch<T, A> const& rhs, requires<arm7>)
        {
            using register_type = typename batch<T, A>::register_type;
            constexpr detail::arm_dispatcher::binary dispatcher =
            {
                std::make_tuple(vaddq_u8, vaddq_s8, vaddq_u16, vaddq_s16, vaddq_u32, vaddq_s32, vaddq_u64, vaddq_s64)
            };
            return dispatcher.run(register_type(lhs), register_type(rhs));
        }

        template <class A>
        batch<float, A> add(batch<float, A> const& lhs, batch<float, A> const& rhs, requires<arm7>)
        {
            return vaddq_f32(lhs, rhs);
        }

        /********
         * sadd *
         ********/

        template <class T, class A, detail::enable_integral_t<T> = 0>
        batch<T, A> sadd(batch<T, A> const& lhs, batch<T, A> const& rhs, requires<arm7>)
        {
            using register_type = typename batch<T, A>::register_type;
            constexpr detail::arm_dispatcher::binary dispatcher =
            {
                std::make_tuple(vqaddq_u8, vqaddq_s8, vqaddq_u16, vqaddq_s16, vqaddq_u32, vqaddq_s32, vqaddq_u64, vqaddq_s64)
            };
            return dispatcher.run(register_type(lhs), register_type(rhs));
        }

        template <class A>
        batch<float, A> sadd(batch<float, A> const& lhs, batch<float, A> const& rhs, requires<arm7>)
        {
            return add(lhs, rhs);
        }

        /*******
         * sub *
         *******/

        template <class T, class A, detail::enable_integral_t<T> = 0>
        batch<T, A> sub(batch<T, A> const& lhs, batch<T, A> const& rhs, requires<arm7>)
        {
            using register_type = typename batch<T, A>::register_type;
            constexpr detail::arm_dispatcher::binary dispatcher =
            {
                std::make_tuple(vsubq_u8, vsubq_s8, vsubq_u16, vsubq_s16, vsubq_u32, vsubq_s32, vsubq_u64, vsubq_s64)
            };
            return dispatcher.run(register_type(lhs), register_type(rhs));
        }

        template <class A>
        batch<float, A> sub(batch<float, A> const& lhs, batch<float, A> const& rhs, requires<arm7>)
        {
            return vsubq_f32(lhs, rhs);
        }

        /********
         * ssub *
         ********/

        template <class T, class A, detail::enable_integral_t<T> = 0>
        batch<T, A> ssub(batch<T, A> const& lhs, batch<T, A> const& rhs, requires<arm7>)
        {
            using register_type = typename batch<T, A>::register_type;
            constexpr detail::arm_dispatcher::binary dispatcher =
            {
                std::make_tuple(vqsubq_u8, vqsubq_s8, vqsubq_u16, vqsubq_s16, vqsubq_u32, vqsubq_s32, vqsubq_u64, vqsubq_s64)
            };
            return dispatcher.run(register_type(lhs), register_type(rhs));
        }

        template <class A>
        batch<float, A> ssub(batch<float, A> const& lhs, batch<float, A> const& rhs, requires<arm7>)
        {
            return sub(lhs, rhs);
        }

        /*******
         * mul *
         *******/

        template <class T, class A, detail::exclude_int64_t<T> = 0>
        batch<T, A> mul(batch<T, A> const& lhs, batch<T, A> const& rhs, requires<arm7>)
        {
            using register_type = typename batch<T, A>::register_type;
            constexpr detail::excluding_int64_dispatcher::binary dispatcher =
            {
                std::make_tuple(vmulq_u8, vmulq_s8, vmulq_u16, vmulq_s16, vmulq_u32, vmulq_s32)
            };
            return dispatcher.run(register_type(lhs), register_type(rhs));
        }

        template <class A>
        batch<float, A> mul(batch<float, A> const& lhs, batch<float, A> const& rhs, requires<arm7>)
        {
            return vmulq_f32(lhs, rhs);
        }

        /*******
         * div *
         *******/

#if defined(XSIMD_FAST_INTEGER_DIVISION)
        template <class T, class A,  detail::enable_sized_signed_t<T, 4> = 0>
        batch<T, A> div(batch<T, A> const& lhs, batch<T, A> const& rhs, requires<arm7>)
        {
            return vcvtq_s32_f32(vcvtq_f32_s32(lhs) / vcvtq_f32_s32(rhs));
        }

        template <class T, class A,  detail::enable_sized_unsigned_t<T, 4> = 0>
        batch<T, A> div(batch<T, A> const& lhs, batch<T, A> const& rhs, requires<arm7>)
        {
            return vcvtq_u32_f32(vcvtq_f32_u32(lhs) / vcvtq_f32_u32(rhs));
        }
#endif

        template <class A>
        batch<float, A> div(batch<float, A> const& lhs, batch<float, A> const& rhs, requires<arm7>)
        {
            // from stackoverflow & https://projectne10.github.io/Ne10/doc/NE10__divc_8neon_8c_source.html
            // get an initial estimate of 1/b.
            float32x4_t reciprocal = vrecpeq_f32(rhs);

            // use a couple Newton-Raphson steps to refine the estimate.  Depending on your
            // application's accuracy requirements, you may be able to get away with only
            // one refinement (instead of the two used here).  Be sure to test!
            reciprocal = vmulq_f32(vrecpsq_f32(rhs, reciprocal), reciprocal);
            reciprocal = vmulq_f32(vrecpsq_f32(rhs, reciprocal), reciprocal);

            // and finally, compute a / b = a * (1 / b)
            return vmulq_f32(lhs, reciprocal);
        }

        /******
         * eq *
         ******/

        template <class T, class A, detail::exclude_int64_t<T> = 0>
        batch_bool<T, A> eq(batch<T, A> const& lhs, batch<T, A> const& rhs, requires<arm7>)
        {
            using register_type = typename batch<T, A>::register_type;
            constexpr detail::excluding_int64_comp_dispatcher::binary dispatcher =
            {
                std::make_tuple(vceqq_u8, vceqq_s8, vceqq_u16, vceqq_s16, vceqq_u32, vceqq_s32)
            };
            return dispatcher.run(register_type(lhs), register_type(rhs));
        }

        template <class A>
        batch_bool<float, A> eq(batch<float, A> const& lhs, batch<float, A> const& rhs, requires<arm7>)
        {
            return vceqq_f32(lhs, rhs);
        }

        /******
         * lt *
         ******/

        template <class T, class A, detail::exclude_int64_t<T> = 0>
        batch_bool<T, A> lt(batch<T, A> const& lhs, batch<T, A> const& rhs, requires<arm7>)
        {
            using register_type = typename batch<T, A>::register_type;
            constexpr detail::excluding_int64_comp_dispatcher::binary dispatcher =
            {
                std::make_tuple(vcltq_u8, vcltq_s8, vcltq_u16, vcltq_s16, vcltq_u32, vcltq_s32)
            };
            return dispatcher.run(register_type(lhs), register_type(rhs));
        }

        template <class A>
        batch_bool<float, A> lt(batch<float, A> const& lhs, batch<float, A> const& rhs, requires<arm7>)
        {
            return vcltq_f32(lhs, rhs);
        }

        /******
         * le *
         ******/

        template <class T, class A, detail::exclude_int64_t<T> = 0>
        batch_bool<T, A> le(batch<T, A> const& lhs, batch<T, A> const& rhs, requires<arm7>)
        {
            using register_type = typename batch<T, A>::register_type;
            constexpr detail::excluding_int64_comp_dispatcher::binary dispatcher =
            {
                std::make_tuple(vcleq_u8, vcleq_s8, vcleq_u16, vcleq_s16, vcleq_u32, vcleq_s32)
            };
            return dispatcher.run(register_type(lhs), register_type(rhs));
        }

        template <class A>
        batch_bool<float, A> le(batch<float, A> const& lhs, batch<float, A> const& rhs, requires<arm7>)
        {
            return vcleq_f32(lhs, rhs);
        }

        /******
         * gt *
         ******/

        template <class T, class A, detail::exclude_int64_t<T> = 0>
        batch_bool<T, A> gt(batch<T, A> const& lhs, batch<T, A> const& rhs, requires<arm7>)
        {
            using register_type = typename batch<T, A>::register_type;
            constexpr detail::excluding_int64_comp_dispatcher::binary dispatcher =
            {
                std::make_tuple(vcgtq_u8, vcgtq_s8, vcgtq_u16, vcgtq_s16, vcgtq_u32, vcgtq_s32)
            };
            return dispatcher.run(register_type(lhs), register_type(rhs));
        }

        template <class A>
        batch_bool<float, A> gt(batch<float, A> const& lhs, batch<float, A> const& rhs, requires<arm7>)
        {
            return vcgtq_f32(lhs, rhs);
        }

        /******
         * ge *
         ******/

        template <class T, class A, detail::exclude_int64_t<T> = 0>
        batch_bool<T, A> ge(batch<T, A> const& lhs, batch<T, A> const& rhs, requires<arm7>)
        {
            using register_type = typename batch<T, A>::register_type;
            constexpr detail::excluding_int64_comp_dispatcher::binary dispatcher =
            {
                std::make_tuple(vcgeq_u8, vcgeq_s8, vcgeq_u16, vcgeq_s16, vcgeq_u32, vcgeq_s32)
            };
            return dispatcher.run(register_type(lhs), register_type(rhs));
        }

        template <class A>
        batch_bool<float, A> ge(batch<float, A> const& lhs, batch<float, A> const& rhs, requires<arm7>)
        {
            return vcgeq_f32(lhs, rhs);
        }
    }
}

#endif
