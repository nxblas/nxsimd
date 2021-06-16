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
            template <class... T>
            struct arm_dispatcher
            {
                using container_type = std::tuple<T (*)(T, T) ...>;
                const container_type m_func;

                template <class U>
                U run(U lhs, U rhs) const
                {
                    using func_type = U (*)(U, U);
                    auto func = xsimd::detail::get<func_type>(m_func);
                    return func(lhs, rhs);
                }
            };

            using arm_dispatcher_type = arm_dispatcher<uint8x16_t, int8x16_t,
                                                       uint16x8_t, int16x8_t,
                                                       uint32x4_t, int32x4_t,
                                                       uint64x2_t, int64x2_t>;
        }

        template <class T, class A, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        batch<T, A> add(batch<T, A> const& lhs, batch<T, A> const& rhs, requires<arm7>)
        {
            using register_type = typename batch<T, A>::register_type;
            constexpr detail::arm_dispatcher_type dispatcher =
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

        namespace conversion
        {
        }
    }
}

#endif
