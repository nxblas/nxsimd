#ifndef XSIMD_ARM8_64_HPP
#define XSIMD_ARM8_64_HPP

#include "../types/xsimd_arm8_64_register.hpp"
#include "../types/xsimd_utils.hpp"

namespace xsimd
{
    namespace kernel
    {
        using namespace types;

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
            return load_aligned<A>(src, convert<double>{}, A{});
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
            store_aligned<A>(dst, src, A{});
        }

        /*******
         * neg *
         *******/

        template <class A>
        batch<double, A> neg(batch<double, A> const& rhs, requires<arm8_64>)
        {
            return vnegq_f64(rhs);
        }
    }
}

#endif

