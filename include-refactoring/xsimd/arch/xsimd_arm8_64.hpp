#ifndef XSIMD_ARM8_64_HPP
#define XSIMD_ARM8_64_HPP

#include "../types/xsimd_arm8_64_register.hpp"
#include "../types/xsimd_utils.hpp"

namespace xsimd
{
    namespace kernel
    {
        using namespace types;

        template <class A>
        batch<double, A> load_aligned(double const* src, convert<double>, requires<arm8_64>)
        {
            return vld1q_f64(src);
        }

        template <class A>
        void store_aligned(double* dst, batch<double, A> const& src, requires<arm8_64>)
        {
            vst1q_f64(dst, src);
        }
    }
}

#endif

