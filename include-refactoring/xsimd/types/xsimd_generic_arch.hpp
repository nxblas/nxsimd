#ifndef XSIMD_GENERIC_ARCH_HPP
#define XSIMD_GENERIC_ARCH_HPP

namespace xsimd
{
    struct generic
    {
        static constexpr bool supported() { return true; }
        static constexpr bool available() { return true; }
        static constexpr bool requires_alignment() { return false; }
    
    protected:
        
        static constexpr unsigned version(unsigned major, unsigned minor, unsigned patch) { return major * 10000u + minor * 100u + patch; }
    };
}

#endif

