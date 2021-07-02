/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay, Wolf Vollprecht and         *
* Martin Renou                                                             *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_UTILS_HPP
#define XSIMD_UTILS_HPP

#include <complex>
#include <cstdint>
#include <cstring>
#include <type_traits>

#ifdef XSIMD_ENABLE_XTL_COMPLEX
#include "xtl/xcomplex.hpp"
#endif

namespace xsimd
{

    template <class T, class A>
    struct batch;

    template <class T, class A>
    struct batch_bool;

    /**************
     * as_integer *
     **************/

    template <class T>
    struct as_integer : std::make_signed<T>
    {
    };

    template <>
    struct as_integer<float>
    {
        using type = int32_t;
    };

    template <>
    struct as_integer<double>
    {
        using type = int64_t;
    };

    template <class T, class A>
    struct as_integer<batch<T, A>>
    {
        using type = batch<typename as_integer<T>::type, A>;
    };

    template <class B>
    using as_integer_t = typename as_integer<B>::type;

    /***********************
     * as_unsigned_integer *
     ***********************/

    template <class T>
    struct as_unsigned_integer : std::make_unsigned<T>
    {
    };

    template <>
    struct as_unsigned_integer<float>
    {
        using type = uint32_t;
    };

    template <>
    struct as_unsigned_integer<double>
    {
        using type = uint64_t;
    };

    template <class T, class A>
    struct as_unsigned_integer<batch<T, A>>
    {
        using type = batch<typename as_unsigned_integer<T>::type, A>;
    };

    template <class T>
    using as_unsigned_integer_t = typename as_unsigned_integer<T>::type;

    /******************
     * flip_sign_type *
     ******************/

    namespace detail
    {
        template <class T, bool is_signed>
        struct flipped_sign_type_impl : std::make_signed<T>
        {
        };

        template <class T>
        struct flipped_sign_type_impl<T, true> : std::make_unsigned<T>
        {
        };
    }

    template <class T>
    struct flipped_sign_type
        : detail::flipped_sign_type_impl<T, std::is_signed<T>::value>
    {
    };

    template <class T>
    using flipped_sign_type_t = typename flipped_sign_type<T>::type;

    /***********
     * as_float *
     ************/

    template <class T>
    struct as_float;

    template <>
    struct as_float<int32_t>
    {
        using type = float;
    };

    template <>
    struct as_float<int64_t>
    {
        using type = double;
    };

    template <class T, class A>
    struct as_float<batch<T, A>>
    {
        using type = batch<typename as_float<T>::type, A>;
    };

    template <class T>
    using as_float_t = typename as_float<T>::type;

    /**************
     * as_logical *
     **************/

    template <class T>
    struct as_logical;

    template <class T, class A>
    struct as_logical<batch<T, A>>
    {
        using type = batch_bool<T, A>;
    };

    template <class T>
    using as_logical_t = typename as_logical<T>::type;

    /********************
     * bit_cast *
     ********************/

    template<class To, class From>
    To bit_cast(From val) {
      static_assert(sizeof(From) == sizeof(To), "casting between compatible layout");
      // FIXME: Some old version of GCC don't support that trait
      //static_assert(std::is_trivially_copyable<From>::value, "input type is trivially copyable");
      //static_assert(std::is_trivially_copyable<To>::value, "output type is trivially copyable");
      To res;
      std::memcpy(&res, &val, sizeof(val));
      return res;
    }


    /*****************************************
     * Backport of index_sequence from c++14 *
     *****************************************/

    // TODO: Remove this once we drop C++11 support
    namespace detail
    {
        template <typename T>
        struct identity { using type = T; };

        #ifdef __cpp_lib_integer_sequence
            using std::integer_sequence;
            using std::index_sequence;
            using std::make_index_sequence;
            using std::index_sequence_for;
        #else
            template <typename T, T... Is>
            struct integer_sequence {
            using value_type = T;
            static constexpr std::size_t size() noexcept { return sizeof...(Is); }
            };

            template <std::size_t... Is>
            using index_sequence = integer_sequence<std::size_t, Is...>;

            template <typename Lhs, typename Rhs>
            struct make_index_sequence_concat;

            template <std::size_t... Lhs, std::size_t... Rhs>
            struct make_index_sequence_concat<index_sequence<Lhs...>,
                                            index_sequence<Rhs...>>
              : identity<index_sequence<Lhs..., (sizeof...(Lhs) + Rhs)...>> {};

            template <std::size_t N>
            struct make_index_sequence_impl;

            template <std::size_t N>
            using make_index_sequence = typename make_index_sequence_impl<N>::type;

            template <std::size_t N>
            struct make_index_sequence_impl
              : make_index_sequence_concat<make_index_sequence<N / 2>,
                                           make_index_sequence<N - (N / 2)>> {};

            template <>
            struct make_index_sequence_impl<0> : identity<index_sequence<>> {};

            template <>
            struct make_index_sequence_impl<1> : identity<index_sequence<0>> {};

            template <typename... Ts>
            using index_sequence_for = make_index_sequence<sizeof...(Ts)>;
        #endif
    }

    /***********************************
     * Backport of std::get from C++14 *
     ***********************************/

    namespace detail
    {
        template <class T, class... Types, size_t I, size_t... Is>
        const T& get_impl(const std::tuple<Types...>& t, std::is_same<T, T>, index_sequence<I, Is...>)
        {
            return std::get<I>(t);
        }

        template <class T, class U, class... Types, size_t I, size_t... Is>
        const T& get_impl(const std::tuple<Types...>& t, std::is_same<T, U>, index_sequence<I, Is...>)
        {
            using tuple_elem = typename std::tuple_element<I+1, std::tuple<Types...>>::type;
            return get_impl<T>(t, std::is_same<T, tuple_elem>(), index_sequence<Is...>());
        }

        template <class T, class... Types>
        const T& get(const std::tuple<Types...>& t)
        {
            using tuple_elem = typename std::tuple_element<0, std::tuple<Types...>>::type;
            return get_impl<T>(t, std::is_same<T, tuple_elem>(), make_index_sequence<sizeof...(Types)>());
        }
    }

    /*********************************
     * Backport of void_t from C++17 *
     *********************************/

    namespace detail
    {
        template <class... T>
        struct make_void
        {
            using type = void;
        };

        template <class... T>
        using void_t = typename make_void<T...>::type;
    }

    /*****************************************
     * Supplementary std::array constructors *
     *****************************************/

    namespace detail
    {
        // std::array constructor from scalar value ("broadcast")
        template <typename T, std::size_t... Is>
        constexpr std::array<T, sizeof...(Is)>
        array_from_scalar_impl(const T& scalar, index_sequence<Is...>)
        {
            // You can safely ignore this silly ternary, the "scalar" is all
            // that matters. The rest is just a dirty workaround...
            return std::array<T, sizeof...(Is)>{ (Is+1) ? scalar : T() ... };
        }

        template <typename T, std::size_t N>
        constexpr std::array<T, N>
        array_from_scalar(const T& scalar)
        {
            return array_from_scalar_impl(scalar, make_index_sequence<N>());
        }

        // std::array constructor from C-style pointer (handled as an array)
        template <typename T, std::size_t... Is>
        constexpr std::array<T, sizeof...(Is)>
        array_from_pointer_impl(const T* c_array, index_sequence<Is...>)
        {
            return std::array<T, sizeof...(Is)>{ c_array[Is]... };
        }

        template <typename T, std::size_t N>
        constexpr std::array<T, N>
        array_from_pointer(const T* c_array)
        {
            return array_from_pointer_impl(c_array, make_index_sequence<N>());
        }
    }

    /************************
     * is_array_initializer *
     ************************/

    namespace detail
    {
        template <bool...> struct bool_pack;

        template <bool... bs>
        using all_true = std::is_same<
            bool_pack<bs..., true>, bool_pack<true, bs...>
        >;

        template <typename T, typename... Args>
        using is_all_convertible = all_true<std::is_convertible<Args, T>::value...>;

        template <typename T, std::size_t N, typename... Args>
        using is_array_initializer = std::enable_if<
            (sizeof...(Args) == N) && is_all_convertible<T, Args...>::value
        >;

        // Check that a variadic argument pack is a list of N values of type T,
        // as usable for instantiating a value of type std::array<T, N>.
        template <typename T, std::size_t N, typename... Args>
        using is_array_initializer_t = typename is_array_initializer<T, N, Args...>::type;
    }

    /**************
     * is_complex *
     **************/

    // This is used in both xsimd_complex_base.hpp and xsimd_traits.hpp
    // However xsimd_traits.hpp indirectly includes xsimd_complex_base.hpp
    // so we cannot define is_complex in xsimd_traits.hpp. Besides, if
    // no file defining batches is included, we still need this definition
    // in xsimd_traits.hpp, so let's define it here.

    namespace detail
    {
        template <class T>
        struct is_complex : std::false_type
        {
        };

        template <class T>
        struct is_complex<std::complex<T>> : std::true_type
        {
        };

#ifdef XSIMD_ENABLE_XTL_COMPLEX
        template <class T, bool i3ec>
        struct is_complex<xtl::xcomplex<T, T, i3ec>> : std::true_type
        {
        };
#endif
    }

/* For Shift instruction: vshlq_n_u8/vshrq_n_u8 (lhs, n),
 * 'n' must be a constant and is the compile-time literal constant.
 *
 * This Macro is to fix compiling issues from llvm(clang):
 * "argument must be a constant..."
 *
 */
#define EXPAND(...) __VA_ARGS__
#define CASE_LHS(op, i)                  \
    case i: return op(lhs, i);

#define XSIMD_REPEAT_8_0(op, addx)       \
    CASE_LHS(EXPAND(op), 1 + addx);      \
    CASE_LHS(EXPAND(op), 2 + addx);      \
    CASE_LHS(EXPAND(op), 3 + addx);      \
    CASE_LHS(EXPAND(op), 4 + addx);      \
    CASE_LHS(EXPAND(op), 5 + addx);      \
    CASE_LHS(EXPAND(op), 6 + addx);      \
    CASE_LHS(EXPAND(op), 7 + addx);

#define XSIMD_REPEAT_8_N(op, addx)       \
    CASE_LHS(EXPAND(op), 0 + addx);      \
    XSIMD_REPEAT_8_0(op, addx);

#define XSIMD_REPEAT_8(op)               \
    XSIMD_REPEAT_8_0(op, 0);

#define XSIMD_REPEAT_16_0(op, addx)      \
    XSIMD_REPEAT_8_0(op, 0 + addx);      \
    XSIMD_REPEAT_8_N(op, 8 + addx);

#define XSIMD_REPEAT_16_N(op, addx)      \
    XSIMD_REPEAT_8_N(op, 0 + addx);      \
    XSIMD_REPEAT_8_N(op, 8 + addx);

#define XSIMD_REPEAT_16(op)              \
    XSIMD_REPEAT_16_0(op, 0);

#define XSIMD_REPEAT_32_0(op, addx)      \
    XSIMD_REPEAT_16_0(op, 0 + addx);     \
    XSIMD_REPEAT_16_N(op, 16 + addx);

#define XSIMD_REPEAT_32_N(op, addx)      \
    XSIMD_REPEAT_16_N(op, 0 + addx);     \
    XSIMD_REPEAT_16_N(op, 16 + addx);

#define XSIMD_REPEAT_32(op)              \
    XSIMD_REPEAT_32_0(op, 0);

#define XSIMD_REPEAT_64(op)              \
    XSIMD_REPEAT_32_0(op, 0);            \
    XSIMD_REPEAT_32_N(op, 32);

/* The Macro is for vext (lhs, rhs, n)
 *
 * _mm_alignr_epi8, _mm_alignr_epi32 ...
 */
#define CASE_LHS_RHS(op, i)              \
    case i: return op(lhs, rhs, i);

#define XSIMD_REPEAT_2_0(op, addx)       \
    CASE_LHS_RHS(EXPAND(op), 1 + addx);

#define XSIMD_REPEAT_2_N(op, addx)       \
    CASE_LHS_RHS(EXPAND(op), 0 + addx);  \
    XSIMD_REPEAT_2_0(op, addx);

#define XSIMD_REPEAT_2(op)               \
    XSIMD_REPEAT_2_0(op, 0);

#define XSIMD_REPEAT_4_0(op, addx)       \
    XSIMD_REPEAT_2_0(op, 0 + addx);      \
    XSIMD_REPEAT_2_N(op, 2 + addx);

#define XSIMD_REPEAT_4_N(op, addx)       \
    XSIMD_REPEAT_2_N(op, 0 + addx);      \
    XSIMD_REPEAT_2_N(op, 2 + addx);

#define XSIMD_REPEAT_4(op)               \
    XSIMD_REPEAT_4_0(op, 0);

#define XSIMD_REPEAT_8_0_v2(op, addx)    \
    XSIMD_REPEAT_4_0(op, 0 + addx);      \
    XSIMD_REPEAT_4_N(op, 4 + addx);

#define XSIMD_REPEAT_8_N_v2(op, addx)    \
    XSIMD_REPEAT_4_N(op, 0 + addx);      \
    XSIMD_REPEAT_4_N(op, 4 + addx);

#define XSIMD_REPEAT_8_v2(op)            \
    XSIMD_REPEAT_8_0_v2(op, 0);

#define XSIMD_REPEAT_16_0_v2(op, addx)   \
    XSIMD_REPEAT_8_0_v2(op, 0 + addx);   \
    XSIMD_REPEAT_8_N_v2(op, 8 + addx);

#define XSIMD_REPEAT_16_N_v2(op, addx)   \
    XSIMD_REPEAT_8_N_v2(op, 0 + addx);   \
    XSIMD_REPEAT_8_N_v2(op, 8 + addx);

#define XSIMD_REPEAT_16_v2(op)           \
    XSIMD_REPEAT_16_0_v2(op, 0);

#define XSIMD_REPEAT_32_0_v2(op, addx)   \
    XSIMD_REPEAT_16_0_v2(op, 0 + addx);  \
    XSIMD_REPEAT_16_N_v2(op, 16 + addx);

#define XSIMD_REPEAT_32_N_v2(op, addx)   \
    XSIMD_REPEAT_16_N_v2(op, 0 + addx);  \
    XSIMD_REPEAT_16_N_v2(op, 16 + addx);

#define XSIMD_REPEAT_32_v2(op)           \
    XSIMD_REPEAT_32_0_v2(op, 0);

#define XSIMD_REPEAT_64_v2(op)           \
    XSIMD_REPEAT_32_0_v2(op, 0);         \
    XSIMD_REPEAT_32_N_v2(op, 32);
}

#endif

