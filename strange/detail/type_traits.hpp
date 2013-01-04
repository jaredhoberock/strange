#pragma once

namespace strange
{
namespace detail
{


template<bool, typename T = void> struct enable_if {};
template<typename T>              struct enable_if<true, T> {typedef T type;};

template<bool, typename T> struct lazy_enable_if {};
template<typename T>       struct lazy_enable_if<true, T> {typedef typename T::type type;};

template<bool condition, typename T = void> struct disable_if : enable_if<!condition, T> {};
template<bool condition, typename T>        struct lazy_disable_if : lazy_enable_if<!condition, T> {};


template<typename T, T v>
  struct integral_constant
{
  static const T                  value = v;
  typedef T                       value_type;
  typedef integral_constant<T, v> type;
};

typedef integral_constant<bool, true>     true_type;
typedef integral_constant<bool, false>    false_type;


template<typename T> struct is_integral                           : public false_type {};
template<>           struct is_integral<bool>                     : public true_type {};
template<>           struct is_integral<char>                     : public true_type {};
template<>           struct is_integral<signed char>              : public true_type {};
template<>           struct is_integral<unsigned char>            : public true_type {};
template<>           struct is_integral<short>                    : public true_type {};
template<>           struct is_integral<unsigned short>           : public true_type {};
template<>           struct is_integral<int>                      : public true_type {};
template<>           struct is_integral<unsigned int>             : public true_type {};
template<>           struct is_integral<long>                     : public true_type {};
template<>           struct is_integral<unsigned long>            : public true_type {};
template<>           struct is_integral<long long>                : public true_type {};
template<>           struct is_integral<unsigned long long>       : public true_type {};
template<>           struct is_integral<const bool>               : public true_type {};
template<>           struct is_integral<const char>               : public true_type {};
template<>           struct is_integral<const unsigned char>      : public true_type {};
template<>           struct is_integral<const short>              : public true_type {};
template<>           struct is_integral<const unsigned short>     : public true_type {};
template<>           struct is_integral<const int>                : public true_type {};
template<>           struct is_integral<const unsigned int>       : public true_type {};
template<>           struct is_integral<const long>               : public true_type {};
template<>           struct is_integral<const unsigned long>      : public true_type {};
template<>           struct is_integral<const long long>          : public true_type {};
template<>           struct is_integral<const unsigned long long> : public true_type {};


template<typename T> struct is_floating_point              : public false_type {};
template<>           struct is_floating_point<float>       : public true_type {};
template<>           struct is_floating_point<double>      : public true_type {};
template<>           struct is_floating_point<long double> : public true_type {};


template<typename T> struct is_void             : public false_type {};
template<>           struct is_void<void>       : public true_type {};
template<>           struct is_void<const void> : public true_type {};


template<typename T> struct is_const          : public false_type {};
template<typename T> struct is_const<const T> : public true_type {};


template<typename T> struct is_volatile             : public false_type {};
template<typename T> struct is_volatile<volatile T> : public true_type {};


template<typename T> struct is_reference     : public false_type {};
template<typename T> struct is_reference<T&> : public true_type {};


template<typename T>
  struct remove_const
{
  typedef T type;
}; // end remove_const

template<typename T>
  struct remove_const<const T>
{
  typedef T type;
}; // end remove_const


namespace type_traits_detail
{


// NB: Careful with reference to void.
template<typename T, bool = (is_void<T>::value || is_reference<T>::value)>
  struct add_reference_helper
{
  typedef T& type;
};

template<typename T>
  struct add_reference_helper<T, true>
{
  typedef T type;
};


} // end type_traits_detail


template<typename T>
  struct add_reference
    : public type_traits_detail::add_reference_helper<T>
{};


template <bool, typename Then, typename Else>
  struct eval_if
{
};

template<typename Then, typename Else>
  struct eval_if<true, Then, Else>
{
  typedef typename Then::type type;
};

template<typename Then, typename Else>
  struct eval_if<false, Then, Else>
{
  typedef typename Else::type type;
};


template<typename T> struct identity {typedef T type;};


template<typename T> struct remove_reference {typedef T type;};
template<typename T> struct remove_reference<T&> {typedef T type;};


namespace type_traits_detail
{

template<typename T>
  struct is_int_or_cref
{
  typedef typename remove_reference<T>::type type_sans_ref;
  static const bool value = (is_integral<T>::value
                             || (is_integral<type_sans_ref>::value
                                 && is_const<type_sans_ref>::value
                                 && !is_volatile<type_sans_ref>::value));
}; // end is_int_or_cref


template<typename From, typename To>
  struct is_convertible_sfinae
{
  private:
    typedef char                          one_byte;
    typedef struct { char two_chars[2]; } two_bytes;

    static one_byte  test(To);
    static two_bytes test(...);
    static From      m_from;

  public:
    static const bool value = sizeof(test(m_from)) == sizeof(one_byte);
}; // end is_convertible_sfinae


template<typename From, typename To>
  struct is_convertible_needs_simple_test
{
  static const bool from_is_void      = is_void<From>::value;
  static const bool to_is_void        = is_void<To>::value;
  static const bool from_is_float     = is_floating_point<typename remove_reference<From>::type>::value;
  static const bool to_is_int_or_cref = is_int_or_cref<To>::value;

  static const bool value = (from_is_void || to_is_void || (from_is_float && to_is_int_or_cref));
}; // end is_convertible_needs_simple_test


template<typename From, typename To,
         bool = is_convertible_needs_simple_test<From,To>::value>
  struct is_convertible
{
  static const bool value = (is_void<To>::value
                             || (is_int_or_cref<To>::value
                                 && !is_void<From>::value));
}; // end is_convertible


template<typename From, typename To>
  struct is_convertible<From, To, false>
{
  static const bool value = (is_convertible_sfinae<typename
                             add_reference<From>::type, To>::value);
}; // end is_convertible


} // end type_traits_detail


template<typename From, typename To>
  struct is_convertible
    : public integral_constant<bool, type_traits_detail::is_convertible<From, To>::value>
{
}; // end is_convertible


} // end detail
} // end strange

