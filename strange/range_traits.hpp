#pragma once

#include <thrust/detail/type_traits.h>

namespace strange
{
namespace detail
{


template<typename T>
  struct has_iterator
{
  typedef char yes_type;
  typedef int  no_type;
  template<typename S> static yes_type test(typename S::iterator *);
  template<typename S> static no_type  test(...);
  static bool const value = sizeof(test<T>(0)) == sizeof(yes_type);
  typedef thrust::detail::integral_constant<bool, value> type;
}; // end has_iterator

template<typename T>
  struct has_const_iterator
{
  typedef char yes_type;
  typedef int  no_type;
  template<typename S> static yes_type test(typename S::const_iterator *);
  template<typename S> static no_type  test(...);
  static bool const value = sizeof(test<T>(0)) == sizeof(yes_type);
  typedef thrust::detail::integral_constant<bool, value> type;
}; // end has_const_iterator

template<typename Range>
  struct nested_const_iterator
{
  typedef typename Range::const_iterator type;
};

template<typename Range>
  struct nested_iterator
{
  typedef typename Range::iterator type;
};


// if the Range has a nested const_iterator type, return it
// else, return its nested iterator type
template<typename Range>
  struct range_const_iterator
    : thrust::detail::eval_if<
        detail::has_const_iterator<Range>::value,
        detail::nested_const_iterator<Range>,
        detail::nested_iterator<Range>
      >
{};

template<typename T, std::size_t sz>
  struct range_const_iterator<T[sz]>
{
  typedef const T* type;
};


template<typename Range>
  struct range_mutable_iterator
    : nested_iterator<Range>
{};


template<typename T, std::size_t sz>
  struct range_mutable_iterator<T[sz]>
{
  typedef T* type;
};


template<typename T>
  struct is_range
    : detail::has_iterator<T>
{};

template<typename T, std::size_t sz>
  struct is_range<T[sz]>
    : thrust::detail::true_type
{};

template<typename T>
  struct is_not_range
    : thrust::detail::integral_constant<
        bool,
        !is_range<T>::value
      >
{};

template<typename T, typename Result = void>
  struct enable_if_range
    : thrust::detail::enable_if<
        is_range<T>::value,
        Result
      >
{};

template<typename T1, typename T2, typename Result = void>
  struct enable_if_ranges
    : thrust::detail::enable_if<
        is_range<T1>::value && is_range<T2>::value,
        Result
      >
{};

template<typename T1, typename T2, typename Result = void>
  struct enable_if_range_and_scalar
    : thrust::detail::enable_if<
        is_range<T1>::value && is_not_range<T2>::value,
        Result
      >
{};

template<typename T1, typename T2, typename Result = void>
  struct enable_if_scalar_and_range
    : thrust::detail::enable_if<
        is_not_range<T1>::value && is_range<T2>::value,
        Result
      >
{};

template<typename T1, typename T2, typename Result = void>
  struct enable_if_at_least_one_is_range
    : thrust::detail::enable_if<
        is_range<T1>::value || is_range<T2>::value,
        Result
      >
{};

template<typename T, typename Result>
  struct lazy_enable_if_range
    : thrust::detail::lazy_enable_if<
        is_range<T>::value,
        Result
      >
{};

template<typename T1, typename T2, typename Result>
  struct lazy_enable_if_ranges
    : thrust::detail::lazy_enable_if<
        is_range<T1>::value && is_range<T2>::value,
        Result
      >
{};

template<typename T1, typename T2, typename Result>
  struct lazy_enable_if_range_and_scalar
    : thrust::detail::lazy_enable_if<
        is_range<T1>::value && is_not_range<T2>::value,
        Result
      >
{};

template<typename T1, typename T2, typename Result>
  struct lazy_enable_if_scalar_and_range
    : thrust::detail::lazy_enable_if<
        is_not_range<T1>::value && is_range<T2>::value,
        Result
      >
{};

template<typename T1, typename T2, typename Result>
  struct lazy_enable_if_at_least_one_is_range
    : thrust::detail::lazy_enable_if<
        is_range<T1>::value || is_range<T2>::value,
        Result
      >
{};

template<typename T, typename Result = void>
  struct disable_if_range
    : thrust::detail::disable_if<
        is_range<T>::value,
        Result
      >
{};

template<typename T, typename Result>
  struct lazy_disable_if_range
    : thrust::detail::lazy_disable_if<
        is_range<T>::value,
        Result
      >
{};


} // end detail


template<typename Range>
  struct range_iterator
    : thrust::detail::eval_if<
        thrust::detail::is_const<Range>::value,
        detail::range_const_iterator<
          typename thrust::detail::remove_const<Range>::type
        >,
        detail::range_mutable_iterator<Range>
      >
{};

template<typename Range>
  struct range_value
    : thrust::iterator_value<
        typename range_iterator<Range>::type
      >
{};

template<typename Range>
  struct range_difference
    : thrust::iterator_difference<
        typename range_iterator<Range>::type
      >
{};


} // end strange

