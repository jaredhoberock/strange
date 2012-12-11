#pragma once

#include <strange/range.hpp>
#include <strange/tabulated_range.hpp>

namespace strange
{
namespace detail
{


template<typename T>
  struct linear_functor
    : thrust::unary_function<T,T>
{
  T a;

  inline __host__ __device__
  linear_functor(T a)
    : a(a)
  {}

  template<typename U>
  inline __host__ __device__
  T operator()(const U &x) const
  {
    return a * x;
  }
};


template<typename T>
  struct linear_range_base
{
  // XXX tabulating linear_function is inefficient because every dereference incurs a multiply
  //     instead, we should implement it through a novel iterator similar to counting_iterator which
  //     increments by a instead of 1
  //     this will make size() incur a divide but this is typically only done once per loop
  //     alternatively, we could cache the size inside the range and only do the divide when taking the difference of iterators
  typedef tabulated_range<linear_functor<T>, T> type;
};


} // end detail


template<typename T>
  class linear_range
    : public detail::linear_range_base<T>::type
{
  typedef typename detail::linear_range_base<T>::type super_t;

  public:
    inline __host__ __device__
    linear_range(T a, T size)
      : super_t(T(0), size, detail::linear_functor<T>(a))
    {}
};


template<typename T>
inline __host__ __device__
linear_range<T> make_linear_range(T a, T size)
{
  return linear_range<T>(a,size);
}


} // end strange

