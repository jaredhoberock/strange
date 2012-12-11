#pragma once

#include <strange/range.hpp>
#include <thrust/iterator/counting_iterator.h>

namespace strange
{
namespace detail
{


template<typename Integral>
  struct make_signed
{
  typedef Integral type; 
};

template<> struct make_signed<unsigned char>      { typedef signed char type; };
template<> struct make_signed<unsigned short>     { typedef short type; };
template<> struct make_signed<unsigned int>       { typedef int type; };
template<> struct make_signed<unsigned long>      { typedef long type; };
template<> struct make_signed<unsigned long long> { typedef long long type; };


template<typename Integral>
  struct counting_range_base
{
  typedef typename make_signed<
    Integral
  >::type signed_integral;

  typedef thrust::counting_iterator<Integral,thrust::use_default,thrust::use_default,signed_integral> counting_iter;

  typedef range<counting_iter> type;

  inline __host__ __device__
  static type make(Integral first, Integral last)
  {
    return type(counting_iter(first), counting_iter(last));
  }
};


} // end detail


template<typename Integral>
  class counting_range
    : public detail::counting_range_base<Integral>::type
{
  typedef typename detail::counting_range_base<Integral>::type super_t;

  public:
    inline __host__ __device__
    counting_range(Integral first, Integral last)
      : super_t(detail::counting_range_base<Integral>::make(first, last))
    {}
}; // end counting_range


template<typename Integral>
inline __host__ __device__
counting_range<Integral> make_counting_range(Integral first, Integral last)
{
  return counting_range<Integral>(first, last);
} // end make_counting_range()


template<typename Integral>
inline __host__ __device__
counting_range<Integral> make_counting_range(Integral last)
{
  return counting_range<Integral>(Integral(0), last);
} // end make_counting_range()


} // end strange

