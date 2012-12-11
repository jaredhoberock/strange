#pragma once

#include <strange/counting_range.hpp>
#include <strange/transform_range.hpp>

namespace strange
{
namespace detail
{


template<typename Function, typename Integral>
  struct tabulated_range_base
{
  typedef counting_range<Integral>                                    counting_range;
  typedef transform_range<Function,typename counting_range::iterator> type;
}; // end tabulated_range_base


} // end detail


template<typename Function, typename Integral = typename Function::argument_type>
  class tabulated_range
    : public detail::tabulated_range_base<Function,Integral>::type
{
  typedef typename detail::tabulated_range_base<Function,Integral>::type super_t;

  public:
    inline __host__ __device__
    tabulated_range(Integral first, Integral last, Function f)
      : super_t(make_counting_range(first, last), f)
    {}

    template<typename CountingRange>
    inline __host__ __device__
    tabulated_range(const CountingRange &rng, Function f)
      : super_t(rng, f)
    {}
}; // end tabulated_range


template<typename Integral, typename Function>
inline __host__ __device__
tabulated_range<Function,Integral> make_tabulated_range(Integral first, Integral last, Function f)
{
  return tabulated_range<Function,Integral>(first, last, f);
} // end make_tabulated_range()


template<typename Integral, typename Function>
inline __host__ __device__
tabulated_range<Function,Integral> make_tabulated_range(Integral last, Function f)
{
  return make_tabulated_range(Integral(0), last, f);
} // end make_tabulated_range(0


} // end strange

