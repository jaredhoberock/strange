#pragma once

#include <strange/range.hpp>
#include <thrust/iterator/transform_iterator.h>

namespace strange
{


template<typename Function, typename Iterator>
  class transform_range
    : public range<thrust::transform_iterator<Function,Iterator> >
{
  typedef range<thrust::transform_iterator<Function,Iterator> > super_t;

  public:
    inline __host__ __device__
    transform_range(Iterator first, Iterator last, Function f)
      : super_t(thrust::make_transform_iterator(first,f),
                thrust::make_transform_iterator(last,f))
    {}

    template<typename Range>
    inline __host__ __device__
    transform_range(Range &rng, Function f)
      : super_t(thrust::make_transform_iterator(adl_begin(rng), f),
                thrust::make_transform_iterator(adl_end(rng), f))
    {}

    template<typename Range>
    inline __host__ __device__
    transform_range(const Range &rng, Function f)
      : super_t(thrust::make_transform_iterator(adl_begin(rng), f),
                thrust::make_transform_iterator(adl_end(rng), f))
    {}
}; // end transform_range


template<typename Function, typename Iterator>
inline __host__ __device__
transform_range<Function, Iterator> make_transform_range(Iterator first, Iterator last, Function f)
{
  return transform_range<Function,Iterator>(first, last, f);
}


template<typename Function, typename Range>
inline __host__ __device__
transform_range<Function, typename range_iterator<Range>::type> make_transform_range(Range &rng, Function f)
{
  return transform_range<Function,typename range_iterator<Range>::type>(rng, f);
}


template<typename Function, typename Range>
inline __host__ __device__
transform_range<Function, typename range_iterator<const Range>::type> make_transform_range(const Range &rng, Function f)
{
  return transform_range<Function,typename range_iterator<const Range>::type>(rng, f);
}


} // end strange


