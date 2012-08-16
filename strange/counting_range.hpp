#pragma once

#include <strange/range.hpp>
#include <thrust/iterator/counting_iterator.h>

namespace strange
{


template<typename T>
  class counting_range
    : public range<thrust::counting_iterator<T> >
{
  typedef range<thrust::counting_iterator<T> > super_t;

  public:
    inline __host__ __device__
    counting_range(T first, T last)
      : super_t(thrust::make_counting_iterator(first),
                thrust::make_counting_iterator(last))
    {}
}; // end counting_range


template<typename T>
inline __host__ __device__
counting_range<T> make_counting_range(T first, T last)
{
  return counting_range<T>(first, last);
} // end make_counting_range()


template<typename T>
inline __host__ __device__
counting_range<T> make_counting_range(T last)
{
  return counting_range<T>(T(0), last);
} // end make_counting_range()


} // end strange

