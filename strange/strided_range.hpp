#pragma once

#include <strange/range.hpp>
#include <thrust/functional.h>
#include <strange/linear_range.hpp>
#include <strange/permutation_range.hpp>

namespace strange
{
namespace detail
{


template<typename RandomAccessIterator>
  struct strided_range_base
{
  typedef typename thrust::iterator_difference<RandomAccessIterator>::type       difference_type;
  typedef linear_range<difference_type>                                          linear_rng;
  typedef permutation_range<RandomAccessIterator, typename linear_rng::iterator> type;
};


template<typename T>
inline __host__ __device__
T divide_ri(T numerator, T denominator)
{
  return (numerator + (denominator - 1)) / denominator;
}


} // end detail


template<typename RandomAccessIterator>
  class strided_range
    : public detail::strided_range_base<RandomAccessIterator>::type
{
  typedef typename detail::strided_range_base<RandomAccessIterator>::type super_t;

  public:
    typedef typename super_t::difference_type difference_type;

    template<typename Range>
    inline __host__ __device__
    strided_range(Range &rng,
                  difference_type stride_size)
      : super_t(make_permutation_range(rng, make_linear_range(stride_size, detail::divide_ri(rng.size(), stride_size))))
    {}

    template<typename Range>
    inline __host__ __device__
    strided_range(const Range &rng,
                  difference_type stride_size)
      : super_t(make_permutation_range(rng, make_linear_range(stride_size, detail::divide_ri(rng.size(), stride_size))))
    {}
};


template<typename Range>
inline __host__ __device__
strided_range<typename range_iterator<Range>::type> make_strided_range(Range &rng, typename range_difference<Range>::type stride_size)
{
  return strided_range<typename range_iterator<Range>::type>(rng, stride_size);
}


template<typename Range>
inline __host__ __device__
strided_range<typename range_iterator<const Range>::type> make_strided_range(Range &rng, typename range_difference<Range>::type stride_size)
{
  return strided_range<typename range_iterator<const Range>::type>(rng, stride_size);
}


} // end strange

