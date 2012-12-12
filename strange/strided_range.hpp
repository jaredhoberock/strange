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
                  difference_type stride_size,
                  difference_type num_strides)
      : super_t(make_permutation_range(rng, make_linear_range(stride_size, num_strides)))
    {}

    template<typename Range>
    inline __host__ __device__
    strided_range(const Range &rng,
                  difference_type stride_size,
                  difference_type num_strides)
      : super_t(make_permutation_range(rng, make_linear_range(stride_size, num_strides)))
    {}
};


template<typename Range>
inline __host__ __device__
strided_range<typename range_iterator<Range>::type> make_strided_range(Range &rng, typename range_difference<Range>::type stride_size, typename range_difference<Range>::type num_strides)
{
  return strided_range<typename range_iterator<Range>::type>(rng, stride_size, num_strides);
}


template<typename Range>
inline __host__ __device__
strided_range<typename range_iterator<const Range>::type> make_strided_range(const Range &rng, typename range_difference<const Range>::type stride_size, typename range_difference<const Range>::type num_strides)
{
  return strided_range<typename range_iterator<const Range>::type>(rng, stride_size, num_strides);
}


} // end strange

