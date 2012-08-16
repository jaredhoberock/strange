#pragma once

#include <strange/range.hpp>
#include <thrust/functional.h>
#include <strange/counting_range.hpp>
#include <strange/transform_range.hpp>
#include <strange/permutation_range.hpp>

namespace strange
{
namespace detail
{

template<typename Integral>
  struct stride_functor
    : thrust::unary_function<Integral,Integral>
{
  Integral stride_size;

  inline __host__ __device__
  stride_functor(Integral stride_size)
    : stride_size(stride_size)
  {}

  inline __host__ __device__
  Integral operator()(Integral i) const
  {
    return i * stride_size;
  }
};


template<typename RandomAccessIterator>
  struct strided_range_base
{
  typedef typename thrust::iterator_difference<RandomAccessIterator>::type                   difference_type;
  typedef counting_range<difference_type>                                                    counting_rng;
  typedef transform_range<stride_functor<difference_type>, typename counting_rng::iterator>  transform_rng;
  typedef permutation_range<RandomAccessIterator, typename transform_rng::iterator>          type;
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
                  difference_type stride_size)
      : super_t(
          make_permutation_range(
            rng,
            make_transform_range(
              make_counting_range(rng.size()),
              detail::stride_functor<difference_type>(stride_size)
            )
          )
        )
    {}

    template<typename Range>
    inline __host__ __device__
    strided_range(const Range &rng,
                  difference_type stride_size)
      : super_t(
          make_permutation_range(
            rng,
            make_transform_range(
              make_counting_range(rng.size()),
              detail::stride_functor<difference_type>(stride_size)
            )
          )
        )
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

