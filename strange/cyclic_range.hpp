#pragma once

#include <strange/range.hpp>
#include <thrust/functional.h>
#include <strange/tabulated_range.hpp>
#include <strange/permutation_range.hpp>

namespace strange
{
namespace detail
{


template<typename Integral>
  struct modulus_functor
    : thrust::unary_function<Integral,Integral>
{
  Integral denominator;

  inline __host__ __device__
  modulus_functor(Integral denominator)
    : denominator(denominator)
  {}

  inline __host__ __device__
  Integral operator()(Integral i) const
  {
    return i % denominator;
  }
};


template<typename RandomAccessIterator>
  struct cyclic_range_base
{
  typedef typename thrust::iterator_difference<RandomAccessIterator>::type          index_type;
  typedef tabulated_range<modulus_functor<index_type>, index_type>                  tabulated_rng;
  typedef permutation_range<RandomAccessIterator, typename tabulated_rng::iterator> type;
};


} // end detail


template<typename RandomAccessIterator>
  class cyclic_range
    : public detail::cyclic_range_base<RandomAccessIterator>::type
{
  typedef typename detail::cyclic_range_base<RandomAccessIterator>::type   super_t;
  typedef typename thrust::iterator_difference<RandomAccessIterator>::type index_type;

  public:
    typedef typename super_t::difference_type difference_type;

    template<typename Range>
    inline __host__ __device__
    cyclic_range(Range &rng, difference_type n)
      : super_t(rng, make_tabulated_range<index_type>(0, n, detail::modulus_functor<index_type>(size(rng))))
    {}
};


template<typename Range>
inline __host__ __device__
cyclic_range<typename range_iterator<Range>::type> make_cyclic_range(Range &rng, typename range_difference<Range>::type n)
{
  return cyclic_range<typename range_iterator<Range>::type>(rng, n);
}


template<typename Range>
inline __host__ __device__
cyclic_range<typename range_iterator<const Range>::type> make_cyclic_range(const Range &rng, typename range_difference<Range>::type n)
{
  return cyclic_range<typename range_iterator<const Range>::type>(rng, n);
}


} // end strange

