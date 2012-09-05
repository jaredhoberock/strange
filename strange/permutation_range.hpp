#pragma once

#include <strange/range.hpp>
#include <thrust/iterator/permutation_iterator.h>

namespace strange
{


template<typename ElementIterator, typename IndexIterator>
  class permutation_range
    : public range<thrust::permutation_iterator<ElementIterator,IndexIterator> >
{
  typedef range<thrust::permutation_iterator<ElementIterator,IndexIterator> > super_t;

  public:
    inline __host__ __device__
    permutation_range(ElementIterator elements_first, IndexIterator indices_first, IndexIterator indices_last)
      : super_t(thrust::make_permutation_iterator(elements_first,indices_first),
                thrust::make_permutation_iterator(elements_first,indices_last))
    {}

    template<typename Range1, typename Range2>
    inline __host__ __device__
    permutation_range(Range1 &elements_rng, const Range2 &indices_rng)
      : super_t(thrust::make_permutation_iterator(detail::adl_begin(elements_rng), detail::adl_begin(indices_rng)),
                thrust::make_permutation_iterator(detail::adl_begin(elements_rng), detail::adl_end(indices_rng)))
    {}

    template<typename Range1, typename Range2>
    inline __host__ __device__
    permutation_range(const Range1 &elements_rng, const Range2 &indices_rng)
      : super_t(thrust::make_permutation_iterator(detail::adl_begin(elements_rng), detail::adl_begin(indices_rng)),
                thrust::make_permutation_iterator(detail::adl_begin(elements_rng), detail::adl_end(indices_rng)))
    {}
}; // end permutation_range


template<typename Range1, typename Range2>
inline __host__ __device__
permutation_range<typename range_iterator<const Range1>::type, typename range_iterator<const Range2>::type>
  make_permutation_range(const Range1 &elements_rng, const Range2 &indices_rng)
{
  return permutation_range<typename range_iterator<const Range1>::type, typename range_iterator<const Range2>::type>(elements_rng, indices_rng);
}


template<typename Range1, typename Range2>
inline __host__ __device__
permutation_range<typename range_iterator<Range1>::type, typename range_iterator<const Range2>::type>
  make_permutation_range(Range1 &elements_rng, const Range2 &indices_rng)
{
  return permutation_range<typename range_iterator<Range1>::type, typename range_iterator<const Range2>::type>(elements_rng, indices_rng);
}


} // end strange

