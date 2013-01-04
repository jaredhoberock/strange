#pragma once

#include <strange/range.hpp>
#include <thrust/iterator/constant_iterator.h>

namespace strange
{


template<typename Value>
  class constant_range
    : public range<
        thrust::constant_iterator<Value>
      >
{
  private:
    typedef range<
      thrust::constant_iterator<Value>
    > super_t;

  public:
    typedef typename super_t::value_type      value_type;
    typedef typename super_t::difference_type difference_type;

    inline __host__ __device__
    constant_range(value_type c, difference_type n)
      : super_t(thrust::constant_iterator<Value>(c, difference_type(0)),
                thrust::constant_iterator<Value>(c, n))
    {}
};


template<typename Value, typename Size>
inline __host__ __device__
constant_range<Value> make_constant_range(Value c, Size n)
{
  return constant_range<Value>(c,n);
}


} // end newton

