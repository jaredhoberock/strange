#pragma once

#include <strange/range_traits.hpp>
#include <strange/detail/container_traits.hpp>

namespace strange
{


template<typename Range>
inline __host__ __device__
  typename detail::lazy_disable_if_container<
    Range,
    range_iterator<Range>
  >::type
    begin(Range &rng)
{
  return rng.begin();
}

// XXX WAR CUDA's warnings about __host__ __device__ functions calling __host__ functions
// with this overload for containers
template<typename Container>
inline __host__ __device__
  typename detail::lazy_enable_if_container<
    Container,
    range_iterator<Container>
  >::type
    begin(Container &c)
{
#if __CUDA_ARCH__
  return typename range_iterator<Container>::type();
#else
  return c.begin();
#endif
}

template<typename Range>
inline __host__ __device__
  typename detail::lazy_disable_if_container<
    Range,
    range_iterator<const Range>
  >::type
    begin(const Range &rng)
{
  return rng.begin();
}

// XXX WAR CUDA's warnings about __host__ __device__ functions calling __host__ functions
// with this overload for containers
template<typename Container>
inline __host__ __device__
  typename detail::lazy_enable_if_container<
    Container,
    range_iterator<const Container>
  >::type
    begin(const Container &c)
{
#if __CUDA_ARCH__
  return typename range_iterator<const Container>::type();
#else
  return c.begin();
#endif
}

template<typename T, std::size_t N>
inline __host__ __device__
  typename range_iterator<T[N]>::type
    begin(T (&array)[N])
{
  return array;
}

template<typename Range>
inline __host__ __device__
  typename detail::lazy_disable_if_container<
    Range,
    range_iterator<Range>
  >::type
    end(Range &rng)
{
  return rng.end();
}

// XXX WAR CUDA's warnings about __host__ __device__ functions calling __host__ functions
// with this overload for containers
template<typename Container>
inline __host__ __device__
  typename detail::lazy_enable_if_container<
    Container,
    range_iterator<Container>
  >::type
    end(Container &c)
{
#if __CUDA_ARCH__
  return typename range_iterator<Container>::type();
#else
  return c.end();
#endif
}

template<typename Range>
inline __host__ __device__
  typename detail::lazy_disable_if_container<
    Range,
    range_iterator<const Range>
  >::type
    end(const Range &rng)
{
  return rng.end();
}

// XXX WAR CUDA's warnings about __host__ __device__ functions calling __host__ functions
// with this overload for containers
template<typename Container>
inline __host__ __device__
  typename detail::lazy_enable_if_container<
    Container,
    range_iterator<const Container>
  >::type
    end(const Container &c)
{
#if __CUDA_ARCH__
  return typename range_iterator<const Container>::type();
#else
  return c.end();
#endif
}

template<typename T, std::size_t N>
inline __host__ __device__
  typename range_iterator<T[N]>::type
    end(T (&array)[N])
{
  return array + N;
}


} // end strange

