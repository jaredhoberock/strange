#pragma once

#include <thrust/iterator/iterator_traits.h>
#include <strange/range_traits.hpp>
#include <strange/detail/begin_end.hpp>

namespace strange
{

template<typename Range>
inline __host__ __device__
  typename range_difference<const Range>::type
    size(const Range &rng)
{
  return end(rng) - begin(rng);
}

namespace detail
{

// helpers to dispatch begin & end via adl
// when in the scope of a similarly-named overload
template<typename Range>
inline __host__ __device__
  typename range_iterator<Range>::type
    adl_begin(Range &rng)
{
  return begin(rng);
}

template<typename Range>
inline __host__ __device__
  typename range_iterator<const Range>::type
    adl_begin(const Range &rng)
{
  return begin(rng);
}

template<typename Range>
inline __host__ __device__
  typename range_iterator<Range>::type
    adl_end(Range &rng)
{
  return end(rng);
}

template<typename Range>
inline __host__ __device__
  typename range_iterator<const Range>::type
    adl_end(const Range &rng)
{
  return end(rng);
}

template<typename Iterator>
  class input_range
{
  public:
    typedef Iterator                                             iterator;
    typedef typename thrust::iterator_difference<iterator>::type difference_type;

    inline __host__ __device__
    input_range(iterator first, iterator last)
      : m_begin(first), m_end(last)
    {}

    inline __host__ __device__
    iterator begin() const
    {
      return m_begin;
    }

    inline __host__ __device__
    iterator end() const
    {
      return m_end;
    }

    inline __host__ __device__
    bool empty() const
    {
      return begin() == end();
    }

    inline __host__ __device__
    void pop_front()
    {
      ++m_begin;
    }

  private:
    iterator m_begin, m_end;
};


template<typename Iterator>
  class random_access_range
{
  public:
    typedef Iterator                                             iterator;
    typedef typename thrust::iterator_difference<iterator>::type difference_type;

    inline __host__ __device__
    random_access_range(iterator first, iterator last)
      : m_begin(first),
        m_size(last - first)
    {}

    inline __host__ __device__
    iterator begin() const
    {
      return m_begin;
    }

    inline __host__ __device__
    iterator end() const
    {
      return begin() + size();
    }

    inline __host__ __device__
    difference_type size() const
    {
      return m_size;
    }

    inline __host__ __device__
    bool empty() const
    {
      return size() == 0;
    }

    inline __host__ __device__
    void pop_front()
    {
      ++m_begin;
      --m_size;
    }

  private:
    iterator m_begin;
    difference_type m_size;
};


template<typename Iterator>
  struct range_base
    : thrust::detail::eval_if<
        thrust::detail::is_convertible<
          typename thrust::iterator_traversal<Iterator>::type,
          thrust::random_access_traversal_tag
        >::value,
        thrust::detail::identity_<random_access_range<Iterator> >,
        thrust::detail::identity_<input_range<Iterator> >
      >
{};


} // end detail


// XXX specialize for RandomAccessIterator
//     to avoid storing two iterators
template<typename Iterator>
  class range
    : public detail::range_base<Iterator>::type
{
  typedef typename detail::range_base<Iterator>::type super_t;

  public:
    typedef Iterator                                             iterator;
    typedef typename thrust::iterator_value<iterator>::type      value_type;
    typedef typename thrust::iterator_reference<iterator>::type  reference;
    typedef typename thrust::iterator_difference<iterator>::type difference_type;

    inline __host__ __device__
    range(iterator first, iterator last)
      : super_t(first, last)
    {}

    template<typename Range>
    inline __host__ __device__
    range(Range &rng)
      : super_t(detail::adl_begin(rng), detail::adl_end(rng))
    {}

    template<typename Range>
    inline __host__ __device__
    range(const Range &rng)
      : super_t(detail::adl_begin(rng), detail::adl_end(rng))
    {}

    inline __host__ __device__
    reference front() const
    {
      return *super_t::begin();
    }

    inline __host__ __device__
    reference operator[](const difference_type &i) const
    {
      return super_t::begin()[i];
    }
};

template<typename Iterator>
__host__ __device__
range<Iterator> make_range(Iterator first, Iterator last)
{
  return range<Iterator>(first,last);
}

template<typename RandomAccessIterator>
__host__ __device__
range<RandomAccessIterator> make_range(RandomAccessIterator first, typename thrust::iterator_difference<RandomAccessIterator> n)
{
  return range<RandomAccessIterator>(first, first + n);
}

template<typename Range>
__host__ __device__
range<typename range_iterator<Range>::type> make_range(Range &rng)
{
  return range<typename range_iterator<Range>::type>(rng);
}

template<typename Range>
__host__ __device__
range<typename range_iterator<const Range>::type> make_range(const Range &rng)
{
  return range<typename range_iterator<const Range>::type>(rng);
}

template<typename Range, typename Size>
__host__ __device__
range<typename range_iterator<Range>::type>
  slice(Range &rng, Size first)
{
  return make_range(begin(rng) + first, end(rng));
}

template<typename Range, typename Size>
__host__ __device__
range<typename range_iterator<const Range>::type>
  slice(const Range &rng, Size first)
{
  return make_range(begin(rng) + first, end(rng));
}

template<typename Range, typename Size>
__host__ __device__
range<typename range_iterator<Range>::type>
  slice(Range &rng, Size first, Size last)
{
  return make_range(begin(rng) + first, begin(rng) + last);
}

template<typename Range, typename Size>
__host__ __device__
range<typename range_iterator<const Range>::type>
  slice(const Range &rng, Size first, Size last)
{
  return make_range(begin(rng) + first, begin(rng) + last);
}

template<typename Range, typename Size>
__host__ __device__
range<typename range_iterator<Range>::type>
  take(Range &rng, Size n)
{
  return slice(rng, Size(0), n);
}

template<typename Range, typename Size>
__host__ __device__
range<typename range_iterator<const Range>::type>
  take(const Range &rng, Size n)
{
  return slice(rng, Size(0), n);
}


} // end strange

