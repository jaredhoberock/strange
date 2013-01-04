#include <strange/zip_range.hpp>
#include <strange/counting_range.hpp>
#include <strange/linear_range.hpp>
#include <iostream>

template<typename T1, typename T2, typename T3>
std::ostream &operator<<(std::ostream &os, thrust::tuple<T1,T2,T3> t)
{
  return os << "(" << thrust::get<0>(t) << ", " << thrust::get<1>(t) << ", " << thrust::get<2>(t) << ")";
}

template<typename Range>
void print_range(std::ostream &os, Range rng)
{
  typedef typename strange::range_value<Range>::type value_type;

  for(; !rng.empty(); rng.pop_front())
  {
    os << rng.front() << std::endl;
  }
}

int main()
{
  size_t n = 10;
  // [0, 1, 2, 3, ..., n)
  strange::counting_range<int> begin_at_zero(0, n);

  // [10, 11, 12, 13, ..., 10 + n)
  strange::counting_range<int> begin_at_ten(10, 10 + n);

  // [0, 3, 6, 9, ..., 10 * 3) 
  strange::linear_range<int>   count_by_three(3, n);

  print_range(std::cout, strange::make_zip_range(begin_at_zero, begin_at_ten, count_by_three));

  return 0;
}

