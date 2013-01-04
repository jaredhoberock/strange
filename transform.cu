#include <strange/counting_range.hpp>
#include <strange/transform_range.hpp>
#include <iostream>

template<typename Range>
void print_range(std::ostream &os, Range rng)
{
  typedef typename strange::range_value<Range>::type value_type;

  for(; rng.size() != 1; rng.pop_front())
  {
    os << rng.front() << " ";
  }

  // omit the final separator
  os << rng.front();
}

struct square_int
{
  typedef int result_type;

  __host__ __device__
  int operator()(int x)
  {
    return x * x;
  }
};

int main()
{
  print_range(std::cout, strange::make_transform_range(strange::make_counting_range(10), square_int()));

  std::cout << std::endl;

  return 0;
}

