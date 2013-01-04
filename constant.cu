#include <strange/constant_range.hpp>
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

int main()
{
  size_t n = 10;

  strange::constant_range<int> zeros(0, n);

  print_range(std::cout, zeros);

  std::cout << std::endl;

  return 0;
}


