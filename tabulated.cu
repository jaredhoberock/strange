#include <strange/tabulated_range.hpp>
#include <thrust/copy.h>
#include <thrust/functional.h>
#include <iostream>

int main()
{
  strange::tabulated_range<thrust::negate<int> > rng(0, 10, thrust::negate<int>());

  std::cout << "negate tabulated: ";
  thrust::copy(rng.begin(), rng.end(), std::ostream_iterator<int>(std::cout, " "));
  std::cout << std::endl;
}


