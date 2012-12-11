#include <strange/cyclic_range.hpp>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <vector>
#include <iostream>

int main()
{
  thrust::device_vector<int> vec(3);
  vec[0] = 1;
  vec[1] = 2;
  vec[2] = 3;

  std::cout << "original: ";
  thrust::copy(vec.begin(), vec.end(), std::ostream_iterator<int>(std::cout, " "));
  std::cout << std::endl;

  size_t num_cycles = 3;

  strange::cyclic_range<thrust::device_vector<int>::iterator> rng(vec, num_cycles * vec.size());

  std::cout << "cycled: ";
  thrust::copy(rng.begin(), rng.end(), std::ostream_iterator<int>(std::cout, " "));
  std::cout << std::endl;
}

