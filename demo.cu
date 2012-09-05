#include <strange/range.hpp>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include "time_invocation_cuda.hpp"
#include <strange/strided_range.hpp>
#include <cassert>
#include <cstdio>

using strange::range;
using strange::slice;

template<typename Range1, typename Range2>
inline __device__ void serial_copy(Range1 src, Range2 dst)
{
  for(; !src.empty(); src.pop_front(), dst.pop_front())
  {
    dst.front() = src.front();
  }
}

template<typename Range1, typename Range2>
__device__ void grid_convergent_copy(Range1 src, Range2 dst)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int grid_size = gridDim.x * blockDim.x;

  //for(; i < src.size(); i += grid_size)
  //{
  //  dst[i] = src[i];
  //}

  strange::strided_range<typename Range1::iterator> strided_src(slice(src, i), grid_size);
  strange::strided_range<typename Range2::iterator> strided_dst(slice(dst, i), grid_size);

  serial_copy(strided_src, strided_dst);
}

__global__ void my_copy_kernel(const int *first, int n, int *result)
{
  range<const int*> input(first, first + n);
  range<int *>      output(result, result + n);

  grid_convergent_copy(input, output);
}

void my_copy(const int *first, int n, int *result)
{
  int block_size = 256;
  int num_blocks = n / block_size;
  if(num_blocks % block_size) ++num_blocks;

  my_copy_kernel<<<num_blocks,block_size>>>(first, n, result);
}

void cuda_memcpy(const int *first, int n, int *result)
{
  cudaMemcpy(result, first, sizeof(int) * n, cudaMemcpyDeviceToDevice);
}

int main()
{
  size_t n = 8 << 20;
  //size_t num_trials = 100;
  size_t num_trials = 1;
  thrust::device_vector<int> src(n), dst(n);
  thrust::sequence(src.begin(), src.end());

  size_t num_bytes = 2 * sizeof(int) * n;
  float gigabytes = float(num_bytes) / (1 << 30);

  int *first  = thrust::raw_pointer_cast(src.data());
  int *result = thrust::raw_pointer_cast(dst.data());

  // first validate my_copy works
  my_copy(first, n, result);
  assert(src == dst);

  float cuda_memcpy_msecs = time_invocation_cuda(num_trials, cuda_memcpy, first, n, result);
  float cuda_memcpy_bandwidth = gigabytes / (cuda_memcpy_msecs / 1000);

  std::cerr << "CUDA error: " << cudaGetErrorString(cudaGetLastError()) << std::endl;

  float my_copy_msecs = time_invocation_cuda(num_trials, my_copy, first, n, result);
  float my_copy_bandwidth = gigabytes / (my_copy_msecs / 1000);

  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, 0);

  std::cerr << "CUDA error: " << cudaGetErrorString(cudaGetLastError()) << std::endl;

  std::cout << std::endl;

  std::cout << "device: " << props.name << std::endl;

  std::cout << "cuda_memcpy_msecs: " << cuda_memcpy_msecs << std::endl;
  std::cout << "cuda_memcpy_bandwidth: " << cuda_memcpy_bandwidth << " GB/s" << std::endl;

  std::cout << std::endl;

  std::cout << "my_copy_msecs: " << my_copy_msecs << std::endl;
  std::cout << "my_copy_bandwidth: " << my_copy_bandwidth << " GB/s" << std::endl;

  std::cout << std::endl;

  std::cout << "my_copy speedup: " << my_copy_bandwidth / cuda_memcpy_bandwidth << std::endl;
}

