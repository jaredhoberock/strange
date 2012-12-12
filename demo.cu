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
inline __device__ void sequential_copy(Range1 src, Range2 dst)
{
  for(; !src.empty(); src.pop_front(), dst.pop_front())
  {
    dst.front() = src.front();
  }
}

template<typename Range1, typename Range2>
inline __device__ void strided_copy(Range1 src, Range2 dst, int stride, int num_strides)
{
  // XXX we repeat ourself with make_strided_range
  //     we might be able to save some registers if we did
  // sequential_for_each(make_strided_range(zip(src,dst), stride, num_strides), assign_functor)
  // for_each(seq, strided(zip(src,dst), stride, num_strides), assign_functor)
  sequential_copy(strange::make_strided_range(src, stride, num_strides),
                  strange::make_strided_range(dst, stride, num_strides));
}

// XXX maximizing num_full_strides seems like a good idea to keep the per thread overhead low
template<typename Range1, typename Range2>
__device__ void grid_convergent_copy(Range1 src, Range2 dst, int num_full_strides)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;

  // drop the first i items
  // XXX these drops are faster than using pop_front for some reason on the macbook
  src = drop(src, i);
  dst = drop(dst, i);
  strided_copy(src, dst, stride, num_full_strides);

  // get the remainder
  // drop everything but the remaining partial strides
  src.pop_front(num_full_strides * stride);
  dst.pop_front(num_full_strides * stride);

  for(; !src.empty(); src.pop_front(stride), dst.pop_front(stride))
  {
    dst.front() = src.front();
  }
}

__global__ void my_copy_kernel(const int *first, int n, int *result, int num_full_strides)
{
  range<const int*> input(first, first + n);
  range<int *>      output(result, result + n);

  grid_convergent_copy(input, output, num_full_strides);
}

void my_copy(const int *first, int n, int *result)
{
  int block_size = 256;
  int num_blocks = 16;

  // num_full_strides is essentially work_per_thread
  int num_full_strides = n / (num_blocks * block_size);

  my_copy_kernel<<<num_blocks,block_size>>>(first, n, result, num_full_strides);
}

void cuda_memcpy(const int *first, int n, int *result)
{
  cudaMemcpy(result, first, sizeof(int) * n, cudaMemcpyDeviceToDevice);
}

int main()
{
  size_t n = 8 << 20;
  size_t num_trials = 100;
  thrust::device_vector<int> src(n, 7), dst(n, 13);
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

