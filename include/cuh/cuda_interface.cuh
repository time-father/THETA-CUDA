#ifndef THETA_CUDA_INTERFACE_H
#define THETA_CUDA_INTERFACE_H

#ifdef THETA_CUDA
#include "cuh/cuda_base.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define CUDA_TIMER(kernel_func, operation_name)                                \
  {                                                                            \
    cudaEvent_t start, stop;                                                   \
    cudaEventCreate(&start);                                                   \
    cudaEventCreate(&stop);                                                    \
    cudaEventRecord(start);                                                    \
    kernel_func;                                                               \
    cudaEventRecord(stop);                                                     \
    cudaEventSynchronize(stop);                                                \
    float elapsed_time;                                                        \
    cudaEventElapsedTime(&elapsed_time, start, stop);                          \
    std::cout << operation_name << ": " << elapsed_time << " ms \n";           \
    cudaEventDestroy(start);                                                   \
    cudaEventDestroy(stop);                                                    \
    cudaDeviceSynchronize();                                                   \
  }

namespace theta {
namespace gpu {

template<typename T>
cudaError_t cuda_malloc_and_copy(
  T** device_ptr, const std::vector<T>& host_vector)
{
  size_t size = sizeof(T) * host_vector.size();
  cudaError_t err = cudaMalloc(device_ptr, size);
  if (err != cudaSuccess)
    return err;
  err =
    cudaMemcpy(*device_ptr, host_vector.data(), size, cudaMemcpyHostToDevice);
  return err;
}

template<typename T>
cudaError_t cuda_malloc_and_copy(
  T* __restrict__* device_ptr, const std::vector<T>& host_vector)
{
  size_t size = sizeof(T) * host_vector.size();
  void* raw_ptr = nullptr;
  cudaError_t err = cudaMalloc(&raw_ptr, size);
  if (err != cudaSuccess)
    return err;
  *device_ptr = static_cast<T*>(raw_ptr);
  err =
    cudaMemcpy(*device_ptr, host_vector.data(), size, cudaMemcpyHostToDevice);
  return err;
}

// English: Upload a CPU double-precision array as the requested CUDA
// semantic type.
// @param device_ptr Destination device pointer.
// @param host_vector Source CPU double array.
template<typename T>
inline cudaError_t cuda_malloc_and_copy_double_as(
  T** device_ptr, const std::vector<double>& host_vector)
{
  std::vector<T> converted_host_vector;
  converted_host_vector.resize(host_vector.size());

  // English: Convert explicitly element by element instead of changing the
  // numerical meaning through a raw byte copy.
  for (size_t i = 0; i < host_vector.size(); ++i) {
    converted_host_vector[i] = static_cast<T>(host_vector[i]);
  }

  size_t size = sizeof(T) * converted_host_vector.size();
  cudaError_t err = cudaMalloc(reinterpret_cast<void**>(device_ptr), size);
  if (err != cudaSuccess)
    return err;

  err = cudaMemcpy(
    *device_ptr, converted_host_vector.data(), size, cudaMemcpyHostToDevice);
  return err;
}

template<typename T>
inline cudaError_t cuda_malloc_and_copy_double_as(
  T* __restrict__* device_ptr, const std::vector<double>& host_vector)
{
  std::vector<T> converted_host_vector;
  converted_host_vector.resize(host_vector.size());

  // English: Convert explicitly element by element instead of changing the
  // numerical meaning through a raw byte copy.
  for (size_t i = 0; i < host_vector.size(); ++i) {
    converted_host_vector[i] = static_cast<T>(host_vector[i]);
  }

  size_t size = sizeof(T) * converted_host_vector.size();
  void* raw_ptr = nullptr;
  cudaError_t err = cudaMalloc(&raw_ptr, size);
  if (err != cudaSuccess)
    return err;
  *device_ptr = static_cast<T*>(raw_ptr);

  err = cudaMemcpy(
    *device_ptr, converted_host_vector.data(), size, cudaMemcpyHostToDevice);
  return err;
}

// English: Compatibility wrapper for existing flux uploads; new code should
// use the semantically explicit template interface.
inline cudaError_t cuda_malloc_and_copy_double_as_real(
  real_t** device_ptr, const std::vector<double>& host_vector)
{
  return cuda_malloc_and_copy_double_as(device_ptr, host_vector);
}

inline cudaError_t cuda_malloc_and_copy_double_as_real(
  real_t* __restrict__* device_ptr, const std::vector<double>& host_vector)
{
  return cuda_malloc_and_copy_double_as(device_ptr, host_vector);
}

template<>
inline cudaError_t cuda_malloc_and_copy<bool>(
  bool** device_ptr, const std::vector<bool>& host_vector)
{
  size_t size = sizeof(bool) * host_vector.size();
  cudaError_t err = cudaMalloc(device_ptr, size);
  if (err != cudaSuccess)
    return err;

  std::vector<char> temp;
  temp.reserve(host_vector.size());
  for (const auto& val : host_vector) {
    temp.push_back(static_cast<char>(val));
  }

  err = cudaMemcpy(*device_ptr, temp.data(), size, cudaMemcpyHostToDevice);
  return err;
}

template<>
inline cudaError_t cuda_malloc_and_copy<bool>(
  bool* __restrict__* device_ptr, const std::vector<bool>& host_vector)
{
  size_t size = sizeof(bool) * host_vector.size();
  void* raw_ptr = nullptr;
  cudaError_t err = cudaMalloc(&raw_ptr, size);
  if (err != cudaSuccess)
    return err;
  *device_ptr = static_cast<bool*>(raw_ptr);

  std::vector<char> temp;
  temp.reserve(host_vector.size());
  for (const auto& val : host_vector) {
    temp.push_back(static_cast<char>(val));
  }

  err = cudaMemcpy(*device_ptr, temp.data(), size, cudaMemcpyHostToDevice);
  return err;
}

} // namespace gpu
} // namespace theta
#endif // THETA_CUDA
#endif
