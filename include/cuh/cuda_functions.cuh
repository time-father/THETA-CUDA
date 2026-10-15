//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

#ifndef THETA_CUDA_FUNCTIONS_H
#define THETA_CUDA_FUNCTIONS_H

#ifdef THETA_CUDA
#define HDD __host__ __device__
#include "cuh/cuda_base.cuh"
#include <cstdint>
#include <functional>

namespace theta {
namespace gpu {

template<typename T>
struct less_compare {
  HDD bool operator()(const T& lhs, const T& rhs) const { return lhs < rhs; }
};

template<typename T>
HDD int countl_zero(T value)
{
  if (value == 0) {
    return static_cast<int>(sizeof(T) * 8);
  }
  int count = 0;
  T highest_bit = T(1) << (sizeof(T) * 8 - 1);
  while ((value & highest_bit) == 0) {
    value <<= 1;
    ++count;
  }
  return count;
}

HDD inline size_t bit_floor(size_t value)
{
  constexpr int num_bits = sizeof(value) * 8;
  return size_t(1) << (num_bits - countl_zero(value) - 1);
}

HDD inline size_t bit_ceil(size_t value)
{
  constexpr int num_bits = sizeof(value) * 8;
  return size_t(1) << (num_bits - countl_zero(value - 1));
}

template<typename It, typename T, typename Cmp>
HDD It branchless_lower_bound(It begin, It end, const T& value, Cmp compare)
{
  size_t length = static_cast<size_t>(end - begin);
  if (length == 0) {
    return end;
  }
  size_t step = bit_floor(length);
  if (step != length && compare(begin[step], value)) {
    length -= step + 1;
    if (length == 0) {
      return end;
    }
    step = bit_ceil(length);
    begin = end - step;
  }
  for (step /= 2; step != 0; step /= 2) {
    if (compare(begin[step], value)) {
      begin += step;
    }
  }
  return begin + compare(*begin, value);
}

template<typename It, typename T>
HDD It find_lower_bound(It begin, It end, const T& value)
{
#ifdef THETA_CUDA
  return branchless_lower_bound(begin, end, value, less_compare<T> {});
#else
  return branchless_lower_bound(begin, end, value, std::less<T> {});
#endif
}

template<typename It, typename T>
HDD size_t binary_search(It begin, It end, const T& value)
{
  if (*begin >= value) {
    return 0;
  }
  return static_cast<size_t>(find_lower_bound(begin, end, value) - begin - 1);
}

template<typename It, typename T, typename Cmp>
HDD It branchless_upper_bound(It begin, It end, const T& value, Cmp compare)
{
  size_t length = static_cast<size_t>(end - begin);
  if (length == 0) {
    return end;
  }
  size_t step = bit_floor(length);
  if (step != length && !compare(value, begin[step])) {
    length -= step + 1;
    if (length == 0) {
      return end;
    }
    step = bit_ceil(length);
    begin = end - step;
  }
  for (step /= 2; step != 0; step /= 2) {
    if (!compare(value, begin[step])) {
      begin += step;
    }
  }
  return begin + !compare(value, *begin);
}

template<typename It, typename T>
HDD It find_upper_bound(It begin, It end, const T& value)
{
  return branchless_upper_bound(begin, end, value, less_compare<T> {});
}

template<typename It, typename T>
HDD int64_t binary_search_upper(It begin, It end, const T& value)
{
  if (*begin > value) {
    return -1;
  }
  return static_cast<int64_t>(find_upper_bound(begin, end, value) - begin - 1);
}

} // namespace gpu
} // namespace theta
#else
#define HDD
#endif

#endif // THETA_CUDA_FUNCTIONS_H
