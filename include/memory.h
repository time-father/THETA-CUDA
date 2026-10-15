#ifndef THETA_MEMORY_H
#define THETA_MEMORY_H

#include <cstdint>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>

namespace theta {

template<typename T, typename... Args>
typename std::enable_if<!std::is_array<T>::value, std::unique_ptr<T>>::type
make_unique(Args&&... args)
{
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

template<typename T>
typename std::enable_if<std::is_array<T>::value && std::extent<T>::value == 0,
  std::unique_ptr<T>>::type
make_unique(size_t size)
{
  using ElementType = typename std::remove_extent<T>::type;
  return std::unique_ptr<T>(new ElementType[size]());
}

template<typename T, typename... Args>
typename std::enable_if<std::extent<T>::value != 0, void>::type make_unique(
  Args&&...) = delete;

} // namespace theta

#endif // THETA_MEMORY_H
