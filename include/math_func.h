#ifndef THETA_MATH_FUNC_H
#define THETA_MATH_FUNC_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <vector>

#include "error.h"

namespace theta {

constexpr double INFTY {std::numeric_limits<double>::max()};
constexpr double PI {3.141592653589793238462643383279502884L};

template<class It, class T>
typename std::iterator_traits<It>::difference_type lower_bound_index(
  It first, It last, const T& value)
{
  if (*first == value) {
    return 0;
  }
  return std::lower_bound(first, last, value) - first - 1;
}

template<class It, class T>
typename std::iterator_traits<It>::difference_type upper_bound_index(
  It first, It last, const T& value)
{
  return std::upper_bound(first, last, value) - first - 1;
}

template<typename Container, typename T>
std::size_t closest(const Container& container, const T& value)
{
  auto it = std::lower_bound(container.begin(), container.end(), value);
  if (it == container.begin()) {
    return 0;
  }
  if (it == container.end()) {
    return container.size() - 1;
  }
  if (*it == value) {
    return std::distance(container.begin(), it);
  }

  auto prev_it = std::prev(it);
  const auto& prev = *prev_it;
  const auto& curr = *it;
  return (std::abs(value - prev) <= std::abs(value - curr)
            ? std::distance(container.begin(), prev_it)
            : std::distance(container.begin(), it));
}

template<typename T>
std::vector<T> extract_row(const std::vector<std::vector<std::vector<T>>>& vec,
  int64_t row_index, int64_t matrix_index)
{
  if (row_index < 0 || row_index >= static_cast<int64_t>(vec.size())) {
    fatal_error("Matrix index is out of range.");
  }

  const auto& matrix = vec[row_index];
  std::vector<T> row;
  row.reserve(matrix.size());
  for (const auto& column : matrix) {
    if (matrix_index < 0 ||
        matrix_index >= static_cast<int64_t>(column.size())) {
      fatal_error("Column index is out of range in one of the rows.");
    }
    row.push_back(column[matrix_index]);
  }
  return row;
}

} // namespace theta

#endif // THETA_MATH_FUNC_H
