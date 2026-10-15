#include "xs/scatt_data.h"

#include <cstring>
#include <iostream>
#include <vector>

#include "error.h"
#include "math_func.h"

namespace theta {

//-------------------------------
// ScatterXsVector
ScatterXsVector::ScatterXsVector() = default;

ScatterXsVector::ScatterXsVector(const ScatterXsVector& other)
{
  this->target_g0_ = other.target_g0();
  this->target_g1_ = other.target_g1();
  this->length_ = other.length();
  if (other.data_to() && length() > 0) {
    to_ = new double[length_];
    std::memcpy(to_, other.to_, length_ * sizeof(double));
    acc_prob_ = new double[length_ + 1];
    std::memcpy(acc_prob_, other.acc_prob_, (length_ + 1) * sizeof(double));
  }
}

ScatterXsVector& ScatterXsVector::operator=(const ScatterXsVector& other)
{
  if (this == &other) {
    return *this;
  }
  free_memory();

  this->target_g0_ = other.target_g0();
  this->target_g1_ = other.target_g1();
  this->length_ = other.length();
  if (other.data_to() && length() > 0) {
    to_ = new double[length_];
    std::memcpy(to_, other.to_, length_ * sizeof(double));
    acc_prob_ = new double[length_ + 1];
    std::memcpy(acc_prob_, other.acc_prob_, (length_ + 1) * sizeof(double));
  }
  return *this;
}

ScatterXsVector::~ScatterXsVector()
{
  free_memory();
}

double ScatterXsVector::get_to(int64_t target_g) const
{
  if (target_g < target_g0() || target_g > target_g1()) {
    return 0.0;
  }
  return to_[target_g - target_g0()];
}

void ScatterXsVector::set_to(int64_t target_g, double val)
{
  if (target_g >= target_g0() && target_g <= target_g1()) {
    to_[target_g - target_g0()] = val;
  }
}

double ScatterXsVector::get_acc_prob(int64_t target_g) const
{
  if (target_g < target_g0() || target_g > target_g1()) {
    return 0.0;
  }
  return acc_prob_[target_g - target_g0()];
}

void ScatterXsVector::set_acc_prob(int64_t target_g, double val)
{
  if (target_g >= target_g0() && target_g <= target_g1()) {
    acc_prob_[target_g - target_g0()] = val;
  }
}

void ScatterXsVector::allocate(int64_t size_to)
{
  free_memory();
  if (size_to > 0) {
    length_ = size_to;
    to_ = new double[size_to];
    std::memset(to_, 0, size_to * sizeof(double));
    acc_prob_ = new double[size_to + 1];
    std::memset(acc_prob_, 0, (size_to + 1) * sizeof(double));
  }
}

void ScatterXsVector::trim_and_set(const std::vector<double>& input)
{
  // Clear previous data
  clear();

  // If the input vector is empty, do nothing and keep default values
  if (input.empty()) {
    return;
  }

  // Find the index of the first non-zero value
  int64_t i0 = -1;
  for (int64_t i = 0; i < static_cast<int64_t>(input.size()); ++i) {
    if (input[i] != 0.0) {
      i0 = i;
      break;
    }
  }

  // Find the index of the last non-zero value
  int64_t i1 = -1;
  for (int64_t i = static_cast<int64_t>(input.size()) - 1; i >= 0; --i) {
    if (input[i] != 0.0) {
      i1 = i;
      break;
    }
  }

  // If no non-zero values were found, exit the function
  if (i0 == -1 || i1 == -1) {
    target_g0_ = -1;
    target_g1_ = -1;
    length_ = 0;
    return;
  }

  // Set ig0 and ig1 to mark the boundaries of the trimmed data
  target_g0_ = i0;
  target_g1_ = i1;

  // Allocate memory for the trimmed section; span_length is the number of
  // elements to copy
  int64_t span_length = i1 - i0 + 1;
  allocate(span_length);

  // Copy the trimmed portion from the input vector into the to_ array
  for (int64_t idx = 0; idx < span_length; ++idx) {
    to_[idx] = input[i0 + idx];
  }
}

void ScatterXsVector::clear()
{
  free_memory();
  target_g0_ = 0;
  target_g1_ = 0;
  length_ = 0;
}

void ScatterXsVector::free_memory()
{
  // free memory of to_
  if (to_) {
    delete[] to_;
    to_ = nullptr;
  }

  // free memory of acc_prob
  if (acc_prob_) {
    delete[] acc_prob_;
    acc_prob_ = nullptr;
  }
}

//-------------------------------
// ScatterXs
ScatterXs::ScatterXs() = default;

ScatterXs::ScatterXs(int64_t num_groups, int64_t num_orders)
{
  allocate(num_groups, num_orders);
}

ScatterXs::ScatterXs(const ScatterXs& other)
{
  num_groups_ = 0;
  num_orders_ = 0;
  data = nullptr;
  *this = other;
}

ScatterXs& ScatterXs::operator=(const ScatterXs& other)
{
  if (this == &other) {
    return *this;
  }

  clear();

  this->num_groups_ = other.num_groups_;
  this->num_orders_ = other.num_orders_;
  if (num_groups_ > 0 && num_orders_ > 0) {
    data = new ScatterXsVector*[num_groups_];
    for (int64_t origin_g = 0; origin_g < num_groups_; ++origin_g) {
      data[origin_g] = new ScatterXsVector[num_orders_];
      for (int64_t order = 0; order < num_orders_; ++order) {
        data[origin_g][order] = other.data[origin_g][order];
      }
    }
  }
  return *this;
}

ScatterXs::~ScatterXs()
{
  clear();
}

void ScatterXs::allocate(int64_t num_groups, int64_t num_orders)
{
  clear();
  if (num_groups <= 0 || num_orders <= 0) {
    return;
  }
  this->num_groups_ = num_groups;
  this->num_orders_ = num_orders;
  data = new ScatterXsVector*[num_groups_];
  for (int64_t origin_g = 0; origin_g < num_groups_; ++origin_g) {
    data[origin_g] = new ScatterXsVector[num_orders_];
  }
}

void ScatterXs::clear()
{
  if (data) {
    for (int64_t origin_g = 0; origin_g < num_groups_; ++origin_g) {
      delete[] data[origin_g];
    }
    delete[] data;
    data = nullptr;
  }
  num_groups_ = 0;
  num_orders_ = 0;
}

// Check if the ScatterXs is empty
bool ScatterXs::empty() const
{
  return (num_groups_ == 0 || num_orders_ == 0 || data == nullptr);
}

ScatterXsVector& ScatterXs::at(int64_t origin_g, int64_t order)
{
  return data[origin_g][order];
}

const ScatterXsVector& ScatterXs::at(int64_t origin_g, int64_t order) const
{
  return data[origin_g][order];
}

int64_t ScatterXs::num_groups() const
{
  return num_groups_;
}

int64_t ScatterXs::num_orders() const
{
  return num_orders_;
}

double ScatterXs::get_to(
  int64_t origin_g, int64_t target_g, int64_t order) const
{

  if (origin_g < 0 || origin_g >= num_groups_ || order < 0 ||
      order >= num_orders_ || target_g < 0 || target_g >= num_groups_) {
    fatal_error("Invalid index in get_to()");
  }
  if (target_g < data[origin_g][order].target_g0() ||
      target_g > data[origin_g][order].target_g1()) {
    return 0.0;
  }
  int64_t g_ = target_g - data[origin_g][order].target_g0();
  return data[origin_g][order].data_to()[g_];
}

void ScatterXs::set_to(
  int64_t origin_g, int64_t target_g, int64_t order, double val)
{
  if (origin_g < 0 || origin_g >= num_groups_ || order < 0 ||
      order >= num_orders_ || target_g < 0 || target_g >= num_groups_) {
    fatal_error("Invalid index in get_to()");
  }
  if (target_g >= data[origin_g][order].target_g0() &&
      target_g <= data[origin_g][order].target_g1()) {
    int64_t g_ = target_g - data[origin_g][order].target_g0();
    data[origin_g][order].data_to()[g_] = val;
  }
}

double ScatterXs::get_acc_prob(
  int64_t origin_g, int64_t target_g, int64_t order) const
{
  if (origin_g < 0 || origin_g >= num_groups_ || order < 0 ||
      order >= num_orders_ || target_g < 0 || target_g >= num_groups_) {
    fatal_error("Invalid index in get_to()");
  }
  if (target_g < data[origin_g][order].target_g0() ||
      target_g > data[origin_g][order].target_g1()) {
    return 0.0;
  }
  int64_t g_ = target_g - data[origin_g][order].target_g0();
  return data[origin_g][order].data_acc_prob()[g_];
}

void ScatterXs::set_acc_prob(
  int64_t origin_g, int64_t target_g, int64_t order, double val)
{
  if (origin_g < 0 || origin_g >= num_groups_ || order < 0 ||
      order >= num_orders_ || target_g < 0 || target_g >= num_groups_) {
    fatal_error("Invalid index in get_to()");
  }
  if (target_g >= data[origin_g][order].target_g0() &&
      target_g <= data[origin_g][order].target_g1()) {
    int64_t g_ = target_g - data[origin_g][order].target_g0();
    data[origin_g][order].data_acc_prob()[g_] = val;
  }
}

//-------------------------------

void ScatterXs::from_2d_vector(
  const std::vector<std::vector<double>>& input, int64_t num_groups)
{
  // if input is empty, clear the ScatterXs
  if (input.empty()) {
    clear();
    return;
  }

  int64_t n_groups_input = static_cast<int64_t>(input.size());
  int64_t n_groups_actual = std::min(num_groups, n_groups_input);

  int64_t n_orders_actual = 1;

  // allocate ScatterXs with (n_groups_actual, n_orders_actual)
  allocate(n_groups_actual, n_orders_actual);

  // input has the same number of columns in each row
  for (int64_t origin_g = 0; origin_g < n_groups_actual; ++origin_g) {
    for (int64_t order = 0; order < n_orders_actual; ++order) {
      // allocate ScatterXsVector with length = 1
      std::vector<double> sub_vector(
        input[origin_g].begin(), input[origin_g].begin() + n_groups_actual);
      data[origin_g][order].trim_and_set(sub_vector);
    }
  }
}

//-------------------------------

void ScatterXs::from_3d_vector(
  const std::vector<std::vector<std::vector<double>>>& input,
  int64_t num_groups, int64_t num_orders)
{
  // if input is empty, clear the ScatterXs
  if (input.empty()) {
    clear();
    return;
  }

  int64_t n_groups_input = static_cast<int64_t>(input.size());
  int64_t n_orders_input = static_cast<int64_t>(input[0][0].size());
  int64_t n_groups_actual = std::min(num_groups, n_groups_input);
  int64_t n_orders_actual = std::min(num_orders, n_orders_input);

  allocate(n_groups_actual, n_orders_actual);

  for (int64_t origin_g = 0; origin_g < n_groups_actual; ++origin_g) {
    for (int64_t order = 0; order < n_orders_actual; ++order) {
      std::vector<double> row = extract_row(input, origin_g, order);
      std::vector<double> sub_vector(
        row.begin(), row.begin() + n_groups_actual);
      data[origin_g][order].trim_and_set(sub_vector);
    }
  }
}

} // namespace theta
