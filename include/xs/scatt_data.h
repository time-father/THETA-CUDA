#ifndef THETA_XS_SCATT_DATA_H
#define THETA_XS_SCATT_DATA_H

#include <cstring>
#include <stdexcept>
#include <vector>

#include "math_func.h"

namespace theta {

//----------------------------------------
// ScatterXsVector declaration: Current energy group scatter xs data (sparse
// matrix format)
class ScatterXsVector {
public:
  ScatterXsVector();
  ScatterXsVector(const ScatterXsVector& other);
  ScatterXsVector& operator=(const ScatterXsVector& other);
  ~ScatterXsVector();

  int64_t& target_g0() { return target_g0_; }
  int64_t& target_g1() { return target_g1_; }
  int64_t& length() { return length_; }
  const int64_t& target_g0() const { return target_g0_; }
  const int64_t& target_g1() const { return target_g1_; }
  const int64_t& length() const { return length_; }

  // get and set methods for the 1d array

  double get_to(int64_t target_g) const;
  void set_to(int64_t target_g, double val);
  double get_acc_prob(int64_t target_g) const;
  void set_acc_prob(int64_t target_g, double val);

  // from std::vector<double> to internal storage
  void allocate(int64_t size_to);
  void trim_and_set(const std::vector<double>& input);
  void clear();

  double* data_to() { return to_; }
  const double* data_to() const { return to_; }
  double* data_acc_prob() { return acc_prob_; }
  const double* data_acc_prob() const { return acc_prob_; }

private:
  void free_memory();

  int64_t target_g0_ = 0; // minimum outgoing group
  int64_t target_g1_ = 0; // maximum outgoing group
  int64_t length_ = 0;

  double* to_ = nullptr; // scattering vector, shape=(target_g0_:target_g1_)
  double* acc_prob_ =
    nullptr; // accumulated probability, shape=(target_g0_ - 1:target_g1_)
};

//----------------------------------------
// ScatterXs declaration: 2d array of ScatterXsVector
class ScatterXs {
public:
  ScatterXs();
  ScatterXs(int64_t num_groups, int64_t num_orders);
  ScatterXs(const ScatterXs& other);
  ScatterXs& operator=(const ScatterXs& other);
  ~ScatterXs();

  void allocate(int64_t num_groups, int64_t num_orders);
  void clear();
  bool empty() const;

  // Simple accessor (no boundary check)
  ScatterXsVector& at(int64_t origin_g, int64_t order);
  const ScatterXsVector& at(int64_t origin_g, int64_t order) const;

  int64_t num_groups() const;
  int64_t num_orders() const;

  // get and set methods for the 3d array

  double get_to(int64_t origin_g, int64_t target_g, int64_t order) const;
  void set_to(int64_t origin_g, int64_t target_g, int64_t order, double val);
  double get_acc_prob(int64_t origin_g, int64_t target_g, int64_t order) const;
  void set_acc_prob(
    int64_t origin_g, int64_t target_g, int64_t order, double val);

  // 2d std::vector<double> to internal storage
  // The third dimension has fixed length of 1
  void from_2d_vector(
    const std::vector<std::vector<double>>& input, int64_t num_groups);

  // 3d std::vector<std::vector<double>> to internal storage
  // The length of the third dimension is determined by the size() of the
  // innermost input
  void from_3d_vector(
    const std::vector<std::vector<std::vector<double>>>& input,
    int64_t num_groups, int64_t num_orders);

private:
  int64_t num_groups_ = 0;
  int64_t num_orders_ = 0;
  ScatterXsVector** data = nullptr;
};

} // namespace theta

#endif // THETA_XS_SCATT_DATA_H
