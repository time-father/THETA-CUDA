#ifndef THETA_CARTESIAN_MESH
#define THETA_CARTESIAN_MESH

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "error.h"
#include "math_func.h"
#include "position.h"
#include "pugixml.hpp"
#include "settings.h"


namespace theta {


using MeshIndex = std::array<int64_t, 3>;
using map3d = std::vector<std::vector<std::vector<int64_t>>>;

/*
 * The BoundaryType class encompasses two types of boundary conditions:
 * VACUUM and REFLECTIVE.
 */

enum class BoundaryType {
  VACUUM = 0,
  REFLECTIVE = 1,
};

/*
 * The Division class encompasses data and methods for storing geometry
 * infomation including boundary type, number of mesh, map edges offset, mesh
 * edges and corresponding map id
 */

class Division {
public:
  //==============================================================================
  // Constructor and destructor
  //==============================================================================
  Division() = default;
  Division(pugi::xml_node node);
  Division(const Division* from);
  ~Division() = default;

  //==============================================================================
  // Member functions for Accessors
  //==============================================================================
  int64_t n_mesh() const { return n_mesh_; }
  int64_t n_mat() const { return n_mat_; }
  const std::vector<double>& mesh_edges() const { return mesh_edges_; }
  const std::vector<double>& mesh_center() const { return mesh_center_; }
  double mesh_edges(int64_t i) const { return mesh_edges_[i]; }
  double mesh_center(int64_t i) const { return mesh_center_[i]; }
  const std::vector<int64_t>& maps() const { return maps_; }
  int64_t maps(int64_t i) const { return maps_[i]; }
  const std::vector<int64_t>& offsets() const { return offsets_; }
  int64_t offsets(int64_t i) const { return offsets_[i]; }
  const BoundaryType& lower_boundary_type() const
  {
    return lower_boundary_type_;
  }
  const BoundaryType& upper_boundary_type() const
  {
    return upper_boundary_type_;
  }

  //==============================================================================
  // Member functions for Methods
  //==============================================================================
  //! Find the mesh index for a given position x.
  //! Throws std::out_of_range if x is outside the mesh domain.
  int64_t get_index(double x) const;

  //! Find the mesh index for a given position x.
  // If x is outside the mesh domain, sets in_mesh to false and returns the
  // nearest valid index.
  int64_t get_mesh_index(double x, bool& in_mesh) const;

private:
  // Private data members ----------------------------------------------------
  int64_t n_mesh_ {1};              // Number of mesh for each division
  int64_t n_mat_ {1};               // Number of materials map for each division
  std::vector<double> mesh_edges_;  // Mesh edges position for each division
  std::vector<double> mesh_center_; // Mesh center position for each division
  std::vector<int64_t> maps_;       // Map id of cuurent mesh for each division
  std::vector<int64_t> offsets_;    // Map edges index offsets for each division
  BoundaryType lower_boundary_type_; // Lower boundary type
  BoundaryType upper_boundary_type_; // Upper boundary type
};

template<typename T>
void fill_values(std::vector<T>& arr, T value)
{
  std::fill(arr.begin(), arr.end(), value);
}

//! Convert a 1D vector to a 3D vector
template<typename T>
std::vector<std::vector<std::vector<T>>> convert_to_3d(
  const std::vector<T>& input, int64_t X, int64_t Y, int64_t Z)
{
  if (input.size() != X * Y * Z) {
    fatal_error(
      "Input size does not match the specified dimensions in convert_to_3d.");
  }

  std::vector<std::vector<std::vector<T>>> output(
    Z, std::vector<std::vector<T>>(Y, std::vector<T>(X)));

  int64_t index = 0;
  for (int64_t k = 0; k < Z; ++k) {
    for (int64_t j = 0; j < Y; ++j) {
      for (int64_t i = 0; i < X; ++i) {
        output[k][j][i] = input[index++];
      }
    }
  }

  return output;
}

//! Create model from input.xml
void read_model(pugi::xml_node node);

//! Free memory model
void free_memory_model();

} // namespace theta

#endif // THETA_CARTESIAN_MESH
