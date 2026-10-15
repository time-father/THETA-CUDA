#ifndef THETA_CUDA_CARTESIAN_MESH_H
#define THETA_CUDA_CARTESIAN_MESH_H

#ifdef THETA_CUDA
#include "cartesian_mesh.h"
#include "cuh/cuda_base.cuh"
#include "cuh/cuda_cartesian_mesh.cuh"
#include "cuh/cuda_functions.cuh"

namespace theta {
namespace gpu {

class cudaDivision {
public:
  int64_t n_;
  int64_t n_mat_;
  geom_t* __restrict__ edges_;
  BoundaryType lower_boundary_type_;
  BoundaryType upper_boundary_type_;

  // Accessors ---------------------------------------------------------------
  const int64_t& n() { return n_; }
  const int64_t& n_mat() { return n_mat_; }
  geom_t edge(int64_t i) const { return edges_[i]; }
  const BoundaryType& lower_boundary_type() { return lower_boundary_type_; }
  const BoundaryType& upper_boundary_type() { return upper_boundary_type_; }

  cudaDivision(const Division* host_division);

  // Methods -----------------------------------------------------------------
  HDD inline int64_t get_index(geom_t x) const
  {
    int64_t i = gpu::binary_search(&edges_[0], &edges_[n_], x);
    // if (i < 0 || i >= n_) {
    //   throw std::out_of_range("Ray is outside of the geometry.");
    // }
    return i;
  }

  HDD inline int64_t get_mesh_index(geom_t x, bool& in_mesh) const
  {
    int64_t i = gpu::binary_search(&edges_[0], &edges_[n_], x);
    // if (i < 0 || i >= n_) {
    //   in_mesh = false;
    //   // Set to a valid index
    //   i = std::max(i, static_cast<int64_t>(0));
    //   i = std::min(i, n_ - 1);
    // }
    return i;
  }
};

void cuda_load_divisions(std::vector<const theta::Division*>& host_divisions);

void cuda_free_divisions();

__device__ extern cudaDivision** __restrict__ d_divisions;
extern std::vector<cudaDivision*> h_divisions;

} // namespace gpu
} // namespace theta

#endif // THETA_CUDA
#endif // THETA_CUDA_CARTESIAN_MESH
