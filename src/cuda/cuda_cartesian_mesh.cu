#include "cartesian_mesh.h"
#include "cuh/cuda_cartesian_mesh.cuh"
#include "cuh/cuda_functions.cuh"
#include "cuh/cuda_interface.cuh"
#include <cuda_runtime.h>
#include <vector>

#ifdef THETA_CUDA
namespace theta {
namespace gpu {

__device__ cudaDivision** __restrict__ d_divisions;
std::vector<cudaDivision*> h_divisions;

cudaDivision::cudaDivision(const Division* host_division)
{
  n_ = host_division->n_mesh();
  n_mat_ = host_division->n_mat();
  // English: Mesh boundaries are geometry data and must use geometry
  // precision on the device.
  cuda_malloc_and_copy_double_as(&edges_, host_division->mesh_edges());
  lower_boundary_type_ = host_division->lower_boundary_type();
  upper_boundary_type_ = host_division->upper_boundary_type();
}

// Host function to copy Division data to GPU
void cuda_load_divisions(std::vector<const theta::Division*>& host_divisions)
{
  int n_dimensions = host_divisions.size();
  h_divisions.clear();
  std::vector<cudaDivision*> h_div_ptrs(n_dimensions);
  for (int i = 0; i < n_dimensions; i++) {
    const auto& host_div = host_divisions[i];

    cudaDivision* h_single_div = new cudaDivision(host_div);

    cudaDivision* d_single_div;
    cudaMalloc((void**)&d_single_div, sizeof(cudaDivision));

    cudaMemcpy(
      d_single_div, h_single_div, sizeof(cudaDivision), cudaMemcpyHostToDevice);

    h_div_ptrs[i] = d_single_div;

    h_divisions.push_back(h_single_div);
  }
  cudaDivision** d_divs_array;
  cudaMalloc((void**)&d_divs_array, n_dimensions * sizeof(cudaDivision*));

  cudaMemcpy(d_divs_array, h_div_ptrs.data(),
    n_dimensions * sizeof(cudaDivision*), cudaMemcpyHostToDevice);

  cudaMemcpyToSymbol(d_divisions, &d_divs_array, sizeof(cudaDivision*));
}

// Free CUDA Division memory
void cuda_free_divisions()
{
  for (size_t i = 0; i < h_divisions.size(); i++) {
    cudaFree(h_divisions[i]->edges_);
    delete h_divisions[i];
  }
}

} // namespace gpu
} // namespace theta
#endif
