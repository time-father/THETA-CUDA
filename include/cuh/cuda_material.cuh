#ifndef THETA_CUDA_MATERIAL_H
#define THETA_CUDA_MATERIAL_H
#ifdef THETA_CUDA
#include "cuh/cuda_interface.cuh"
#include "cuh/cuda_base.cuh"
#include "material.h"

#include <cuda.h>
#include <memory>
#include <vector>

namespace theta {
namespace gpu {

struct CudaMatXs {
  int64_t n_mat;
  int64_t n_groups;
  int64_t n_orders;

  const real_t* __restrict__ sigma_t;
  const real_t* __restrict__ sigma_s; // P0 scatter matrix [mat, g_in, g_out]
  // High-order dense scatter matrix [mat, l, g_in, g_out]
  const real_t* __restrict__ sigma_s_lg;
  // High-order sparse row metadata [mat, l, g_in]
  const int64_t* __restrict__ sigma_s_row_g0;
  const int64_t* __restrict__ sigma_s_row_g1;
  const int64_t* __restrict__ sigma_s_row_offset;
  const real_t* __restrict__ sigma_s_row_values;
  const real_t* __restrict__ nu_sigma_f;
  const real_t* __restrict__ chi;
  const char* __restrict__ fissionable;
};

__device__ extern CudaMatXs* __restrict__ d_mats;
extern CudaMatXs* h_mats;

void cuda_free_materials();

void cuda_load_material(
  const std::vector<std::unique_ptr<theta::Material>>& host_materials);

} // namespace gpu
} // namespace theta
#endif
#endif
