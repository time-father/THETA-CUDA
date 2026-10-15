#ifndef CUDA_FLAT_SOURCE_DOMAIN
#define CUDA_FLAT_SOURCE_DOMAIN
#include "cuh/cuda_material.cuh"
#include "cuh/cuda_settings.cuh"
#include "flat_source_domain.h"

#ifdef THETA_CUDA
namespace theta {
namespace gpu {

struct CuFlatSourceDomain {
  int64_t stride_x_;
  int64_t stride_y_;
  int64_t stride_z_;

  int* __restrict__ n_tracks_;            // The number of hitting ray
  int64_t* __restrict__ ng_;              // number of energy groups
  int64_t* __restrict__ n_dim_;           // number of dimensions
  int64_t* __restrict__ n_order_;         // number of orders
  int64_t* __restrict__ n_region_;        // number of mesh regions
  int64_t* __restrict__ n_source_region_; // n_region_ x ng_
  int64_t* __restrict__ n_source_region_high_order_; // n_region_ x ng_ x
                                                     // n_order_ x n_order_
  int64_t* __restrict__ materials_; // fsr materials layout information
  real_t* __restrict__ new_scalar_flux_;      // new scalar flux
  real_t* __restrict__ old_scalar_flux_;      // old scalar flux
  real_t* __restrict__ final_scalar_flux_;    // final scalar flux
  real_t* __restrict__ final_scalar_flux_sq_; // square of final flux
  real_t* __restrict__ source_;               // angular moments of the source
  real_t* __restrict__ external_source_; // external source in fixed source mode
  // English: Volumes and their accumulators are independent from flux
  // precision to preserve significant digits in long-running estimators.
  geom_t* __restrict__ volume_analytic_;   // analytic volume of each region
  geom_t* __restrict__ volume_estimated_;  // estimated volume of each region
  geom_t* __restrict__ volume_naive_;      // naive volume of each region
  geom_t* __restrict__ volume_;            // volume of each region
  char* __restrict__ has_external_source_; // Flag indicating if the region has
  char* __restrict__ whether_hit_;         // if the region has been hit

};

__device__ extern CuFlatSourceDomain* __restrict__ device_fs_domain;
extern CuFlatSourceDomain* host_fs_domain;

void cuda_load_flat_source_domain(const FlatSourceDomain& domain);

void cuda_free_flat_source_domain();

bool cuda_download_data(std::vector<double>& final_scalar_flux,
  std::vector<double>& final_scalar_flux_sq);

__device__ __forceinline__ int64_t cuda_get_region_id(
  int64_t i, int64_t j, int64_t k)
{
  int64_t stride_x = device_fs_domain->stride_x_;
  int64_t stride_y = device_fs_domain->stride_y_;
  int64_t stride_z = device_fs_domain->stride_z_;
  if (d_settings.n_dim == 1) {
    return i * device_fs_domain->stride_x_;
  }
  if (d_settings.n_dim == 2) {
    return i * device_fs_domain->stride_x_ + j * device_fs_domain->stride_y_;
  }
  if (d_settings.n_dim == 3) {
    return i * device_fs_domain->stride_x_ + j * device_fs_domain->stride_y_ +
           k * device_fs_domain->stride_z_;
  }
  return -1;
}

__global__ void cuda_update_source_kernel(iter_t keff, int run_mode);

__global__ void cuda_add_external_source_kernel(int run_mode);

void cuda_add_external_source_to_flat_source_domain();

void cuda_update_flat_source_domain(iter_t keff);

} // namespace gpu
} // namespace theta
#endif
#endif
