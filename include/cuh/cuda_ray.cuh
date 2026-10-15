//------------------------------------------------------------------------------
// File: cuda_ray.cuh
// Purpose: CUDA ray data structures and device-side ray transport helpers.
// Notes: History-based and event-based kernels share these helpers so that
//        flat-source and linear-source tally math remains consistent.
//------------------------------------------------------------------------------

#ifndef THETA_CUDA_RAY_H
#define THETA_CUDA_RAY_H
#ifdef THETA_CUDA

#include "cuda_runtime.h"
#include "cuh/cuda_base.cuh"
#include "cuh/cuda_cartesian_mesh.cuh"
#include "cuh/cuda_flat_source_domain.cuh"
#include "cuh/cuda_interface.cuh"
#include "cuh/cuda_ray.cuh"
#include "cuh/cuda_settings.cuh"
#include "exponentials.h"
#include <cstdint>
#include <vector>

namespace theta {
namespace gpu {

template<typename T>
void cuda_malloc_zero(T** device_ptr, size_t count)
{
  if (count == 0) {
    *device_ptr = nullptr;
    return;
  }
  cudaMalloc(device_ptr, sizeof(T) * count);
  cudaMemset(*device_ptr, 0, sizeof(T) * count);
}

template<typename T>
void cuda_malloc_zero(T* __restrict__* device_ptr, size_t count)
{
  if (count == 0) {
    *device_ptr = nullptr;
    return;
  }
  void* raw_ptr = nullptr;
  cudaMalloc(&raw_ptr, sizeof(T) * count);
  *device_ptr = static_cast<T*>(raw_ptr);
  cudaMemset(*device_ptr, 0, sizeof(T) * count);
}

class CuRays {
private:
  int64_t n_rays_;
  int64_t ng_;
  int64_t n_order_;
  int64_t n_seg_ {8};

public:
  int64_t* __restrict__ id;
  real_t* __restrict__ wgt;
  bool* __restrict__ is_active;
  int64_t* __restrict__ stream;
  int64_t* __restrict__ num_of_ray_order;
  int64_t* __restrict__ material;
  bool* __restrict__ seg_isactive;
  // English: Segment distances, positions, directions, and crossing targets
  // are ray-geometry state.
  geom_t* __restrict__ seg_distance;
  int8_t* __restrict__ seg_boundary;
  real_t* __restrict__ seg_rn;  // [n_rays * n_seg * n_order * n_order]
  geom_t* __restrict__ seg_pos; // [3 * n_rays * n_seg]
  geom_t* __restrict__ seg_dir; // [3 * n_rays * n_seg]
  geom_t* __restrict__ distance_travelled;
  int64_t* __restrict__ n_event;
  uint64_t* __restrict__ seeds_0;
  uint64_t* __restrict__ seeds_1;
  uint64_t* __restrict__ seeds_2;
  uint64_t* __restrict__ seeds_3;

  real_t* __restrict__ angular_flux;      // [n_rays * ng]
  real_t* __restrict__ rn;                // [n_rays * n_order * n_order]
  real_t* __restrict__ delta_psi_moments;
  geom_t* __restrict__ pos;
  geom_t* __restrict__ dir;
  int64_t* __restrict__ index;
  int64_t* __restrict__ seg_index;

  geom_t* __restrict__ next_distance;    // [n_rays]
  int64_t* __restrict__ next_cross;      // [3 * n_rays]
  int64_t* __restrict__ next_cross_i;    // [n_rays]
  geom_t* __restrict__ next_destination; // [n_rays]
  CuRays();

  void allocate_device_soa(int64_t n_rays, int64_t ng, int64_t n_order);

  void allocate_event_rays(int64_t n_rays, int64_t ng, int64_t n_order);


  void free_soa();

  HDD __forceinline__ int64_t n_rays() const { return n_rays_; }

  HDD __forceinline__ int64_t n_seg() const { return n_seg_; }

  HDD __forceinline__ int64_t vec_offset(int64_t axis, int64_t ray_id) const
  {
    return axis * n_rays_ + ray_id;
  }

  HDD __forceinline__ int64_t seg_vec_offset(
    int64_t axis, int64_t ray_id, int64_t seg_id) const
  {
    return axis * n_rays_ * n_seg_ + ray_id * n_seg_ + seg_id;
  }

  HDD __forceinline__ geom_t& pos_at(int64_t axis, int64_t ray_id)
  {
    return pos[vec_offset(axis, ray_id)];
  }

  HDD const __forceinline__ geom_t& pos_at(int64_t axis, int64_t ray_id) const
  {
    return pos[vec_offset(axis, ray_id)];
  }

  HDD __forceinline__ geom_t& dir_at(int64_t axis, int64_t ray_id)
  {
    return dir[vec_offset(axis, ray_id)];
  }

  HDD const __forceinline__ geom_t& dir_at(int64_t axis, int64_t ray_id) const
  {
    return dir[vec_offset(axis, ray_id)];
  }

  HDD __forceinline__ int64_t& index_at(int64_t axis, int64_t ray_id)
  {
    return index[vec_offset(axis, ray_id)];
  }

  HDD const __forceinline__ int64_t& index_at(
    int64_t axis, int64_t ray_id) const
  {
    return index[vec_offset(axis, ray_id)];
  }

  HDD __forceinline__ int64_t& next_cross_at(int64_t axis, int64_t ray_id)
  {
    return next_cross[vec_offset(axis, ray_id)];
  }

  HDD const __forceinline__ int64_t& next_cross_at(
    int64_t axis, int64_t ray_id) const
  {
    return next_cross[vec_offset(axis, ray_id)];
  }
};

extern CuRays* host_rays;
__device__ extern CuRays* __restrict__ device_rays;

void cuda_allocate_rays(int64_t n_rays, int64_t ng, int64_t n_order);


void cuda_free_rays();

__device__ __forceinline__ void cuda_cross_axis(size_t id_, size_t div_i,
  int64_t* index, int64_t* next_cross, geom_t* u, size_t x0, size_t x1,
  bool apply_vacuum_bc)
{
  *index += *next_cross;
  size_t bd = 1; // reflective boundary by default
  bool hit_boundary = false;
  if (*index == -1) {
    *index = 0;
    *u *= -1.0; // Reverse direction i
    bd = x0;
    hit_boundary = true;
  } else if (*index == d_divisions[div_i]->n_) {
    *index = d_divisions[div_i]->n_ - 1;
    *u *= -1.0;
    bd = x1;
    hit_boundary = true;
  }

  // Recompute spherical harmonics whenever a boundary is hit.
  if (hit_boundary) {
    const int64_t n_order = d_settings.n_order;
    const int64_t rn_offset = id_ * n_order * n_order;
    device_rays->rn[rn_offset] = static_cast<real_t>(1.0);
  }

  // For history-based tracking, vacuum boundary clears incident flux here.
  // For event-based tracking, vacuum handling is done via seg_boundary.
  if (apply_vacuum_bc && bd == 0) {
    real_t* current_angular_flux =
      &device_rays->angular_flux[id_ * d_settings.n_groups];
    for (int64_t g = 0; g < d_settings.n_groups; g++) {
      current_angular_flux[g] = 0.0;
    }
  }
}

__device__ __forceinline__ void cuda_attenuate_flux(
  size_t id_, real_t distance, bool is_active)
{
  size_t ng_ = d_settings.n_groups;
  size_t n_order = d_settings.n_order;
  int64_t i_reg = cuda_get_region_id(device_rays->index_at(0, id_),
    device_rays->index_at(1, id_), device_rays->index_at(2, id_));
  int64_t i_src =
    i_reg * d_settings.n_groups * d_settings.n_order * d_settings.n_order;

  int64_t i_mat = device_fs_domain->materials_[i_reg];

  for (int64_t g = 0; g < d_settings.n_groups; g++) {
    const real_t sigma_t =
      d_mats->sigma_t[i_mat * ng_ + g];
    const real_t optical_depth = sigma_t * distance;
    // English: Exponential attenuation follows flux precision; mixed mode
    // uses the single-precision approximation.
#if defined(THETA_CUDA_PRECISION_DOUBLE)
    const real_t attenuation_factor = double_cjosey_exponential(optical_depth);
#else
    const real_t attenuation_factor = cjosey_exponential(optical_depth);
#endif

    real_t scattering_source = 0.0;
    for (int64_t l = 0; l < n_order; l++) {
      const int64_t num_m = 2 * l + 1;
      for (int64_t m = 0; m < num_m; m++) {
        const real_t harmonic_coeff =
          device_rays
            ->rn[id_ * n_order * n_order + l * l + m];

        const int64_t src_index = i_src + l * l * ng_ + m * ng_ + g;
        scattering_source +=
          harmonic_coeff * device_fs_domain->source_[src_index];
      }
    }

    // ψ_new = (ψ_old - Q_s) * (1 - exp(-Σ_t * s))
    // Δψ = ψ_old - ψ_new
    // ψ_new = ψ_old - Δψ
    const real_t delta_flux =
      (device_rays->angular_flux[id_ * ng_ + g] - scattering_source) *
      attenuation_factor;

    device_rays->angular_flux[id_ * ng_ + g] -= delta_flux;

    if (is_active) {
      for (int64_t l = 0; l < n_order; l++) {
        for (int64_t m = 0; m < 2 * l + 1; m++) {
          const real_t harmonic_coeff =
            device_rays->rn[id_ * n_order * n_order + l * l + m];
          const int64_t src_index = i_src + l * l * ng_ + m * ng_ + g;
#ifdef __CUDA_ARCH__
          atomicAdd(&device_fs_domain->new_scalar_flux_[src_index],
            delta_flux * harmonic_coeff);
#endif
        }
      }
    }
  }

  if (is_active) {
#ifdef __CUDA_ARCH__
    atomicAdd(&device_fs_domain->n_tracks_[i_reg], 1);
    if (d_settings.volume_type != static_cast<size_t>(VolumeType::Analytic)) {
      atomicAdd(
        &device_fs_domain->volume_[i_reg], static_cast<geom_t>(distance));
    }
#endif
  }
}

__device__ void cuda_advance_ray(size_t id_);

__device__ void cuda_cross_surface(size_t id_);

} // namespace gpu
} // namespace theta
#endif
#endif
