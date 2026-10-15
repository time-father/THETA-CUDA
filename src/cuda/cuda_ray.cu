#ifdef THETA_CUDA
#include "cuh/cuda_base.cuh"
#include "cuh/cuda_cartesian_mesh.cuh"
#include "cuh/cuda_flat_source_domain.cuh"
#include "cuh/cuda_interface.cuh"
#include "cuh/cuda_ray.cuh"
#include "cuh/cuda_settings.cuh"
#include "exponentials.h"

namespace theta {
namespace gpu {

CuRays* host_rays = nullptr;
__device__ CuRays* __restrict__ device_rays = nullptr;

void init_cu_rays_pointers(CuRays* rays)
{
  rays->id = nullptr;
  rays->wgt = nullptr;
  rays->is_active = nullptr;
  rays->stream = nullptr;
  rays->num_of_ray_order = nullptr;
  rays->material = nullptr;
  rays->seg_isactive = nullptr;
  rays->seg_distance = nullptr;
  rays->seg_rn = nullptr;
  rays->seg_pos = nullptr;
  rays->seg_dir = nullptr;
  rays->distance_travelled = nullptr;
  rays->n_event = nullptr;
  rays->seg_boundary = nullptr;
  rays->seeds_0 = nullptr;
  rays->seeds_1 = nullptr;
  rays->seeds_2 = nullptr;
  rays->seeds_3 = nullptr;
  rays->angular_flux = nullptr;
  rays->rn = nullptr;
  rays->delta_psi_moments = nullptr;
  rays->pos = nullptr;
  rays->dir = nullptr;
  rays->index = nullptr;
  rays->seg_index = nullptr;
  rays->next_distance = nullptr;
  rays->next_cross = nullptr;
  rays->next_cross_i = nullptr;
  rays->next_destination = nullptr;
}

void CuRays::allocate_device_soa(int64_t n_rays, int64_t ng, int64_t n_order)
{
  n_rays_ = n_rays;
  ng_ = ng;
  n_order_ = n_order;
  n_seg_ = settings::event_seg_count;

  cuda_malloc_zero(&wgt, n_rays_);
  cuda_malloc_zero(&is_active, n_rays_);
  cuda_malloc_zero(&material, n_rays_);
  cuda_malloc_zero(&seg_isactive, n_rays_ * n_seg_);
  cuda_malloc_zero(&seg_distance, n_rays_ * n_seg_);
  cuda_malloc_zero(&seg_boundary, n_rays_ * n_seg_);
  cuda_malloc_zero(&seg_rn, n_rays_ * n_seg_ * n_order_ * n_order_);
  cuda_malloc_zero(&seg_pos, 3 * n_rays_ * n_seg_);
  cuda_malloc_zero(&seg_dir, 3 * n_rays_ * n_seg_);
  cuda_malloc_zero(&seg_index, n_rays_ * n_seg_);
  cuda_malloc_zero(&distance_travelled, n_rays_);
  cuda_malloc_zero(&seeds_0, n_rays_);
  cuda_malloc_zero(&seeds_1, n_rays_);
  cuda_malloc_zero(&seeds_2, n_rays_);
  cuda_malloc_zero(&seeds_3, n_rays_);
  cuda_malloc_zero(&angular_flux, n_rays_ * ng_);
  cuda_malloc_zero(&rn, n_rays_ * n_order_ * n_order_);
  cuda_malloc_zero(&pos, 3 * n_rays_);
  cuda_malloc_zero(&dir, 3 * n_rays_);
  cuda_malloc_zero(&index, 3 * n_rays_);
  cuda_malloc_zero(&next_distance, n_rays_);
  cuda_malloc_zero(&next_cross, 3 * n_rays_);
  cuda_malloc_zero(&next_cross_i, n_rays_);
  cuda_malloc_zero(&next_destination, n_rays_);
}

void CuRays::allocate_event_rays(int64_t n_rays, int64_t ng, int64_t n_order)
{
  n_rays_ = n_rays;
  ng_ = ng;
  n_order_ = n_order;
  n_seg_ = settings::event_seg_count;

  cuda_malloc_zero(&wgt, n_rays_);
  cuda_malloc_zero(&is_active, n_rays_ * n_seg_);
  cuda_malloc_zero(&material, n_rays_ * n_seg_);
  cuda_malloc_zero(&seg_isactive, n_rays_ * n_seg_);
  cuda_malloc_zero(&seg_distance, n_rays_ * n_seg_);
  cuda_malloc_zero(&seg_boundary, n_rays_ * n_seg_);
  cuda_malloc_zero(&seg_rn, n_rays_ * n_seg_ * n_order_ * n_order_);
  cuda_malloc_zero(&seg_pos, 3 * n_rays_ * n_seg_);
  cuda_malloc_zero(&seg_dir, 3 * n_rays_ * n_seg_);
  cuda_malloc_zero(&distance_travelled, n_rays_);
  cuda_malloc_zero(&seeds_0, n_rays_);
  cuda_malloc_zero(&seeds_1, n_rays_);
  cuda_malloc_zero(&seeds_2, n_rays_);
  cuda_malloc_zero(&seeds_3, n_rays_);
  cuda_malloc_zero(&angular_flux, n_rays_ * ng_);
  cuda_malloc_zero(&rn, n_rays_ * n_order_ * n_order_);
  cuda_malloc_zero(&pos, 3 * n_rays_);
  cuda_malloc_zero(&dir, 3 * n_rays_);
  cuda_malloc_zero(&index, 3 * n_rays_);
  cuda_malloc_zero(&seg_index, n_rays_ * n_seg_);
  cuda_malloc_zero(&next_distance, n_rays_);
  cuda_malloc_zero(&next_cross, 3 * n_rays_);
  cuda_malloc_zero(&next_cross_i, n_rays_);
  cuda_malloc_zero(&next_destination, n_rays_);
}

void CuRays::free_soa()
{
  if (id != nullptr)
    cudaFree(id);
  if (wgt != nullptr)
    cudaFree(wgt);
  if (is_active != nullptr)
    cudaFree(is_active);
  if (stream != nullptr)
    cudaFree(stream);
  if (num_of_ray_order != nullptr)
    cudaFree(num_of_ray_order);
  if (material != nullptr)
    cudaFree(material);
  if (seg_isactive != nullptr)
    cudaFree(seg_isactive);
  if (seg_distance != nullptr)
    cudaFree(seg_distance);
  if (seg_boundary != nullptr)
    cudaFree(seg_boundary);
  if (seg_rn != nullptr)
    cudaFree(seg_rn);
  if (seg_pos != nullptr)
    cudaFree(seg_pos);
  if (seg_dir != nullptr)
    cudaFree(seg_dir);
  if (distance_travelled != nullptr)
    cudaFree(distance_travelled);
  if (n_event != nullptr)
    cudaFree(n_event);

  if (seeds_0 != nullptr)
    cudaFree(seeds_0);
  if (seeds_1 != nullptr)
    cudaFree(seeds_1);
  if (seeds_2 != nullptr)
    cudaFree(seeds_2);
  if (seeds_3 != nullptr)
    cudaFree(seeds_3);

  if (angular_flux != nullptr)
    cudaFree(angular_flux);
  if (rn != nullptr)
    cudaFree(rn);
  if (delta_psi_moments != nullptr)
    cudaFree(delta_psi_moments);

  if (pos != nullptr)
    cudaFree(pos);
  if (dir != nullptr)
    cudaFree(dir);
  if (index != nullptr)
    cudaFree(index);
  if (seg_index != nullptr)
    cudaFree(seg_index);
  if (next_distance != nullptr)
    cudaFree(next_distance);
  if (next_cross != nullptr)
    cudaFree(next_cross);
  if (next_cross_i != nullptr)
    cudaFree(next_cross_i);
  if (next_destination != nullptr)
    cudaFree(next_destination);
}

CuRays::CuRays()
{
  init_cu_rays_pointers(this);
}

void cuda_free_rays()
{
  if (host_rays != nullptr) {
    host_rays->free_soa();

    delete host_rays;
    host_rays = nullptr;
  }
}

void cuda_allocate_rays(int64_t n_rays, int64_t ng, int64_t n_order)
{
  if (host_rays != nullptr) {
    cuda_free_rays();
  }
  host_rays = new CuRays();
  host_rays->allocate_device_soa(n_rays, ng, n_order);

  CuRays* d_cuda_rays;
  cudaMalloc((void**)&d_cuda_rays, sizeof(CuRays));
  cudaMemcpy(d_cuda_rays, host_rays, sizeof(CuRays), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(device_rays, &d_cuda_rays, sizeof(CuRays*));
}

__device__ void cuda_distance_to_boundary(size_t id_)
{
  size_t n_dim = d_settings.n_dim;
  real_t d = INFTY;
  int64_t cross_x = 1;
  int64_t cross_y = 0;
  int64_t cross_z = 0;
  int64_t cross_dim = 0;
  real_t destination = 0.0;

  if (fabs(device_rays->dir_at(0, id_)) > 1e-12) {
    int64_t j = (device_rays->dir_at(0, id_) > 0.0) ? 1 : 0;
    real_t bx = d_divisions[0]->edges_[device_rays->index_at(0, id_) + j];
    real_t pos_x = device_rays->pos_at(0, id_);
    real_t dir_x = device_rays->dir_at(0, id_);
    real_t d_i = (bx - pos_x) / dir_x;
    if (d_i < d) {
      d = d_i;
      // copysign equivalent in CUDA
      cross_x = (device_rays->dir_at(0, id_) > 0.0) ? 1 : -1;
      // Specify a position to cross
      cross_dim = 0;
      destination = d_divisions[0]->edges_[device_rays->index_at(0, id_) + j];
    }
  }
  if (n_dim >= 2) {
    if (fabs(device_rays->dir_at(1, id_)) > 1e-12) {
      int64_t j = (device_rays->dir_at(1, id_) > 0.0) ? 1 : 0;
      real_t by = d_divisions[1]->edges_[device_rays->index_at(1, id_) + j];
      real_t pos_y = device_rays->pos_at(1, id_);
      real_t dir_y = device_rays->dir_at(1, id_);
      real_t d_i = (by - pos_y) / dir_y;
      if (d_i < d) {
        d = d_i;
        // copysign equivalent in CUDA
        cross_y = (device_rays->dir_at(1, id_) > 0.0) ? 1 : -1;
        // Specify a position to cross
        cross_dim = 1;
        destination = d_divisions[1]->edges_[device_rays->index_at(1, id_) + j];
      }
    }
  }
  if (n_dim >= 3) {
    if (fabs(device_rays->dir_at(2, id_)) > 1e-12) {
      int64_t j = (device_rays->dir_at(2, id_) > 0.0) ? 1 : 0;
      real_t dz = d_divisions[2]->edges_[device_rays->index_at(2, id_) + j];
      real_t pos_z = device_rays->pos_at(2, id_);
      real_t dir_z = device_rays->dir_at(2, id_);
      real_t d_i = (dz - pos_z) / dir_z;
      if (d_i < d) {
        d = d_i;
        // copysign equivalent in CUDA
        cross_z = (device_rays->dir_at(2, id_) > 0.0) ? 1 : -1;
        // Specify a position to cross
        cross_dim = 2;
        destination = d_divisions[2]->edges_[device_rays->index_at(2, id_) + j];
      }
    }
  }
  real_t dis = device_rays->distance_travelled[id_];
  device_rays->next_distance[id_] = d;
  device_rays->next_cross_at(0, id_) = cross_x;
  device_rays->next_cross_at(1, id_) = cross_y;
  device_rays->next_cross_at(2, id_) = cross_z;
  device_rays->next_cross_i[id_] = cross_dim;
  device_rays->next_destination[id_] = destination;
}

__device__ void cuda_advance_ray(size_t id_)
{
  // update ray's distance to mesh boundaries
  cuda_distance_to_boundary(id_);
  real_t distance = device_rays->next_distance[id_];
  if (device_rays->is_active[id_]) {
    if (distance + device_rays->distance_travelled[id_] >=
        d_settings.active_length) {
      distance =
        d_settings.active_length - device_rays->distance_travelled[id_];
      device_rays->wgt[id_] = 0.0;
    }
    device_rays->distance_travelled[id_] += distance;
    // Update angular flux and scalar flux

      cuda_attenuate_flux(id_, distance, true);

  } else {
    if (device_rays->distance_travelled[id_] + distance >=
        d_settings.inactive_length) {
      // From an inactive ray to an active ray
      device_rays->is_active[id_] = true;

      // Treat the inactive phase
      real_t distance_dead =
        d_settings.inactive_length - device_rays->distance_travelled[id_];

        cuda_attenuate_flux(id_, distance_dead, false);


      // Treat the active phase
      real_t distance_alive = distance - distance_dead;

      // Ensure we haven't travelled past the active phase as well
      if (distance_alive > d_settings.active_length) {
        distance_alive = d_settings.active_length;
        // Calculate the actual distance of ray movement
        distance = distance_alive + distance_dead;
        device_rays->wgt[id_] = 0.0;
      }


        cuda_attenuate_flux(id_, distance_alive, true);


      // Reset distance_travelled for active phase
      device_rays->distance_travelled[id_] = distance_alive;
    } else {
      // Travel in the inactive phase
      device_rays->distance_travelled[id_] += distance;

        cuda_attenuate_flux(id_, distance, false);

    }
  }
  // Update distance to boundary
  device_rays->pos_at(0, id_) += device_rays->dir_at(0, id_) * distance;
  device_rays->pos_at(1, id_) += device_rays->dir_at(1, id_) * distance;
  device_rays->pos_at(2, id_) += device_rays->dir_at(2, id_) * distance;
}

__device__ void cuda_cross_surface(size_t id_)
{
  // Set next mesh index
  int64_t i = device_rays->next_cross_i[id_];
  if (i == 0) {
    cuda_cross_axis(id_, 0, &device_rays->index_at(0, id_),
      &device_rays->next_cross_at(0, id_), &device_rays->dir_at(0, id_),
      d_settings.x0_boundary, d_settings.x1_boundary, true);
  }
  if (i == 1) {
    cuda_cross_axis(id_, 1, &device_rays->index_at(1, id_),
      &device_rays->next_cross_at(1, id_), &device_rays->dir_at(1, id_),
      d_settings.y0_boundary, d_settings.y1_boundary, true);
  }
  if (i == 2) {
    cuda_cross_axis(id_, 2, &device_rays->index_at(2, id_),
      &device_rays->next_cross_at(2, id_), &device_rays->dir_at(2, id_),
      d_settings.z0_boundary, d_settings.z1_boundary, true);
  }
  int64_t i_reg = cuda_get_region_id(device_rays->index_at(0, id_),
    device_rays->index_at(1, id_), device_rays->index_at(2, id_));

  device_rays->material[id_] = device_fs_domain->materials_[i_reg];
}
} // namespace gpu
} // namespace theta
#endif
