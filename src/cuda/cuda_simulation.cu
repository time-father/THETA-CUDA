//------------------------------------------------------------------------------
// File: cuda_simulation.cu
// Purpose: CUDA random-ray simulation driver and transport kernels.
// Notes: The file owns batch initialization, source update, transport,
//        normalization, final flux reduction, and GPU timing orchestration.
//------------------------------------------------------------------------------

#include "cuh/cuda_random.cuh"
#include "cuh/cuda_simulation.cuh"
#include "run_info.h"
#include "timer.h"

#ifdef THETA_CUDA
namespace theta {
namespace gpu {

CuSimulation host_simulation;
__device__ CuSimulation cu_simulation;

namespace {

// Store the start and stop events for one event-transport kernel invocation.
// Recording is asynchronous; elapsed times are queried after the batch ends.
struct CudaKernelEventPair {
  cudaEvent_t start {nullptr};
  cudaEvent_t stop {nullptr};
};

// Collect and destroy CUDA events after the associated stream work completes.
double collect_cuda_kernel_time(std::vector<CudaKernelEventPair>& event_pairs)
{
  double elapsed_seconds = 0.0;
  for (const CudaKernelEventPair& event_pair : event_pairs) {
    float elapsed_milliseconds = 0.0f;
    cugo(cudaEventElapsedTime(
      &elapsed_milliseconds, event_pair.start, event_pair.stop));
    elapsed_seconds += static_cast<double>(elapsed_milliseconds) / 1000.0;
    cugo(cudaEventDestroy(event_pair.start));
    cugo(cudaEventDestroy(event_pair.stop));
  }
  event_pairs.clear();
  return elapsed_seconds;
}


} // namespace

void cuda_load_simulation()
{
  host_simulation.fiss_rate_new = 0;
  host_simulation.fiss_rate_old = 0;
  host_simulation.keff_cycle = 1;
  host_simulation.mean_keff = 0;
  host_simulation.std_dev_keff = 0;
  host_simulation.total_n_tracks = 0;
  cudaError_t err =
    cudaMemcpyToSymbol(cu_simulation, &host_simulation, sizeof(CuSimulation));
  if (err != cudaSuccess) {
    fprintf(stderr, "cudaMemcpyToSymbol failed: %s\n", cudaGetErrorString(err));
    return;
  }
  cudaDeviceSynchronize();
}

void cuda_load_data()
{
  cuda_load_settings();
  cuda_load_simulation();
  cuda_load_material(simulation::materials);
  cuda_load_flat_source_domain(*simulation::model);

  std::vector<const Division*> divisions(simulation::model->n_dim());
  for (int i = 0; i < simulation::model->n_dim(); i++) {
    divisions[i] = simulation::model->geometry(i);
  }
  cuda_load_divisions(divisions);
  cuda_allocate_rays(
    settings::n_track, simulation::model->ng(), simulation::model->n_order());
}

__global__ void cuda_init_batch()
{
  int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t stride = gridDim.x * blockDim.x;
  real_t* __restrict__ fs_scalar_flux = device_fs_domain->new_scalar_flux_;
  for (int64_t idx = tid; idx < d_settings.n_source_region_high_order;
       idx += stride) {
    fs_scalar_flux[idx] = 0;
  }



  for (int64_t i_reg = tid; i_reg < d_settings.n_region; i_reg += stride) {
    device_fs_domain->n_tracks_[i_reg] = 0;
    if (d_settings.volume_type != static_cast<size_t>(VolumeType::Analytic)) {
      device_fs_domain->volume_[i_reg] = 0;
    }
  }
}

__device__ void cuda_init_ray(size_t i_ray, int64_t ray_seed)
{
  uint64_t source_ray_seeds[N_STREAMS];
  int64_t i_stream = STREAM_TRACKING;
  cu_init_particle_seeds(ray_seed, source_ray_seeds);

  geom_t edge_0 = d_divisions[0]->edges_[0];
  geom_t edge_1 = d_divisions[0]->edges_[d_divisions[0]->n_];
  geom_t* edge_0p = &d_divisions[0]->edges_[0];
  geom_t* edge_1p = &d_divisions[0]->edges_[d_divisions[0]->n_];
  if (d_settings.n_dim >= 1) {
    device_rays->pos_at(0, i_ray) =
      edge_0 + cu_prn(&source_ray_seeds[i_stream]) * (edge_1 - edge_0);
    device_rays->index_at(0, i_ray) =
      binary_search(edge_0p, edge_1p, device_rays->pos_at(0, i_ray));
  }
  if (d_settings.n_dim >= 2) {
    edge_0 = d_divisions[1]->edges_[0];
    edge_1 = d_divisions[1]->edges_[d_divisions[1]->n_];
    edge_0p = &d_divisions[1]->edges_[0];
    edge_1p = &d_divisions[1]->edges_[d_divisions[1]->n_];
    device_rays->pos_at(1, i_ray) =
      edge_0 + cu_prn(&source_ray_seeds[i_stream]) * (edge_1 - edge_0);
    device_rays->index_at(1, i_ray) =
      binary_search(edge_0p, edge_1p, device_rays->pos_at(1, i_ray));
  }
  if (d_settings.n_dim >= 3) {
    edge_0 = d_divisions[2]->edges_[0];
    edge_1 = d_divisions[2]->edges_[d_divisions[2]->n_];
    edge_0p = &d_divisions[2]->edges_[0];
    edge_1p = &d_divisions[2]->edges_[d_divisions[2]->n_];
    device_rays->pos_at(2, i_ray) =
      edge_0 + cu_prn(&source_ray_seeds[i_stream]) * (edge_1 - edge_0);
    device_rays->index_at(2, i_ray) =
      binary_search(edge_0p, edge_1p, device_rays->pos_at(2, i_ray));
  }

  const geom_t mu =
    static_cast<geom_t>(cu_prn(&source_ray_seeds[i_stream])) * 2.0 - 1.0;
  const geom_t phi =
    static_cast<geom_t>(cu_prn(&source_ray_seeds[i_stream])) * 2.0 * PI;
  device_rays->dir_at(0, i_ray) = mu;
  device_rays->dir_at(1, i_ray) = sqrt(1.0 - mu * mu) * cos(phi);
  device_rays->dir_at(2, i_ray) = sqrt(1.0 - mu * mu) * sin(phi);

  size_t rn_offset = i_ray * d_settings.n_order * d_settings.n_order;
  device_rays->rn[rn_offset] = static_cast<real_t>(1.0);

  uint64_t* single_ray_seeds[N_STREAMS] = {&device_rays->seeds_0[i_ray],
    &device_rays->seeds_1[i_ray], &device_rays->seeds_2[i_ray],
    &device_rays->seeds_3[i_ray]};
  for (int stream = 0; stream < N_STREAMS; ++stream) {
    *single_ray_seeds[stream] = cu_init_seed(ray_seed, stream);
  }

  int64_t i_reg = cuda_get_region_id(device_rays->index_at(0, i_ray),
    device_rays->index_at(1, i_ray), device_rays->index_at(2, i_ray));
  device_rays->material[i_ray] = device_fs_domain->materials_[i_reg];
  device_rays->is_active[i_ray] = false;
  device_rays->distance_travelled[i_ray] = 0.0;
  device_rays->wgt[i_ray] = 1.0;

  size_t src_offset =
    i_reg * d_settings.n_groups * d_settings.n_order * d_settings.n_order;
  for (int64_t g = 0; g < d_settings.n_groups; g++) {
    device_rays->angular_flux[i_ray * d_settings.n_groups + g] =
      device_fs_domain->source_[src_offset + g];
  }
}

__global__ void cuda_update_rays(size_t i_batch)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t stride = gridDim.x * blockDim.x;
  for (int i_ray = tid; i_ray < d_settings.n_track; i_ray += stride) {
    int64_t ray_seed =
      static_cast<int64_t>(i_batch * d_settings.n_track + i_ray + 1);
    cuda_init_ray(i_ray, ray_seed);
  }
}

__global__ void cuda_update_rays_immortal(size_t i_batch, real_t resample_rate)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t stride = gridDim.x * blockDim.x;
  CuRays* __restrict__ rays = device_rays;
  for (int i_ray = tid; i_ray < d_settings.n_track; i_ray += stride) {
    uint64_t seed = static_cast<uint64_t>(i_batch + i_ray);
    real_t u = cu_prn(&seed);
    if (u <= resample_rate) {
      int64_t ray_seed =
        static_cast<int64_t>(i_batch * d_settings.n_track + i_ray + 1);
      cuda_init_ray(i_ray, ray_seed);
    } else {
      rays->wgt[i_ray] = 1.0;
      rays->is_active[i_ray] = true;
      rays->distance_travelled[i_ray] = 0.0;
    }
  }
}

__global__ void cuda_track_history(unsigned long long* segment_count)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned stride = gridDim.x * blockDim.x;
  CuRays* __restrict__ rays = device_rays;
  CuFlatSourceDomain* __restrict__ fs = device_fs_domain;
  cudaDivision** __restrict__ divisions = d_divisions;
  real_t* __restrict__ wgt = rays->wgt;
  for (int id_ = tid; id_ < d_settings.n_track; id_ += stride) {
    real_t local_wgt = wgt[id_];
    unsigned long long segments_in_ray = 0;
    while (local_wgt > 0) {
      // update ray's distance to mesh boundaries
      size_t n_dim = d_settings.n_dim;
      geom_t d = static_cast<geom_t>(INFTY);
      int64_t cross_x = 1;
      int64_t cross_y = 0;
      int64_t cross_z = 0;
      int64_t cross_dim = 0;
      geom_t destination = 0.0;

      if (fabs(rays->dir_at(0, id_)) > 1e-12) {
        int64_t j = (rays->dir_at(0, id_) > 0.0) ? 1 : 0;
        const geom_t bx = divisions[0]->edges_[rays->index_at(0, id_) + j];
        const geom_t pos_x = rays->pos_at(0, id_);
        const geom_t dir_x = rays->dir_at(0, id_);
        const geom_t d_i = (bx - pos_x) / dir_x;
        if (d_i < d) {
          d = d_i;
          // copysign equivalent in CUDA
          cross_x = (rays->dir_at(0, id_) > 0.0) ? 1 : -1;
          // Specify a position to cross
          cross_dim = 0;
          destination = divisions[0]->edges_[rays->index_at(0, id_) + j];
        }
      }
      if (n_dim >= 2) {
        if (fabs(rays->dir_at(1, id_)) > 1e-12) {
          int64_t j = (rays->dir_at(1, id_) > 0.0) ? 1 : 0;
          const geom_t by = divisions[1]->edges_[rays->index_at(1, id_) + j];
          const geom_t pos_y = rays->pos_at(1, id_);
          const geom_t dir_y = rays->dir_at(1, id_);
          const geom_t d_i = (by - pos_y) / dir_y;
          if (d_i < d) {
            d = d_i;
            // copysign equivalent in CUDA
            cross_y = (rays->dir_at(1, id_) > 0.0) ? 1 : -1;
            // Specify a position to cross
            cross_dim = 1;
            destination = divisions[1]->edges_[rays->index_at(1, id_) + j];
          }
        }
      }
      if (n_dim >= 3) {
        if (fabs(rays->dir_at(2, id_)) > 1e-12) {
          int64_t j = (rays->dir_at(2, id_) > 0.0) ? 1 : 0;
          const geom_t dz = divisions[2]->edges_[rays->index_at(2, id_) + j];
          const geom_t pos_z = rays->pos_at(2, id_);
          const geom_t dir_z = rays->dir_at(2, id_);
          const geom_t d_i = (dz - pos_z) / dir_z;
          if (d_i < d) {
            d = d_i;
            // copysign equivalent in CUDA
            cross_z = (rays->dir_at(2, id_) > 0.0) ? 1 : -1;
            // Specify a position to cross
            cross_dim = 2;
            destination = divisions[2]->edges_[rays->index_at(2, id_) + j];
          }
        }
      }
      geom_t dis = rays->distance_travelled[id_];
      rays->next_distance[id_] = d;
      rays->next_cross_at(0, id_) = cross_x;
      rays->next_cross_at(1, id_) = cross_y;
      rays->next_cross_at(2, id_) = cross_z;
      rays->next_cross_i[id_] = cross_dim;
      rays->next_destination[id_] = destination;

      geom_t distance = rays->next_distance[id_];
      if (distance <= static_cast<geom_t>(0.0)) {
        rays->wgt[id_] = 0.0;
        break;
      }

      if (rays->is_active[id_]) {
        if (distance + rays->distance_travelled[id_] >=
            d_settings.active_length) {
          distance = d_settings.active_length - rays->distance_travelled[id_];
          rays->wgt[id_] = 0.0;
        }
        rays->distance_travelled[id_] += distance;
        if (distance > static_cast<geom_t>(0.0)) {
          ++segments_in_ray;
        }
        // Update angular flux and scalar flux

          cuda_attenuate_flux(id_, distance, true);

      } else {
        if (rays->distance_travelled[id_] + distance >=
            d_settings.inactive_length) {
          // From an inactive ray to an active ray
          rays->is_active[id_] = true;

          // Treat the inactive phase
          geom_t distance_dead =
            d_settings.inactive_length - rays->distance_travelled[id_];

            cuda_attenuate_flux(id_, distance_dead, false);

          if (distance_dead > static_cast<geom_t>(0.0)) {
            ++segments_in_ray;
          }

          // Treat the active phase
          geom_t distance_alive = distance - distance_dead;

          // Ensure we haven't travelled past the active phase as well
          if (distance_alive > d_settings.active_length) {
            distance_alive = d_settings.active_length;
            // Calculate the actual distance of ray movement
            distance = distance_alive + distance_dead;
            rays->wgt[id_] = 0.0;
          }


            cuda_attenuate_flux(id_, distance_alive, true);

          if (distance_alive > static_cast<geom_t>(0.0)) {
            ++segments_in_ray;
          }

          // Reset distance_travelled for active phase
          rays->distance_travelled[id_] = distance_alive;
        } else {
          // Travel in the inactive phase
          rays->distance_travelled[id_] += distance;

            cuda_attenuate_flux(id_, distance, false);

          if (distance > static_cast<geom_t>(0.0)) {
            ++segments_in_ray;
          }
        }
      }
      // Update distance to boundary
      rays->pos_at(0, id_) += rays->dir_at(0, id_) * distance;
      rays->pos_at(1, id_) += rays->dir_at(1, id_) * distance;
      rays->pos_at(2, id_) += rays->dir_at(2, id_) * distance;
      // if (device_rays->wgt[id_] > 0) {
      local_wgt = wgt[id_];
      if (local_wgt <= 0) {
        break;
      }
      // Set next mesh index
      int64_t i = rays->next_cross_i[id_];
      rays->pos_at(i, id_) = destination;
      if (i == 0) {
        cuda_cross_axis(id_, 0, &rays->index_at(0, id_),
          &rays->next_cross_at(0, id_), &rays->dir_at(0, id_),
          d_settings.x0_boundary, d_settings.x1_boundary, true);
      }
      if (i == 1) {
        cuda_cross_axis(id_, 1, &rays->index_at(1, id_),
          &rays->next_cross_at(1, id_), &rays->dir_at(1, id_),
          d_settings.y0_boundary, d_settings.y1_boundary, true);
      }
      if (i == 2) {
        cuda_cross_axis(id_, 2, &rays->index_at(2, id_),
          &rays->next_cross_at(2, id_), &rays->dir_at(2, id_),
          d_settings.z0_boundary, d_settings.z1_boundary, true);
      }
      int64_t i_reg = cuda_get_region_id(
        rays->index_at(0, id_), rays->index_at(1, id_), rays->index_at(2, id_));

      rays->material[id_] = fs->materials_[i_reg];
    }
    if (segments_in_ray > 0) {
      atomicAdd(segment_count, segments_in_ray);
    }
  }
}

__global__ void cuda_track_events(unsigned long long* segment_count)
{
  CuRays* __restrict__ rays = device_rays;
  real_t* __restrict__ wgt = rays->wgt;
  geom_t* __restrict__ pos = rays->pos;
  geom_t* __restrict__ dir = rays->dir;
  // Track segments
  bool* __restrict__ seg_isactive = rays->seg_isactive;
  int64_t* __restrict__ seg_index = rays->seg_index;
  geom_t* __restrict__ seg_dis = rays->seg_distance;
  int8_t* __restrict__ seg_boundary = rays->seg_boundary;
  real_t* __restrict__ seg_rn = rays->seg_rn;
  geom_t* __restrict__ seg_pos = rays->seg_pos;
  geom_t* __restrict__ seg_dir = rays->seg_dir;
  // Ray info
  geom_t* __restrict__ distance_travelled = rays->distance_travelled;
  int64_t* __restrict__ index = rays->index;

  uint8_t ndim = d_settings.n_dim;
  const int64_t n_llm = 1;
  const int64_t n_seg = d_settings.event_seg_count;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t stride = gridDim.x * blockDim.x;

  for (int ray_id = tid; ray_id < d_settings.n_track; ray_id += stride) {
    if (wgt[ray_id] <= 0.0) {
      for (int64_t seg_id = 0; seg_id < n_seg; seg_id++) {
        int64_t global_seg_id = ray_id * n_seg + seg_id;
        seg_dis[global_seg_id] = 0.0;
        seg_isactive[global_seg_id] = false;
        seg_boundary[global_seg_id] = -1;
        seg_index[global_seg_id] = -1;
        seg_pos[rays->seg_vec_offset(0, ray_id, seg_id)] = 0.0;
        seg_pos[rays->seg_vec_offset(1, ray_id, seg_id)] = 0.0;
        seg_pos[rays->seg_vec_offset(2, ray_id, seg_id)] = 0.0;
        seg_dir[rays->seg_vec_offset(0, ray_id, seg_id)] = 0.0;
        seg_dir[rays->seg_vec_offset(1, ray_id, seg_id)] = 0.0;
        seg_dir[rays->seg_vec_offset(2, ray_id, seg_id)] = 0.0;
        for (int64_t llm = 0; llm < n_llm; ++llm) {
          seg_rn[global_seg_id * n_llm + llm] = 0.0;
        }
      }
      continue;
    }

    geom_t pos_x = pos[rays->vec_offset(0, ray_id)];
    geom_t pos_y = pos[rays->vec_offset(1, ray_id)];
    geom_t pos_z = pos[rays->vec_offset(2, ray_id)];
    geom_t u_x = dir[rays->vec_offset(0, ray_id)];
    geom_t u_y = dir[rays->vec_offset(1, ray_id)];
    geom_t u_z = dir[rays->vec_offset(2, ray_id)];
    int64_t idx_x = index[rays->vec_offset(0, ray_id)];
    int64_t idx_y = index[rays->vec_offset(1, ray_id)];
    int64_t idx_z = index[rays->vec_offset(2, ray_id)];
    geom_t ray_distance = distance_travelled[ray_id];
    bool ray_active = rays->is_active[ray_id];
    unsigned long long segments_in_ray = 0;

    for (int64_t seg_id = 0; seg_id < n_seg;) {
      int64_t global_seg_id = ray_id * n_seg + seg_id;
      if (wgt[ray_id] <= 0.0) {
        seg_dis[global_seg_id] = 0.0;
        seg_isactive[global_seg_id] = false;
        seg_boundary[global_seg_id] = -1;
        seg_index[global_seg_id] = -1;
        seg_pos[rays->seg_vec_offset(0, ray_id, seg_id)] = 0.0;
        seg_pos[rays->seg_vec_offset(1, ray_id, seg_id)] = 0.0;
        seg_pos[rays->seg_vec_offset(2, ray_id, seg_id)] = 0.0;
        seg_dir[rays->seg_vec_offset(0, ray_id, seg_id)] = 0.0;
        seg_dir[rays->seg_vec_offset(1, ray_id, seg_id)] = 0.0;
        seg_dir[rays->seg_vec_offset(2, ray_id, seg_id)] = 0.0;
        for (int64_t llm = 0; llm < n_llm; ++llm) {
          seg_rn[global_seg_id * n_llm + llm] = 0.0;
        }
        seg_id++;
        continue;
      }

      geom_t distance_to_surf = static_cast<geom_t>(INFTY);
      int64_t cross_x = 0;
      int64_t cross_y = 0;
      int64_t cross_z = 0;
      int64_t cross_dim = 0;
      geom_t cross_destination = 0.0;

      if (ndim >= 1 && fabs(u_x) > 1e-12) {
        int64_t j = (u_x > 0.0) ? 1 : 0;
        const geom_t bx = d_divisions[0]->edges_[idx_x + j];
        const geom_t d_i = (bx - pos_x) / u_x;
        if (d_i < distance_to_surf) {
          distance_to_surf = d_i;
          cross_x = (u_x > 0.0) ? 1 : -1;
          cross_dim = 0;
          cross_destination = bx;
        }
      }
      if (ndim >= 2 && fabs(u_y) > 1e-12) {
        int64_t j = (u_y > 0.0) ? 1 : 0;
        const geom_t by = d_divisions[1]->edges_[idx_y + j];
        const geom_t d_i = (by - pos_y) / u_y;
        if (d_i < distance_to_surf) {
          distance_to_surf = d_i;
          cross_y = (u_y > 0.0) ? 1 : -1;
          cross_dim = 1;
          cross_destination = by;
        }
      }
      if (ndim >= 3 && fabs(u_z) > 1e-12) {
        int64_t j = (u_z > 0.0) ? 1 : 0;
        const geom_t bz = d_divisions[2]->edges_[idx_z + j];
        const geom_t d_i = (bz - pos_z) / u_z;
        if (d_i < distance_to_surf) {
          distance_to_surf = d_i;
          cross_z = (u_z > 0.0) ? 1 : -1;
          cross_dim = 2;
          cross_destination = bz;
        }
      }

      if (distance_to_surf <= 0.0) {
        wgt[ray_id] = 0.0;
        seg_dis[global_seg_id] = 0.0;
        seg_isactive[global_seg_id] = false;
        seg_boundary[global_seg_id] = -1;
        seg_index[global_seg_id] = -1;
        seg_pos[rays->seg_vec_offset(0, ray_id, seg_id)] = 0.0;
        seg_pos[rays->seg_vec_offset(1, ray_id, seg_id)] = 0.0;
        seg_pos[rays->seg_vec_offset(2, ray_id, seg_id)] = 0.0;
        seg_dir[rays->seg_vec_offset(0, ray_id, seg_id)] = 0.0;
        seg_dir[rays->seg_vec_offset(1, ray_id, seg_id)] = 0.0;
        seg_dir[rays->seg_vec_offset(2, ray_id, seg_id)] = 0.0;
        for (int64_t llm = 0; llm < n_llm; ++llm) {
          seg_rn[global_seg_id * n_llm + llm] = 0.0;
        }
        seg_id++;
        continue;
      }

      const geom_t ray_step_distance = distance_to_surf;
      geom_t seg_start_x = pos_x;
      geom_t seg_start_y = pos_y;
      geom_t seg_start_z = pos_z;
      geom_t seg_distance = distance_to_surf;
      geom_t ray_advance = 0.0;
      bool clipped = false;

      if (!ray_active &&
          ray_distance + distance_to_surf >= d_settings.inactive_length) {
        const geom_t inactive_part = d_settings.inactive_length - ray_distance;
        seg_dis[global_seg_id] = inactive_part;
        seg_isactive[global_seg_id] = false;
        seg_boundary[global_seg_id] = -1;
        seg_index[global_seg_id] = cuda_get_region_id(idx_x, idx_y, idx_z);
        seg_pos[rays->seg_vec_offset(0, ray_id, seg_id)] = pos_x;
        seg_pos[rays->seg_vec_offset(1, ray_id, seg_id)] = pos_y;
        seg_pos[rays->seg_vec_offset(2, ray_id, seg_id)] = pos_z;
        seg_dir[rays->seg_vec_offset(0, ray_id, seg_id)] = u_x;
        seg_dir[rays->seg_vec_offset(1, ray_id, seg_id)] = u_y;
        seg_dir[rays->seg_vec_offset(2, ray_id, seg_id)] = u_z;
        seg_rn[global_seg_id] = static_cast<real_t>(1.0);
        if (inactive_part > static_cast<geom_t>(0.0)) {
          ++segments_in_ray;
        }

        ray_active = true;
        ray_advance += inactive_part;
        seg_start_x = pos_x + u_x * inactive_part;
        seg_start_y = pos_y + u_y * inactive_part;
        seg_start_z = pos_z + u_z * inactive_part;
        seg_distance = ray_step_distance - inactive_part;

        seg_id++;
        if (seg_id >= n_seg) {
          // The inactive segment has been recorded, but there is no slot left
          // for the active segment in this kernel pass. Resume next pass from
          // the active-phase start position.
          pos_x = seg_start_x;
          pos_y = seg_start_y;
          pos_z = seg_start_z;
          ray_distance = 0.0;
          break;
        }
        if (wgt[ray_id] <= 0.0) {
          continue;
        }

        global_seg_id = ray_id * n_seg + seg_id;
        if (seg_distance > d_settings.active_length) {
          seg_distance = d_settings.active_length;
          wgt[ray_id] = 0.0;
          clipped = true;
        }

        ray_distance = seg_distance;
      } else if (ray_active) {
        if (ray_distance + seg_distance >= d_settings.active_length) {
          seg_distance = d_settings.active_length - ray_distance;
          wgt[ray_id] = 0.0;
          clipped = true;
        }
        ray_distance += seg_distance;
      } else {
        ray_distance += seg_distance;
      }

      int8_t boundary = -1;
      if (!clipped) {
        if (cross_dim == 0) {
          if (idx_x == 0 && cross_x < 0) {
            boundary = static_cast<int8_t>(d_settings.x0_boundary);
          } else if (idx_x == d_divisions[0]->n_ - 1 && cross_x > 0) {
            boundary = static_cast<int8_t>(d_settings.x1_boundary);
          }
        } else if (cross_dim == 1) {
          if (idx_y == 0 && cross_y < 0) {
            boundary = static_cast<int8_t>(d_settings.y0_boundary);
          } else if (idx_y == d_divisions[1]->n_ - 1 && cross_y > 0) {
            boundary = static_cast<int8_t>(d_settings.y1_boundary);
          }
        } else if (cross_dim == 2) {
          if (idx_z == 0 && cross_z < 0) {
            boundary = static_cast<int8_t>(d_settings.z0_boundary);
          } else if (idx_z == d_divisions[2]->n_ - 1 && cross_z > 0) {
            boundary = static_cast<int8_t>(d_settings.z1_boundary);
          }
        }
      }

      seg_dis[global_seg_id] = seg_distance;
      seg_isactive[global_seg_id] = ray_active;
      seg_boundary[global_seg_id] = boundary;
      seg_index[global_seg_id] = cuda_get_region_id(idx_x, idx_y, idx_z);
      seg_pos[rays->seg_vec_offset(0, ray_id, seg_id)] = seg_start_x;
      seg_pos[rays->seg_vec_offset(1, ray_id, seg_id)] = seg_start_y;
      seg_pos[rays->seg_vec_offset(2, ray_id, seg_id)] = seg_start_z;
      seg_dir[rays->seg_vec_offset(0, ray_id, seg_id)] = u_x;
      seg_dir[rays->seg_vec_offset(1, ray_id, seg_id)] = u_y;
      seg_dir[rays->seg_vec_offset(2, ray_id, seg_id)] = u_z;
      seg_rn[global_seg_id] = static_cast<real_t>(1.0);
      if (seg_distance > static_cast<geom_t>(0.0)) {
        ++segments_in_ray;
      }

      ray_advance += seg_distance;

      pos_x += u_x * ray_advance;
      pos_y += u_y * ray_advance;
      pos_z += u_z * ray_advance;

      if (wgt[ray_id] <= 0.0) {
        seg_id++;
        continue;
      }

      if (cross_dim == 0) {
        pos_x = cross_destination;
      } else if (cross_dim == 1) {
        pos_y = cross_destination;
      } else if (cross_dim == 2) {
        pos_z = cross_destination;
      }

      if (cross_dim == 0) {
        cuda_cross_axis(ray_id, 0, &idx_x, &cross_x, &u_x,
          d_settings.x0_boundary, d_settings.x1_boundary, false);
      } else if (cross_dim == 1) {
        cuda_cross_axis(ray_id, 1, &idx_y, &cross_y, &u_y,
          d_settings.y0_boundary, d_settings.y1_boundary, false);
      } else if (cross_dim == 2) {
        cuda_cross_axis(ray_id, 2, &idx_z, &cross_z, &u_z,
          d_settings.z0_boundary, d_settings.z1_boundary, false);
      }
      seg_id++;
    }

    pos[rays->vec_offset(0, ray_id)] = pos_x;
    pos[rays->vec_offset(1, ray_id)] = pos_y;
    pos[rays->vec_offset(2, ray_id)] = pos_z;
    dir[rays->vec_offset(0, ray_id)] = u_x;
    dir[rays->vec_offset(1, ray_id)] = u_y;
    dir[rays->vec_offset(2, ray_id)] = u_z;
    index[rays->vec_offset(0, ray_id)] = idx_x;
    index[rays->vec_offset(1, ray_id)] = idx_y;
    index[rays->vec_offset(2, ray_id)] = idx_z;
    distance_travelled[ray_id] = ray_distance;
    rays->is_active[ray_id] = ray_active;
    if (segments_in_ray > 0) {
      // Use one atomic update per ray rather than one update per segment.
      atomicAdd(segment_count, segments_in_ray);
    }
  }

  if (false) {
    for (int __ray_id = 0; __ray_id < d_settings.n_track; __ray_id++) {
      // printf("\nray %u pos %.6f %.6f %.6f dir %.6f %.6f %.6f wgt %.6f \n",
      //   __ray_id, static_cast<double>(pos[rays->vec_offset(0, __ray_id)]),
      //   static_cast<double>(pos[rays->vec_offset(1, __ray_id)]),
      //   static_cast<double>(pos[rays->vec_offset(2, __ray_id)]),
      //   static_cast<double>(dir[rays->vec_offset(0, __ray_id)]),
      //   static_cast<double>(dir[rays->vec_offset(1, __ray_id)]),
      //   static_cast<double>(dir[rays->vec_offset(2, __ray_id)]),
      //   static_cast<double>(wgt[__ray_id]));

      // Debug output for angular flux and segment information
      // printf("angular flux: ");
      // for (int64_t g = 0; g < d_settings.n_groups; g++) {
      //   printf(
      //     "%.6f ", static_cast<double>(
      //                rays->angular_flux[__ray_id * d_settings.n_groups +
      //                g]));
      // }
      // printf("\n");

      // printf("seg_rn :");
      // for (int64_t rn_idx = 0;
      //   rn_idx < d_settings.n_order * d_settings.n_order * n_seg; rn_idx++) {
      //   printf("%.6f ", static_cast<double>(rays->seg_rn[rn_idx]));
      // }
      // printf("\n");

      printf("\nseg_dis");
      for (int64_t seg_id = 0; seg_id < n_seg; seg_id++) {
        int64_t seg_global = static_cast<int64_t>(__ray_id) * n_seg + seg_id;
        printf(" %.10f", static_cast<double>(seg_dis[seg_global]));
      }

      printf("\nseg_index");
      for (int64_t seg_id = 0; seg_id < n_seg; seg_id++) {
        int64_t seg_global = static_cast<int64_t>(__ray_id) * n_seg + seg_id;
        printf(" %lld", static_cast<long long>(seg_index[seg_global]));
      }

      printf("\nseg_isactive");
      for (int64_t seg_id = 0; seg_id < n_seg; seg_id++) {
        int64_t seg_global = static_cast<int64_t>(__ray_id) * n_seg + seg_id;
        printf(" %d", seg_isactive[seg_global] ? 1 : 0);
      }

      printf("\n");
      for (int64_t seg_id = 0; seg_id < n_seg; seg_id++) {
        int64_t seg_global = static_cast<int64_t>(__ray_id) * n_seg + seg_id;
        printf("ray %d seg %lld pos: %.20f %.20f %.20f dir: %.3f %.3f %.3f\n",
          __ray_id, static_cast<long long>(seg_id),
          static_cast<double>(
            seg_pos[rays->seg_vec_offset(0, __ray_id, seg_id)]),
          static_cast<double>(
            seg_pos[rays->seg_vec_offset(1, __ray_id, seg_id)]),
          static_cast<double>(
            seg_pos[rays->seg_vec_offset(2, __ray_id, seg_id)]),
          static_cast<double>(
            seg_dir[rays->seg_vec_offset(0, __ray_id, seg_id)]),
          static_cast<double>(
            seg_dir[rays->seg_vec_offset(1, __ray_id, seg_id)]),
          static_cast<double>(
            seg_dir[rays->seg_vec_offset(2, __ray_id, seg_id)]));
      }
      printf("\n");
    }
  }
}

__global__ void cuda_attenuate_flux_kernel()
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t stride = gridDim.x * blockDim.x;
  const int64_t n_groups = d_settings.n_groups;
  const int64_t n_order = d_settings.n_order;
  const int64_t n_llm = n_order * n_order;
  const int64_t n_rays = d_settings.n_track;
  const int64_t n_seg = d_settings.event_seg_count;
  const int64_t total = n_rays * n_groups;

  CuRays* __restrict__ rays = device_rays;
  CuFlatSourceDomain* __restrict__ fs = device_fs_domain;
  CudaMatXs* __restrict__ mats = d_mats;

  for (int idx = tid; idx < total; idx += stride) {
    int64_t ray_id = idx / n_groups;
    int64_t g = idx - ray_id * n_groups;
    const int64_t flux_index = ray_id * n_groups + g;

    for (int64_t seg_id = 0; seg_id < n_seg; seg_id++) {
      int64_t seg_global = ray_id * n_seg + seg_id;
      real_t distance = rays->seg_distance[seg_global];
      if (distance <= 0.0) {
        continue;
      }

      int64_t i_reg = rays->seg_index[seg_global];
      if (i_reg < 0 || i_reg >= d_settings.n_region) {
        continue;
      }

      const bool is_active = rays->seg_isactive[seg_global];
      const int64_t i_src = i_reg * n_groups * n_order * n_order;
      const int64_t i_mat = fs->materials_[i_reg];

      const real_t sigma_t = mats->sigma_t[i_mat * n_groups + g];
      const real_t optical_depth = sigma_t * distance;
#if defined(THETA_CUDA_PRECISION_DOUBLE)
      const real_t attenuation_factor =
        double_cjosey_exponential(optical_depth);
#else
      const real_t attenuation_factor = cjosey_exponential(optical_depth);
#endif

      // !
      real_t scattering_source = 0.0;
      const int64_t seg_rn_offset = seg_global * n_llm;
      const int64_t src_base = i_src + g;
      for (int64_t llm = 0; llm < n_llm; ++llm) {
        const real_t harmonic_coeff = rays->seg_rn[seg_rn_offset + llm];
        const int64_t src_index = src_base + llm * n_groups;
        scattering_source += harmonic_coeff * fs->source_[src_index];
      }

      // real_t angular_flux = rays->angular_flux[flux_index];
      // printf("angular_flux %.6f \n", angular_flux);
      const real_t delta_flux =
        (rays->angular_flux[flux_index] - scattering_source) *
        attenuation_factor;

      rays->angular_flux[flux_index] -= delta_flux;

      if (rays->seg_boundary[seg_global] == 0) {
        rays->angular_flux[flux_index] = 0.0;
      }

      if (is_active) {
        if (g == 0) {
          atomicAdd(&fs->n_tracks_[i_reg], 1);
          if (d_settings.volume_type !=
              static_cast<size_t>(VolumeType::Analytic)) {
            // English: Volumes use geometry precision; convert the transport
            // distance when building in mixed precision.
            atomicAdd(&fs->volume_[i_reg], static_cast<geom_t>(distance));
          }
        }
        if (delta_flux != 0.0) {
          const int64_t src_base = i_src + g;
          for (int64_t llm = 0; llm < n_llm; ++llm) {
            const real_t harmonic_coeff = rays->seg_rn[seg_rn_offset + llm];
            const int64_t src_index = src_base + llm * n_groups;
            atomicAdd(
              &fs->new_scalar_flux_[src_index], delta_flux * harmonic_coeff);
          }
        }
      }
    }
  }
}

__global__ void cuda_count_alive_rays(unsigned long long* alive_count)
{
  extern __shared__ unsigned int s_count[];
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t stride = gridDim.x * blockDim.x;
  unsigned int local = 0;
  CuRays* __restrict__ rays = device_rays;
  for (int id = tid; id < d_settings.n_track; id += stride) {
    if (rays->wgt[id] > 0.0) {
      local += 1;
    }
  }
  unsigned int lane = threadIdx.x;
  s_count[lane] = local;
  __syncthreads();
  for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (lane < offset) {
      s_count[lane] += s_count[lane + offset];
    }
    __syncthreads();
  }
#ifdef __CUDA_ARCH__
  if (lane == 0) {
    atomicAdd(alive_count, static_cast<unsigned long long>(s_count[0]));
  }
#endif
}

__global__ void cuda_normalize_flux(size_t i_batch)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t stride = gridDim.x * blockDim.x;
  CuFlatSourceDomain* __restrict__ fs = device_fs_domain;
  for (int i_reg = tid; i_reg < d_settings.n_region; i_reg += stride) {
    // Calculate normalization factor (per cycle and accumulated value)
    geom_t total_distance_per_cycle =
      d_settings.n_track * d_settings.active_length;
    geom_t norm_cycle = 1.0 / total_distance_per_cycle;
    geom_t total_distance = total_distance_per_cycle * (i_batch + 1);
    geom_t norm_total = 1.0 / total_distance;
    // Calculate estimated volume
    if (d_settings.volume_type != static_cast<size_t>(VolumeType::Analytic)) {
      if (d_settings.volume_type == static_cast<size_t>(VolumeType::Adaptive) ||
          d_settings.volume_type ==
            static_cast<size_t>(VolumeType::BatchEstimate)) {
        // Use estimated volume per batch
        fs->volume_naive_[i_reg] = fs->volume_[i_reg] * norm_cycle;
      }

      if (d_settings.volume_type ==
          static_cast<size_t>(VolumeType::AverageEstimate)) {
        // Use average estimate for volume
        fs->volume_estimated_[i_reg] += fs->volume_[i_reg];
        fs->volume_[i_reg] = fs->volume_estimated_[i_reg] * norm_total;
      }
    }
    for (int64_t g = 0;
         g < d_settings.n_groups * d_settings.n_order * d_settings.n_order;
         g++) {
      int64_t i_src =
        i_reg * d_settings.n_groups * d_settings.n_order * d_settings.n_order +
        g;
      fs->new_scalar_flux_[i_src] *= norm_cycle;
    }

  }
}

__global__ void cuda_add_source_to_scalar_flux()
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t stride = gridDim.x * blockDim.x;
  CuFlatSourceDomain* __restrict__ fs = device_fs_domain;
  CudaMatXs* __restrict__ mats = d_mats;
  for (int i_reg = tid; i_reg < d_settings.n_region; i_reg += stride) {
    geom_t volume;
    // Analytic case
    if (d_settings.volume_type == static_cast<size_t>(VolumeType::Analytic)) {
      volume = fs->volume_analytic_[i_reg];
    }
    // Adaptive case
    else if (d_settings.volume_type ==
             static_cast<size_t>(VolumeType::Adaptive)) {
      volume = fs->volume_analytic_[i_reg];
      // Treat small region and external source region
      if (fs->has_external_source_[i_reg] ||
          (fs->n_tracks_[i_reg] < 5 && fs->n_tracks_[i_reg] > 0)) {
        volume = fs->volume_naive_[i_reg];
      }
    }
    // Batch estimate volume case
    else if (d_settings.volume_type ==
             static_cast<size_t>(VolumeType::BatchEstimate)) {
      volume = fs->volume_naive_[i_reg];
    }
    // Total batch average estimate case
    else {
      volume = fs->volume_[i_reg];
    }

    // Set material and source idx
    int64_t i_mat = fs->materials_[i_reg];
    int64_t i_src =
      i_reg * d_settings.n_groups * d_settings.n_order * d_settings.n_order;
    for (int64_t l = 0; l < d_settings.n_order; l++) {
      for (int64_t m = 0; m < 2 * l + 1; m++) {
        int64_t llm = l * l + m;
        int64_t _ng = d_settings.n_groups;
        if (fs->n_tracks_[i_reg] > 0) {
          // Hit in this cycle
          for (int64_t g = 0; g < _ng; g++) {
            real_t sigma_t = mats->sigma_t[i_mat * _ng + g];

            // High order scalar flux
            fs->new_scalar_flux_[i_src + llm * _ng + g] /=
              static_cast<flux_t>(volume) * sigma_t;

            // Source term
            fs->new_scalar_flux_[i_src + llm * _ng + g] +=
              fs->source_[i_src + llm * _ng + g];


          }

          // Flag the region has been hitting
          if (!fs->whether_hit_[i_reg]) {
            fs->whether_hit_[i_reg] = true;
          }
        } else if (fs->whether_hit_[i_reg]) {
          // Missed in this cycle but hit before. Same operations as THETA
          if (fs->has_external_source_[i_reg]) {
            for (int64_t g = 0; g < _ng; g++) {
              fs->new_scalar_flux_[i_src + llm * _ng + g] =
                fs->old_scalar_flux_[i_src + llm * _ng + g];


            }
          } else {
            for (int64_t g = 0; g < _ng; g++) {
              fs->new_scalar_flux_[i_src + llm * _ng + g] =
                fs->source_[i_src + llm * _ng + g];

            }
          }
        } else {
          // Never hit
          for (int64_t g = 0; g < _ng; g++) {
            fs->new_scalar_flux_[i_src + llm * _ng + g] = 0.0;

          }
        }
      }
    }
  }
}

__global__ void cuda_compute_k_eff_kernel()
{
  // English: The keff reduction buffer uses iteration precision, which is
  // double precision in mixed mode.
  extern __shared__ iter_t sdata[];

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t stride = gridDim.x * blockDim.x;

  iter_t* fiss_rate_new = &cu_simulation.fiss_rate_new;
  iter_t* fiss_rate_old = &cu_simulation.fiss_rate_old;
  CuFlatSourceDomain* __restrict__ fs = device_fs_domain;
  CudaMatXs* __restrict__ mats = d_mats;

  iter_t local_fiss_rate_old = 0.0;
  iter_t local_fiss_rate_new = 0.0;
  int64_t n_region = d_settings.n_region;
  int64_t ng = d_settings.n_groups;
  int64_t n_order = d_settings.n_order;
  for (int i = tid; i < n_region; i += stride) {
    const geom_t volume = fs->volume_analytic_[i];
    int64_t i_mat = fs->materials_[i];

    if (volume == 0.0 || (!mats->fissionable[i_mat])) {
      continue;
    }

    int64_t i_src = i * ng * n_order * n_order;

    iter_t i_fiss_rate_old = 0.0;
    iter_t i_fiss_rate_new = 0.0;

    for (int64_t g = 0; g < ng; g++) {
      i_fiss_rate_old += static_cast<iter_t>(fs->old_scalar_flux_[i_src + g]) *
                         static_cast<iter_t>(mats->nu_sigma_f[i_mat * ng + g]);
      i_fiss_rate_new += static_cast<iter_t>(fs->new_scalar_flux_[i_src + g]) *
                         static_cast<iter_t>(mats->nu_sigma_f[i_mat * ng + g]);
    }

    i_fiss_rate_old *= static_cast<iter_t>(volume);
    i_fiss_rate_new *= static_cast<iter_t>(volume);

    local_fiss_rate_old += i_fiss_rate_old;
    local_fiss_rate_new += i_fiss_rate_new;
  }

  int local_tid = threadIdx.x;
  sdata[local_tid] = local_fiss_rate_old;
  sdata[blockDim.x + local_tid] = local_fiss_rate_new;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (local_tid < s) {
      sdata[local_tid] += sdata[local_tid + s];
      sdata[blockDim.x + local_tid] += sdata[blockDim.x + local_tid + s];
    }
    __syncthreads();
  }

  if (local_tid == 0) {
    atomicAdd(fiss_rate_old, sdata[0]);
    atomicAdd(fiss_rate_new, sdata[blockDim.x]);
  }
}

__global__ void cuda_update_keff_kernel(int64_t current_batch)
{
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    iter_t* fiss_rate_new = &cu_simulation.fiss_rate_new;
    iter_t* fiss_rate_old = &cu_simulation.fiss_rate_old;
    iter_t* keff_cycle = &cu_simulation.keff_cycle;

    if (*fiss_rate_old > 0) {
      const iter_t new_keff = *keff_cycle * (*fiss_rate_new) / (*fiss_rate_old);
      *keff_cycle = new_keff;
      printf("keff: %.5f ", new_keff);
    }

    if (current_batch >= d_settings.inactive_batch) {
      cu_simulation.mean_keff += *keff_cycle;
      cu_simulation.std_dev_keff += (*keff_cycle) * (*keff_cycle);
    }
    *fiss_rate_new = 0.0;
    *fiss_rate_old = 0.0;
  }
}

// __global__ void cuda_calc_keff()
// {
//   int tid = blockIdx.x * blockDim.x + threadIdx.x;
//   int64_t stride = gridDim.x * blockDim.x;
//   real_t* fiss_rate_new = &cu_simulation.fiss_rate_new;
//   real_t* fiss_rate_old = &cu_simulation.fiss_rate_old;
//   *fiss_rate_new = 0.0;
//   *fiss_rate_old = 0.0;
//   real_t* keff_cycle = &cu_simulation.keff_cycle;
//   CuFlatSourceDomain* __restrict__ fs = device_fs_domain;
//   CudaMatXs* __restrict__ mats = d_mats;
//   int64_t n_region = d_settings.n_region;
//   int64_t ng = d_settings.n_groups;
//   int64_t n_order = d_settings.n_order;
//   for (int i_reg = tid; i_reg < d_settings.n_region; i_reg += stride) {
//     if (i_reg >= d_settings.n_region) {
//       return;
//     }
//     real_t volume = fs->volume_analytic_[i_reg];
//     int64_t i_mat = fs->materials_[i_reg];

//     if (volume == 0.0 || (!mats->fissionable[i_mat])) {
//       continue;
//     }

//     int64_t i_src = i_reg * ng * n_order * n_order;

//     real_t i_fiss_rate_old = 0.0;
//     real_t i_fiss_rate_new = 0.0;

//     for (int64_t g = 0; g < ng; g++) {
//       i_fiss_rate_old +=
//         fs->old_scalar_flux_[i_src + g] * mats->nu_sigma_f[i_mat * ng + g];
//       i_fiss_rate_new +=
//         fs->new_scalar_flux_[i_src + g] * mats->nu_sigma_f[i_mat * ng + g];
//     }

//     i_fiss_rate_old *= volume;
//     i_fiss_rate_new *= volume;

//     *fiss_rate_old += i_fiss_rate_old;
//     *fiss_rate_new += i_fiss_rate_new;
//   }
//   real_t new_keff = *keff_cycle * (*fiss_rate_new) / (*fiss_rate_old);
//   *keff_cycle = new_keff;
//   printf("keff: %f\n", new_keff);
// }

__global__ void cuda_calc_final(int64_t current_batch)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t stride = gridDim.x * blockDim.x;
  CuFlatSourceDomain* __restrict__ fs = device_fs_domain;
  for (int i_src = tid; i_src < d_settings.n_source_region_high_order;
       i_src += stride) {
    if (current_batch >= d_settings.inactive_batch) {
      fs->final_scalar_flux_[i_src] += fs->new_scalar_flux_[i_src];
      fs->final_scalar_flux_sq_[i_src] +=
        fs->new_scalar_flux_[i_src] * fs->new_scalar_flux_[i_src];
    }
    fs->old_scalar_flux_[i_src] = fs->new_scalar_flux_[i_src];
  }


}

__global__ void cuda_calc_mean_std_dev(bool calculate_keff_stats)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t stride = gridDim.x * blockDim.x;
  int64_t n_active_batch = d_settings.n_batch - d_settings.inactive_batch;
  CuFlatSourceDomain* __restrict__ fs = device_fs_domain;
  for (int element = tid; element < d_settings.n_source_region_high_order;
       element += stride) {
    real_t flx = fs->final_scalar_flux_[element] / n_active_batch;
    real_t flx_sq = fs->final_scalar_flux_sq_[element];
    fs->final_scalar_flux_[element] = flx;
    if (flx != 0.0) {
      fs->final_scalar_flux_sq_[element] =
        sqrt(abs(flx_sq / n_active_batch - flx * flx) / (n_active_batch - 1));
    }
  }
  if (calculate_keff_stats && tid == 0) {
    cu_simulation.mean_keff /= n_active_batch;
    cu_simulation.std_dev_keff /= n_active_batch;
    cu_simulation.std_dev_keff =
      sqrt(abs(cu_simulation.std_dev_keff -
               cu_simulation.mean_keff * cu_simulation.mean_keff) /
           (n_active_batch - 1));
    printf("mean keff: %.5f\n", cu_simulation.mean_keff);
    printf("std dev keff: %.6f\n", cu_simulation.std_dev_keff);
  }
}

__global__ void cuda_print_flux()
{
  real_t* __restrict__ flux = device_fs_domain->new_scalar_flux_;
  printf("print flux\n");
  for (int i = 0; i < d_settings.n_source_region_high_order; i++) {
    printf("%.4e ", flux[i]);
  }
  printf("\n");
}

void run_simulation(iter_t keff, double& mean_keff, double& std_dev_keff)
{
  int64_t current_batch = 0;
  int grid_size = 0;
  // Keep auxiliary kernels at 128 threads so the experiment changes only event
  // tracking and attenuation.
  const int auxiliary_block_size = DEBUG ? 1 : 128;
  const int track_block_size = DEBUG ? 1 : settings::cuda_track_block_size;
  const int attenuate_block_size =
    DEBUG ? 1 : settings::cuda_attenuate_block_size;
  int64_t n_region = simulation::model->n_region();
  int64_t n_source_region_high_order =
    simulation::model->n_source_region_high_order();
  unsigned long long* d_alive_count = nullptr;
  unsigned long long* d_segment_count = nullptr;
  unsigned long long h_alive_count = 0;
  unsigned long long h_segment_count = 0;
  const bool event_algorithm =
    settings::transport_algorithm == TransportAlgorithm::Event;

  cugo(cudaMalloc(
    reinterpret_cast<void**>(&d_segment_count), sizeof(unsigned long long)));
  // settings, materials, flat_source_domain, divisions
  timer::cuda_time_total.start();
  timer::cuda_time_initialize.start();
  cuda_load_data();
  timer::cuda_time_initialize.stop();
  while (current_batch < settings::n_batch) {
    std::vector<CudaKernelEventPair> track_kernel_events;
    std::vector<CudaKernelEventPair> attenuate_kernel_events;
    int64_t event_while_iterations = 0;
    h_segment_count = 0;
    cugo(cudaMemset(d_segment_count, 0, sizeof(unsigned long long)));

    printf("batch: %lld ", static_cast<long long>(current_batch + 1));
    timer::cuda_time_inactive.start();
    if (current_batch == settings::inactive_batch) {
      timer::cuda_time_active.start();
      timer::cuda_time_inactive.stop();
    }
    grid_size = DEBUG
                  ? 1
                  : (n_source_region_high_order + auxiliary_block_size - 1) /
                      auxiliary_block_size;
    cuda_init_batch<<<grid_size, auxiliary_block_size>>>();
    // update flat source domain
    timer::cuda_time_update_src.start();
    cuda_update_flat_source_domain(keff);

    cuda_add_external_source_to_flat_source_domain();
    timer::cuda_time_update_src.stop();

    size_t n_rays = settings::n_track;
    grid_size =
      DEBUG ? 1 : (n_rays + auxiliary_block_size - 1) / auxiliary_block_size;
    if (current_batch == 0) {
      cuda_update_rays<<<grid_size, auxiliary_block_size>>>(current_batch);
      if (settings::ray_mode == RayMode::IMMORTAL) {
        cuda_update_rays_immortal<<<grid_size, auxiliary_block_size>>>(
          current_batch, 0.0);
      }
    } else if (settings::ray_mode == RayMode::IMMORTAL) {
      cuda_update_rays_immortal<<<grid_size, auxiliary_block_size>>>(
        current_batch, settings::ray_resample_rate);
    } else {
      cuda_update_rays<<<grid_size, auxiliary_block_size>>>(current_batch);
    }

    if (d_alive_count == nullptr) {
      cudaMalloc((void**)&d_alive_count, sizeof(unsigned long long));
    }
    cudaDeviceSynchronize();

    timer::cuda_time_transport.start();
    // if (DEBUG) {
    //   printf("delta_psi_ : ");
    // }
    if (settings::transport_algorithm == TransportAlgorithm::History) {

        cuda_track_history<<<grid_size, auxiliary_block_size>>>(
          d_segment_count);

    } else {
      // Compute tracking and attenuation grids separately so block-size changes
      // cover every ray and ray-group task.
      const int track_grid_size =
        DEBUG ? 1 : (n_rays + track_block_size - 1) / track_block_size;
      while (true) {
        ++event_while_iterations;
        CudaKernelEventPair track_kernel_event;
        CudaKernelEventPair attenuate_kernel_event;
        cugo(cudaEventCreate(&track_kernel_event.start));
        cugo(cudaEventCreate(&track_kernel_event.stop));
        cugo(cudaEventCreate(&attenuate_kernel_event.start));
        cugo(cudaEventCreate(&attenuate_kernel_event.stop));

        cugo(cudaEventRecord(track_kernel_event.start));
        cuda_track_events<<<track_grid_size, track_block_size>>>(
          d_segment_count);
        cugo(cudaEventRecord(track_kernel_event.stop));

        int64_t n_ray_groups =
          static_cast<int64_t>(n_rays) * simulation::model->ng();
        const int attenuate_grid_size =
          DEBUG
            ? 1
            : (n_ray_groups + attenuate_block_size - 1) / attenuate_block_size;
        cugo(cudaEventRecord(attenuate_kernel_event.start));

          cuda_attenuate_flux_kernel<<<attenuate_grid_size,
            attenuate_block_size>>>();

        cugo(cudaEventRecord(attenuate_kernel_event.stop));
        track_kernel_events.push_back(track_kernel_event);
        attenuate_kernel_events.push_back(attenuate_kernel_event);
        cudaMemset(d_alive_count, 0, sizeof(unsigned long long));
        cuda_count_alive_rays<<<grid_size, auxiliary_block_size,
          auxiliary_block_size * sizeof(unsigned int)>>>(d_alive_count);
        cudaMemcpy(&h_alive_count, d_alive_count, sizeof(unsigned long long),
          cudaMemcpyDeviceToHost);
        if (h_alive_count == 0) {
          break;
        }
      }
    }

    cudaDeviceSynchronize();
    if (event_algorithm) {
      // Query event timings only after the batch completes so no per-pass
      // device synchronization is introduced.
      timer::cuda_time_event_track.add(
        collect_cuda_kernel_time(track_kernel_events));
      timer::cuda_time_event_attenuate.add(
        collect_cuda_kernel_time(attenuate_kernel_events));
    }
    cugo(cudaMemcpy(&h_segment_count, d_segment_count,
      sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    timer::cuda_time_transport.stop();

    grid_size =
      DEBUG ? 1 : (n_region + auxiliary_block_size - 1) / auxiliary_block_size;
    timer::cuda_time_trrm_flux_treatment.start();
    cuda_normalize_flux<<<grid_size, auxiliary_block_size>>>(current_batch);

    cuda_add_source_to_scalar_flux<<<grid_size, auxiliary_block_size>>>();
    timer::cuda_time_trrm_flux_treatment.stop();

    if (settings::run_mode == RunMode::EIGENVALUE) {
      cuda_compute_k_eff_kernel<<<grid_size, auxiliary_block_size,
        auxiliary_block_size * 2 * sizeof(iter_t)>>>();
      cuda_update_keff_kernel<<<1, 1>>>(current_batch);
    }

    timer::cuda_time_finalize.start();
    grid_size = DEBUG
                  ? 1
                  : (n_source_region_high_order + auxiliary_block_size - 1) /
                      auxiliary_block_size;
    cuda_calc_final<<<grid_size, auxiliary_block_size>>>(current_batch);
    timer::cuda_time_finalize.stop();
    cudaDeviceSynchronize();


    double batch_keff = 0.0;
    double batch_keff_std = 0.0;
    if (settings::run_mode == RunMode::EIGENVALUE) {
      cugo(cudaMemcpyFromSymbol(&host_simulation, cu_simulation,
        sizeof(CuSimulation), 0, cudaMemcpyDeviceToHost));
      if (current_batch >= settings::inactive_batch) {
        batch_keff = static_cast<double>(host_simulation.keff_cycle);
        const int64_t n_active = current_batch - settings::inactive_batch + 1;
        if (n_active > 1) {
          const double mean_sum =
            static_cast<double>(host_simulation.mean_keff);
          const double square_sum =
            static_cast<double>(host_simulation.std_dev_keff);
          const double mean = mean_sum / n_active;
          batch_keff_std = std::sqrt(
            std::abs(square_sum / n_active - mean * mean) / (n_active - 1));
        }
      }
    }
    run_info::record_batch(current_batch + 1,
      event_algorithm ? event_while_iterations : 0, h_segment_count, batch_keff,
      batch_keff_std);

    current_batch++;
    printf("\n");
    fflush(stdout);
  }
  timer::cuda_time_active.stop();
  if (d_alive_count != nullptr) {
    cudaFree(d_alive_count);
  }
  if (d_segment_count != nullptr) {
    cudaFree(d_segment_count);
  }
  cuda_calc_mean_std_dev<<<grid_size, auxiliary_block_size>>>(
    settings::run_mode == RunMode::EIGENVALUE);
  cudaDeviceSynchronize();

  cudaError_t stats_error = cudaMemcpyFromSymbol(&host_simulation,
    cu_simulation, sizeof(CuSimulation), 0, cudaMemcpyDeviceToHost);
  if (stats_error != cudaSuccess) {
    fprintf(stderr, "cudaMemcpyFromSymbol cu_simulation failed: %s\n",
      cudaGetErrorString(stats_error));
    mean_keff = 0.0;
    std_dev_keff = 0.0;
  } else {
    mean_keff = static_cast<double>(host_simulation.mean_keff);
    std_dev_keff = static_cast<double>(host_simulation.std_dev_keff);
  }
  timer::cuda_time_total.stop();
}

} // namespace gpu
} // namespace theta
#endif
