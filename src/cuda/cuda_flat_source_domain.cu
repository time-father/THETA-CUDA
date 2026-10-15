#include "cuh/cuda_flat_source_domain.cuh"
#include "cuh/cuda_interface.cuh"
#include "cuh/cuda_settings.cuh"
#include "cuh/cuda_simulation.cuh"
#include "simulation.h"
#include <iostream>
#include <vector>

#ifdef THETA_CUDA
namespace theta {
namespace gpu {

CuFlatSourceDomain* host_fs_domain;
__device__ CuFlatSourceDomain* __restrict__ device_fs_domain;

void cuda_initialize_flat_source_domain()
{
  host_fs_domain = (CuFlatSourceDomain*)malloc(sizeof(CuFlatSourceDomain));
  if (host_fs_domain == nullptr) {
    fprintf(stderr, "Failed to allocate host memory for CuFlatSourceDomain\n");
    exit(EXIT_FAILURE);
  }
  memset(host_fs_domain, 0, sizeof(CuFlatSourceDomain));
}

void cuda_free_flat_source_domain()
{
  if (host_fs_domain != nullptr) {
    cudaFree(host_fs_domain->n_tracks_);
    cudaFree(host_fs_domain->ng_);
    cudaFree(host_fs_domain->n_dim_);
    cudaFree(host_fs_domain->n_order_);
    cudaFree(host_fs_domain->n_region_);
    cudaFree(host_fs_domain->n_source_region_);
    cudaFree(host_fs_domain->n_source_region_high_order_);
    //
    cudaFree(host_fs_domain->new_scalar_flux_);
    cudaFree(host_fs_domain->old_scalar_flux_);
    cudaFree(host_fs_domain->final_scalar_flux_);
    cudaFree(host_fs_domain->final_scalar_flux_sq_);
    cudaFree(host_fs_domain->source_);
    cudaFree(host_fs_domain->external_source_);
    cudaFree(host_fs_domain->volume_analytic_);
    cudaFree(host_fs_domain->volume_estimated_);
    cudaFree(host_fs_domain->volume_naive_);
    cudaFree(host_fs_domain->volume_);
    //
    cudaFree(host_fs_domain->has_external_source_);
    cudaFree(host_fs_domain->whether_hit_);
    //
    cudaFree(host_fs_domain);
    host_fs_domain = nullptr;
  }
}

bool cuda_download_data(std::vector<double>& final_scalar_flux,
  std::vector<double>& final_scalar_flux_sq)
{
  if (host_fs_domain == nullptr) {
    return false;
  }

  auto copy_double = [](std::vector<double>& dst, real_t* src,
                       const char* name) -> bool {
    if (dst.empty() || src == nullptr) {
      return true;
    }
    std::vector<real_t> temp(dst.size());
    cudaError_t err = cudaMemcpy(
      temp.data(), src, sizeof(real_t) * dst.size(), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
      fprintf(
        stderr, "cudaMemcpy %s failed: %s\n", name, cudaGetErrorString(err));
      return false;
    }
    dst.resize(temp.size());
    for (size_t i = 0; i < temp.size(); ++i) {
      dst[i] = static_cast<double>(temp[i]);
    }
    return true;
  };

  bool ok = true;
  ok = copy_double(final_scalar_flux, host_fs_domain->final_scalar_flux_,
         "final_scalar_flux") &&
       ok;
  ok = copy_double(final_scalar_flux_sq, host_fs_domain->final_scalar_flux_sq_,
         "final_scalar_flux_sq") &&
       ok;
  return ok;
}

void cuda_load_flat_source_domain(const FlatSourceDomain& domain)
{
  cuda_initialize_flat_source_domain();
  host_fs_domain->stride_x_ = domain.stride(0);
  host_fs_domain->stride_y_ = domain.stride(1);
  host_fs_domain->stride_z_ = domain.stride(2);

  std::vector<int64_t> n_group(1, domain.ng());
  cuda_malloc_and_copy(&(host_fs_domain->ng_), n_group);
  std::vector<int64_t> n_dim(1, domain.n_dim());
  cuda_malloc_and_copy(&(host_fs_domain->n_dim_), n_dim);
  std::vector<int64_t> n_order(1, domain.n_order());
  cuda_malloc_and_copy(&(host_fs_domain->n_order_), n_order);
  std::vector<int64_t> n_region(1, domain.n_region());
  cuda_malloc_and_copy(&(host_fs_domain->n_region_), n_region);
  std::vector<int64_t> n_source_region(1, domain.n_source_region());
  cuda_malloc_and_copy(&(host_fs_domain->n_source_region_), n_source_region);
  std::vector<int64_t> n_source_region_high_order(
    1, domain.n_source_region_high_order());
  cuda_malloc_and_copy(
    &(host_fs_domain->n_source_region_high_order_), n_source_region_high_order);

  cuda_malloc_and_copy(&(host_fs_domain->materials_), domain.materials());
  cuda_malloc_and_copy_double_as_real(
    &(host_fs_domain->new_scalar_flux_), domain.new_scalar_flux_);
  cuda_malloc_and_copy_double_as_real(
    &(host_fs_domain->old_scalar_flux_), domain.old_scalar_flux_);
  cuda_malloc_and_copy_double_as_real(
    &(host_fs_domain->final_scalar_flux_), domain.final_scalar_flux_);
  cuda_malloc_and_copy_double_as_real(
    &(host_fs_domain->final_scalar_flux_sq_), domain.final_scalar_flux_sq_);
  cuda_malloc_and_copy_double_as_real(
    &(host_fs_domain->source_), domain.source_);
  cuda_malloc_and_copy_double_as_real(
    &(host_fs_domain->external_source_), domain.external_source_);
  // English: Volume estimates and accumulators use geometry precision, which
  // is double precision in mixed mode.
  cuda_malloc_and_copy_double_as(
    &(host_fs_domain->volume_analytic_), domain.volume_analytic_);
  cuda_malloc_and_copy_double_as(
    &(host_fs_domain->volume_estimated_), domain.volume_estimated_);
  cuda_malloc_and_copy_double_as(
    &(host_fs_domain->volume_naive_), domain.volume_naive_);
  cuda_malloc_and_copy_double_as(&(host_fs_domain->volume_), domain.volume_);
  //
  std::vector<char> char_has_external_source_(
    domain.has_external_source_.size());
  for (size_t i = 0; i < domain.has_external_source_.size(); ++i) {
    char_has_external_source_[i] = domain.has_external_source_[i] ? 1 : 0;
  }
  cuda_malloc_and_copy(
    &(host_fs_domain->has_external_source_), char_has_external_source_);
  //
  std::vector<int> n_tracks_converted(
    domain.n_tracks_.begin(), domain.n_tracks_.end());
  cuda_malloc_and_copy(&(host_fs_domain->n_tracks_), n_tracks_converted);
  //
  std::vector<char> char_whether_hit_(domain.whether_hit_.size());
  for (size_t i = 0; i < domain.has_external_source_.size(); ++i) {
    char_whether_hit_[i] = domain.whether_hit_[i] ? 1 : 0;
  }
  cuda_malloc_and_copy(&(host_fs_domain->whether_hit_), char_whether_hit_);
  //
  CuFlatSourceDomain* d_fs_domain;
  cudaMalloc((void**)&d_fs_domain, sizeof(CuFlatSourceDomain));
  cudaMemcpy(d_fs_domain, host_fs_domain, sizeof(CuFlatSourceDomain),
    cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(
    device_fs_domain, &d_fs_domain, sizeof(CuFlatSourceDomain*));
}

// __device__ __forceinline__ int64_t cuda_get_region_id(int64_t i, int64_t j,
// int64_t k)
// {
//   int64_t stride_x = device_fs_domain->stride_x_;
//   int64_t stride_y = device_fs_domain->stride_y_;
//   int64_t stride_z = device_fs_domain->stride_z_;
//   if (d_settings.n_dim == 1) {
//     return i * device_fs_domain->stride_x_;
//   }
//   if (d_settings.n_dim == 2) {
//     return i * device_fs_domain->stride_x_ + j * device_fs_domain->stride_y_;
//   }
//   if (d_settings.n_dim == 3) {
//     return i * device_fs_domain->stride_x_ + j * device_fs_domain->stride_y_
//     +
//            k * device_fs_domain->stride_z_;
//   }
//   return -1;
// }

__global__ void cuda_update_source_kernel(iter_t keff, int run_mode)
{
  int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t stride = gridDim.x * blockDim.x;
  int64_t n_region = d_settings.n_region;
  int64_t n_order = d_settings.n_order;
  int64_t ng = d_settings.n_groups;
  CuFlatSourceDomain* __restrict__ fs = device_fs_domain;
  CudaMatXs* __restrict__ mats = d_mats;

  for (int64_t i = tid; i < n_region; i += stride) {
    if (i > n_region) {
      return;
    }
    int64_t i_mat = fs->materials_[i];

    int64_t i_src = i * ng * n_order * n_order;

    for (int64_t l = 0; l < n_order; l++) {
      for (int64_t m = 0; m < 2 * l + 1; m++) {
        for (int64_t g_out = 0; g_out < ng; ++g_out) {
          real_t scatter_source = 0.0;

          for (int64_t g_in = 0; g_in < ng; ++g_in) {
            real_t flux =
              fs->old_scalar_flux_[i_src + l * l * ng + m * ng + g_in];

            real_t scatter_xs =
              mats->sigma_s[i_mat * ng * ng + g_in * ng + g_out];
            scatter_source += scatter_xs * flux;
          }

          fs->source_[i_src + l * l * ng + m * ng + g_out] =
            scatter_source / mats->sigma_t[i_mat * ng + g_out];
        }
      }
    }

    if (run_mode == 0) {
      if (mats->fissionable[i_mat]) {
        for (int64_t g_out = 0; g_out < ng; g_out++) {
          real_t fission_source = 0.0;
          for (int64_t g_in = 0; g_in < ng; g_in++) {
            fission_source += mats->nu_sigma_f[i_mat * ng + g_in] *
                              fs->old_scalar_flux_[i_src + g_in];
          }

          fission_source *= mats->chi[i_mat * ng + g_out];

          // device_fs_domain->source_[i_src + g_out] +=
          //   fission_source * inverse_keff / d_mats->sigma_t[i_mat * ng +
          //   g_out];
        }
      }
    }
  }
}

__global__ void cuda_update_scatter_source_kernel()
{
  int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t stride = gridDim.x * blockDim.x;
  int64_t n_region = d_settings.n_region;
  int64_t n_order = d_settings.n_order;
  int64_t ng = d_settings.n_groups;
  CuFlatSourceDomain* __restrict__ fs = device_fs_domain;
  CudaMatXs* __restrict__ mats = d_mats;

  int64_t n_llm = n_order * n_order;
  int64_t elements_per_region = n_llm * ng;
  int64_t total_elements = n_region * elements_per_region;

  // Each thread updates one (region, llm, g_out) source moment using sparse
  // scattering rows.
  for (int64_t element_idx = tid; element_idx < total_elements;
       element_idx += stride) {
    int64_t region_idx = element_idx / elements_per_region;
    int64_t local_idx = element_idx % elements_per_region;
    int64_t llm = local_idx / ng;
    int64_t g_out = local_idx % ng;

    // Recover l,m from llm = l*l + m
    int64_t l_val = 0;
    while ((l_val + 1) * (l_val + 1) <= llm) {
      ++l_val;
    }
    int64_t m_val = llm - l_val * l_val;
    if (m_val < 0 || m_val >= 2 * l_val + 1) {
      continue;
    }
    if (l_val >= mats->n_orders) {
      continue;
    }

    int64_t i_mat = fs->materials_[region_idx];
    int64_t i_src_base = region_idx * ng * n_order * n_order;
    int64_t flux_llm_base = i_src_base + llm * ng;

    real_t scatter_source = 0.0;
    for (int64_t g_in = 0; g_in < ng; ++g_in) {
      int64_t row_idx = (i_mat * mats->n_orders + l_val) * ng + g_in;
      int64_t g0 = mats->sigma_s_row_g0[row_idx];
      int64_t g1 = mats->sigma_s_row_g1[row_idx];
      if (g0 < 0 || g1 < g0 || g_out < g0 || g_out > g1) {
        continue;
      }

      int64_t row_offset = mats->sigma_s_row_offset[row_idx];
      int64_t value_idx = row_offset + (g_out - g0);
      real_t scatter_xs = mats->sigma_s_row_values[value_idx];
      real_t flux = fs->old_scalar_flux_[flux_llm_base + g_in];
      scatter_source += scatter_xs * flux;
    }

    fs->source_[flux_llm_base + g_out] =
      scatter_source / mats->sigma_t[i_mat * ng + g_out];
  }
}

__global__ void cuda_update_fission_source_kernel()
{
  int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t stride = gridDim.x * blockDim.x;

  int64_t n_region = d_settings.n_region;
  int64_t n_order = d_settings.n_order;
  int64_t ng = d_settings.n_groups;
  const flux_t inverse_keff =
    static_cast<flux_t>(static_cast<iter_t>(1.0) / cu_simulation.keff_cycle);
  CuFlatSourceDomain* __restrict__ fs = device_fs_domain;
  CudaMatXs* __restrict__ mats = d_mats;

  int64_t total_elements = n_region * ng;

  for (int64_t element_idx = tid; element_idx < total_elements;
       element_idx += stride) {
    int64_t region_idx = element_idx / ng;
    int64_t g_out = element_idx % ng;

    int64_t i_mat = fs->materials_[region_idx];

    if (!mats->fissionable[i_mat])
      continue;

    int64_t elements_per_region = ng * n_order * n_order;
    int64_t i_src_base = region_idx * ng * n_order * n_order;

    real_t fission_source = 0.0;
    for (int64_t g_in = 0; g_in < ng; g_in++) {
      int64_t flux_pos =
        i_src_base + g_in;

      fission_source +=
        mats->nu_sigma_f[i_mat * ng + g_in] * fs->old_scalar_flux_[flux_pos];
    }

    fission_source *= mats->chi[i_mat * ng + g_out];

    fs->source_[i_src_base + g_out] +=
      fission_source * inverse_keff / mats->sigma_t[i_mat * ng + g_out];
  }
}

__global__ void cuda_add_external_source_kernel(int run_mode)
{
  int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t stride = gridDim.x * blockDim.x;
  CuFlatSourceDomain* __restrict__ fs = device_fs_domain;
  int64_t n_region = *fs->n_region_;
  int64_t n_order = *fs->n_order_;
  int64_t ng = *fs->ng_;

  for (int64_t i = tid; i < n_region; i += stride) {
    if (i > n_region) {
      return;
    }
    if (run_mode == 1 && fs->has_external_source_[i]) {
      int64_t i_src = i * ng * n_order * n_order;
      int64_t i_ext = i * ng;
      for (int64_t g = 0; g < ng; g++) {
        fs->source_[i_src + g] += fs->external_source_[i_ext + g];
      }
    }
  }
}

__global__ void cuda_print_source()
{
  real_t* __restrict__ source = device_fs_domain->source_;
  printf("print source\n");
  for (int i = 0; i < d_settings.n_source_region_high_order; i++) {
    printf("%.3f ", source[i]);
  }
  printf("\n");
}

void cuda_update_flat_source_domain(iter_t keff)
{
  if (host_fs_domain == nullptr) {
    fprintf(stderr, "Error: host_fs_domain is not initialized\n");
    return;
  }
  int64_t n_region = simulation::model->n_region();
  int block_size = DEBUG ? 1 : 128;
  int64_t n_order = simulation::model->n_order();
  int64_t ng = simulation::model->ng();
  int64_t scatter_work = n_region * n_order * n_order * ng;
  int grid_size =
    DEBUG ? 1 : static_cast<int>((scatter_work + block_size - 1) / block_size);
  cuda_update_scatter_source_kernel<<<grid_size, block_size>>>();
  if (settings::run_mode == RunMode::EIGENVALUE) {
    grid_size =
      DEBUG ? 1
            : static_cast<int>((n_region * ng + block_size - 1) / block_size);
    cuda_update_fission_source_kernel<<<grid_size, block_size>>>();
  }
  // cuda_print_source<<<1, 1>>>();
  cudaDeviceSynchronize();

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA kernel error in update_source_kernel: %s\n",
      cudaGetErrorString(err));
    return;
  }
}

void cuda_add_external_source_to_flat_source_domain()
{
  if (host_fs_domain == nullptr) {
    fprintf(stderr, "Error: host_fs_domain is not initialized\n");
    return;
  }
  if (settings::run_mode != RunMode::FIXED_SOURCE) {
    return;
  }

  const int64_t n_region = simulation::model->n_region();
  const int block_size = DEBUG ? 1 : 128;
  const int grid_size =
    DEBUG ? 1 : static_cast<int>((n_region + block_size - 1) / block_size);
  cuda_add_external_source_kernel<<<grid_size, block_size>>>(
    static_cast<int>(settings::run_mode));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA kernel error in add_external_source_kernel: %s\n",
      cudaGetErrorString(err));
  }
}

} // namespace gpu
} // namespace theta
#endif
