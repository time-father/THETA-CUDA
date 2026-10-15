#include "cuh/cuda_interface.cuh"
#include "cuh/cuda_material.cuh"
#include "cuh/cuda_base.cuh"
#include "material.h"
#include <algorithm>

#ifdef THETA_CUDA
namespace theta {
namespace gpu {

__device__ CudaMatXs* __restrict__ d_mats = nullptr;
CudaMatXs* h_mats = nullptr;
static CudaMatXs* d_mats_alloc = nullptr;

void cuda_free_materials()
{
  if (h_mats != nullptr) {
    if (d_mats_alloc != nullptr) {
      cudaFree(d_mats_alloc);
      d_mats_alloc = nullptr;
    }
    cudaFree((void*)h_mats->sigma_t);
    cudaFree((void*)h_mats->sigma_s);
    cudaFree((void*)h_mats->sigma_s_lg);
    cudaFree((void*)h_mats->sigma_s_row_g0);
    cudaFree((void*)h_mats->sigma_s_row_g1);
    cudaFree((void*)h_mats->sigma_s_row_offset);
    cudaFree((void*)h_mats->sigma_s_row_values);
    cudaFree((void*)h_mats->nu_sigma_f);
    cudaFree((void*)h_mats->fissionable);
    cudaFree((void*)h_mats->chi);
    delete h_mats;
    h_mats = nullptr;
  }
}

void cuda_load_material(
  const std::vector<std::unique_ptr<theta::Material>>& host_materials)
{
  std::vector<real_t> all_sigma_t;
  std::vector<real_t> all_sigma_s;
  std::vector<real_t> all_sigma_nu_fission;
  std::vector<bool> all_fissionable;
  std::vector<real_t> all_chi;

  size_t n_mat = host_materials.size();
  int64_t ng = 0;
  int64_t n_order = 0;
  if (n_mat > 0) {
    ng = static_cast<int64_t>(host_materials[0]->sigma_t.size());
  }
  for (size_t i = 0; i < n_mat; ++i) {
    n_order = std::max(
      n_order, host_materials[i]->sigma_s_high_order.num_orders());
  }

  std::vector<real_t> all_sigma_s_lg(
    static_cast<size_t>(n_mat) * n_order * ng * ng, 0.0);
  std::vector<int64_t> all_sigma_s_row_g0(
    static_cast<size_t>(n_mat) * n_order * ng, -1);
  std::vector<int64_t> all_sigma_s_row_g1(
    static_cast<size_t>(n_mat) * n_order * ng, -1);
  std::vector<int64_t> all_sigma_s_row_offset(
    static_cast<size_t>(n_mat) * n_order * ng + 1, 0);
  std::vector<real_t> all_sigma_s_row_values;
  all_sigma_s_row_values.reserve(
    static_cast<size_t>(n_mat) * n_order * ng * std::max<int64_t>(1, ng / 4));

  for (size_t i = 0; i < n_mat; ++i) {
    const auto& mat = host_materials[i];
    all_sigma_t.insert(
      all_sigma_t.end(), mat->sigma_t.begin(), mat->sigma_t.end());
    all_sigma_nu_fission.insert(all_sigma_nu_fission.end(),
      mat->nu_sigma_f.begin(), mat->nu_sigma_f.end());
    all_fissionable.push_back(mat->fissionable);
    all_chi.insert(all_chi.end(), mat->chi.begin(), mat->chi.end());

    const int64_t mat_orders = mat->sigma_s_high_order.num_orders();
    for (int64_t l = 0; l < n_order; ++l) {
      for (int64_t g_in = 0; g_in < ng; ++g_in) {
        const size_t row_idx =
          (static_cast<size_t>(i) * n_order + l) * ng + g_in;
        const auto& row =
          (l < mat_orders) ? mat->sigma_s_high_order.at(g_in, l)
                           : mat->sigma_s_high_order.at(g_in, 0);
        int64_t g0 = -1;
        int64_t g1 = -1;
        if (l < mat_orders) {
          g0 = row.target_g0();
          g1 = row.target_g1();
        }
        if (g0 >= 0 && g1 >= g0 && row.data_to() != nullptr) {
          all_sigma_s_row_g0[row_idx] = g0;
          all_sigma_s_row_g1[row_idx] = g1;
          all_sigma_s_row_offset[row_idx] =
            static_cast<int64_t>(all_sigma_s_row_values.size());
          const double* row_data = row.data_to();
          for (int64_t g_out = g0; g_out <= g1; ++g_out) {
            const real_t xs = static_cast<real_t>(row_data[g_out - g0]);
            all_sigma_s_row_values.push_back(xs);
            const size_t dense_idx =
              (((static_cast<size_t>(i) * n_order + l) * ng + g_in) * ng) +
              g_out;
            all_sigma_s_lg[dense_idx] = xs;
            if (l == 0) {
              const size_t p0_idx =
                ((static_cast<size_t>(i) * ng + g_in) * ng) + g_out;
              if (all_sigma_s.size() <= p0_idx) {
                all_sigma_s.resize(static_cast<size_t>(n_mat) * ng * ng, 0.0);
              }
              all_sigma_s[p0_idx] = xs;
            }
          }
          all_sigma_s_row_offset[row_idx + 1] =
            static_cast<int64_t>(all_sigma_s_row_values.size());
        } else {
          all_sigma_s_row_offset[row_idx] =
            static_cast<int64_t>(all_sigma_s_row_values.size());
          all_sigma_s_row_offset[row_idx + 1] =
            static_cast<int64_t>(all_sigma_s_row_values.size());
        }
      }
    }
  }
  if (all_sigma_s.empty()) {
    all_sigma_s.resize(static_cast<size_t>(n_mat) * ng * ng, 0.0);
  }
  if (all_sigma_s_row_values.empty()) {
    all_sigma_s_row_values.push_back(0.0);
  }
  for (size_t row = 1; row < all_sigma_s_row_offset.size(); ++row) {
    if (all_sigma_s_row_offset[row] < all_sigma_s_row_offset[row - 1]) {
      all_sigma_s_row_offset[row] = all_sigma_s_row_offset[row - 1];
    }
  }
  // Convert bool to char
  std::vector<char> all_fissionable_char(all_fissionable.size());
  for (size_t i = 0; i < all_fissionable.size(); ++i) {
    all_fissionable_char[i] = all_fissionable[i] ? 1 : 0;
  }

  real_t *d_sigma_t, *d_sigma_s, *d_sigma_s_lg, *d_sigma_s_row_values,
    *d_sigma_nu_fission, *d_chi;
  int64_t *d_sigma_s_row_g0, *d_sigma_s_row_g1, *d_sigma_s_row_offset;
  char* d_fissionable;
  cuda_malloc_and_copy(&d_sigma_t, all_sigma_t);
  cuda_malloc_and_copy(&d_sigma_s, all_sigma_s);
  cuda_malloc_and_copy(&d_sigma_s_lg, all_sigma_s_lg);
  cuda_malloc_and_copy(&d_sigma_s_row_g0, all_sigma_s_row_g0);
  cuda_malloc_and_copy(&d_sigma_s_row_g1, all_sigma_s_row_g1);
  cuda_malloc_and_copy(&d_sigma_s_row_offset, all_sigma_s_row_offset);
  cuda_malloc_and_copy(&d_sigma_s_row_values, all_sigma_s_row_values);
  cuda_malloc_and_copy(&d_sigma_nu_fission, all_sigma_nu_fission);
  cuda_malloc_and_copy(&d_chi, all_chi);
  cuda_malloc_and_copy(&d_fissionable, all_fissionable_char);

  // Create host-side material cross-section structure
  h_mats = new CudaMatXs {static_cast<int64_t>(n_mat), ng, n_order, d_sigma_t,
    d_sigma_s, d_sigma_s_lg, d_sigma_s_row_g0, d_sigma_s_row_g1,
    d_sigma_s_row_offset, d_sigma_s_row_values, d_sigma_nu_fission, d_chi,
    d_fissionable};
  cudaMalloc((void**)&d_mats_alloc, sizeof(CudaMatXs));
  cudaMemcpy(
    d_mats_alloc, h_mats, sizeof(CudaMatXs), cudaMemcpyHostToDevice);
  // Store device pointer in device symbol
  cudaMemcpyToSymbol(d_mats, &d_mats_alloc, sizeof(CudaMatXs*));
}

} // namespace gpu
} // namespace theta
#endif
