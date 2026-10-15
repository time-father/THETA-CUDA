
#include "random_ray_simulation.h"

#include "cuh/cuda_simulation.cuh"
#include "run_info.h"
#include "settings.h"
#include "simulation.h"

namespace theta {

RandomRaySimulation::RandomRaySimulation(FlatSourceDomain* domain)
  : domain(domain)
{}

void RandomRaySimulation::simulation()
{
  if (settings::run_mode == RunMode::FIXED_SOURCE) {
    domain->set_fixed_source();
  }

  run_info::begin_batch_info();
  gpu::run_simulation(keff_cycle, mean_keff, std_dev_keff);

  gpu_final_scalar_flux_.assign(domain->n_source_region_high_order(), 0.0);
  gpu_final_scalar_flux_sq_.assign(domain->n_source_region_high_order(), 0.0);
  has_gpu_results_ = gpu::cuda_download_data(
    gpu_final_scalar_flux_, gpu_final_scalar_flux_sq_);
  run_info::end_batch_info();
}

} // namespace theta
