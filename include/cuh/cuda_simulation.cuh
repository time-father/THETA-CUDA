#ifndef THETA_CUDA_SIMULATION_H
#define THETA_CUDA_SIMULATION_H
#ifdef THETA_CUDA
#include "cuh/cuda_cartesian_mesh.cuh"
#include "cuh/cuda_flat_source_domain.cuh"
#include "cuh/cuda_functions.cuh"
#include "cuh/cuda_interface.cuh"
#include "cuh/cuda_material.cuh"
#include "cuh/cuda_ray.cuh"
#include "cuh/cuda_settings.cuh"
#include "material.h"
#include "simulation.h"

#include <cuda.h>
#include <memory>
#include <vector>

namespace theta {
namespace gpu {

class CuSimulation {
public:
  // English: Fission rates and keff iteration always use iteration precision,
  // which is double precision in mixed mode.
  iter_t fiss_rate_old;
  iter_t fiss_rate_new;
  iter_t keff_cycle;   //! K-effective for current cycle
  iter_t mean_keff;    //! Mean k-effective over all cycles
  iter_t std_dev_keff; //! Standard deviation of k-effective

  int64_t total_n_tracks; //! Total number of tracks
};

void cuda_load_data();

void run_simulation(iter_t keff, double& mean_keff, double& std_dev_keff);

void cuda_load_simulation();

void cuda_free_simulation();

extern __device__ CuSimulation cu_simulation;
extern CuSimulation host_simulation;

} // namespace gpu
} // namespace theta
#endif
#endif
