#ifndef THETA_RANDOM_RAY_SIMULATION_H
#define THETA_RANDOM_RAY_SIMULATION_H


#include <vector>

#include "flat_source_domain.h"

namespace theta {

class RandomRaySimulation {
public:
  explicit RandomRaySimulation(FlatSourceDomain* domain);

  void simulation();

  FlatSourceDomain* domain {nullptr};
  double keff_cycle {1.0};
  double mean_keff {0.0};
  double std_dev_keff {0.0};
  std::vector<double> gpu_final_scalar_flux_;
  std::vector<double> gpu_final_scalar_flux_sq_;
  bool has_gpu_results_ {false};
};

} // namespace theta

#endif // THETA_RANDOM_RAY_SIMULATION_H
