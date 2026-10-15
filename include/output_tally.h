//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

#ifndef THETA_OUTPUT_TALLY_H
#define THETA_OUTPUT_TALLY_H

#include <string>
#include <vector>

#include "output.h"
#include "random_ray_simulation.h"

namespace theta {

void output_flux_with_data(const RandomRaySimulation& sim, Output out,
  const std::vector<double>& flux_mean,
  const std::vector<double>& flux_variance);

} // namespace theta

#endif // THETA_OUTPUT_TALLY_H
