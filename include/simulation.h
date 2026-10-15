#ifndef THETA_SIMULATION_H
#define THETA_SIMULATION_H

#include "flat_source_domain.h"
#include "material.h"
#include "output.h"
#include "source.h"

#include <array>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

/**
 * ==============================================================================
 * @file   simulation.h
 * @brief  Global declarations for the simulation space.
 *
 * This header contains **only** forward declarations of variables and
 * free-functions that are used across multiple translation units of the
 * simulation module.
 * When you add a new global symbol that belongs to `theta::simulation`,
 * please append its declaration here and keep the implementation in
 * `simulation.cpp`.
 *
 * @date    2025-07-15
 * @copyright Copyright (c) 2025 The Theta Authors.
 * ==============================================================================
 */

namespace theta {

namespace simulation {

//==============================================================================
// Error message buffer for simulation module
//==============================================================================
// extern char theta_err_msg[256]; // Buffer for store error message

//==============================================================================
// Materials information for simulation module
//==============================================================================
extern int64_t n_materials; // Number of materials in the simulation
extern std::vector<std::unique_ptr<Material>>
  materials; // Material objects for the simulation
extern std::unordered_map<int64_t, int64_t>
  mat_map; // Map from material.xml id to inner material id
//==============================================================================
// Geometry model and domain decomposition information for simulation module
//==============================================================================
extern std::unique_ptr<FlatSourceDomain> model; // Geometry model for simulation
extern std::array<double, 3> lower_left_full;   // Geometry lower left boundary
extern std::array<double, 3> upper_right_full;  // Geometry right boundary

//==============================================================================
// Sources information for simulation module
//==============================================================================
extern std::vector<std::unique_ptr<Source>>
  sources; // Source objects for the simulation

//==============================================================================
// Simulation information for simulation module
//==============================================================================
extern int64_t current_batch; // Current batch number in the simulation

//==============================================================================
// Output File information for simulation module
//==============================================================================
extern Output out_file; // Output file object for the simulation

} // namespace simulation

} // namespace theta

#endif
