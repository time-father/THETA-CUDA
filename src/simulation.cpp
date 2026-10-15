#include "simulation.h"

namespace theta {
namespace simulation {

//==============================================================================
// Materials information for simulation
//==============================================================================
int64_t n_materials;                              // Number of materials
std::vector<std::unique_ptr<Material>> materials; // Materials information
std::unordered_map<int64_t, int64_t> mat_map;     // Material ID mapping

//==============================================================================
// Geometry model and domain decomposition information for simulation
//==============================================================================
std::unique_ptr<FlatSourceDomain> model; // Geometry model for simulation
std::array<double, 3> lower_left_full;   // Geometry lower left boundary
std::array<double, 3> upper_right_full;  // Geometry upper right boundary


//==============================================================================
// Sources information for simulation
//==============================================================================
std::vector<std::unique_ptr<Source>> sources;

//==============================================================================
// Simulation information for simulation module
//==============================================================================
int64_t current_batch; // Current batch number in the simulation

//==============================================================================
// Output File information for simulation
//==============================================================================
Output out_file;

} // namespace simulation

} // namespace theta
