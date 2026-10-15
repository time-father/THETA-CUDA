//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

#include "output_tally.h"

#include <fstream>
#include <string>
#include <vector>

#include "error.h"

namespace theta {
namespace {

void write_value(double value, bool is_last_group, std::ofstream& file,
  const Output& output)
{
  if (output.binary == "true") {
    if (output.bin_precision == "float") {
      const float value_float = static_cast<float>(value);
      file.write(reinterpret_cast<const char*>(&value_float), sizeof(value_float));
    } else {
      file.write(reinterpret_cast<const char*>(&value), sizeof(value));
    }
  } else if (is_last_group) {
    file << value << '\n';
  } else {
    file << value << ',';
  }
}

void write_region_flux(const RandomRaySimulation& sim,
  const std::string& filename, const std::vector<double>& values,
  const Output& output)
{
  std::ofstream file(filename,
    output.binary == "true" ? std::ios::binary : std::ios::out);
  if (!file.is_open()) {
    fatal_error("Cannot open file: " + filename);
  }

  const int64_t n_groups = sim.domain->ng();
  const int64_t n_regions = sim.domain->n_region();
  for (int64_t region = 0; region < n_regions; ++region) {
    const int64_t offset = region * n_groups;
    for (int64_t group = 0; group < n_groups; ++group) {
      write_value(values[offset + group], group == n_groups - 1, file, output);
    }
  }
  if (!file) {
    fatal_error("Failed while writing file: " + filename);
  }
}

} // namespace

void output_flux_with_data(const RandomRaySimulation& sim, Output out,
  const std::vector<double>& flux_mean,
  const std::vector<double>& flux_variance)
{
  write_region_flux(sim, out.flux, flux_mean, out);
  write_region_flux(sim, out.flux_std, flux_variance, out);
}

} // namespace theta
