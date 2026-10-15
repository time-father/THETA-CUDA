//------------------------------------------------------------------------------
// File: run_info.cpp
// Purpose: Write per-batch and per-run execution records.
//------------------------------------------------------------------------------

#include "run_info.h"

#include <fstream>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "settings.h"
#include "simulation.h"
#include "timer.h"
#include "version.h"

namespace theta {
namespace run_info {

namespace {

std::ofstream batch_info_file;

std::string json_escape(const std::string& value)
{
  std::ostringstream escaped;
  for (char character : value) {
    switch (character) {
    case '"':
      escaped << "\\\"";
      break;
    case '\\':
      escaped << "\\\\";
      break;
    case '\n':
      escaped << "\\n";
      break;
    case '\r':
      escaped << "\\r";
      break;
    case '\t':
      escaped << "\\t";
      break;
    default:
      escaped << character;
      break;
    }
  }
  return escaped.str();
}

const char* transport_name()
{
  return settings::transport_algorithm == TransportAlgorithm::Event
           ? "event-based"
           : "history-based";
}

const char* solver_name()
{
  return settings::solver_type == SolverType::RandomRay ? "random_ray" : "mgmc";
}

const char* run_mode_name()
{
  switch (settings::run_mode) {
  case RunMode::EIGENVALUE:
    return "eigenvalue";
  case RunMode::FIXED_SOURCE:
    return "fixed_source";
  case RunMode::PLOT:
    return "plot";
  }
  return "unknown";
}

void write_timer(std::ofstream& output, const char* key, double value,
  bool trailing_comma)
{
  if (!std::isfinite(value)) {
    value = 0.0;
  }
  output << "      \"" << key << "\": " << std::setprecision(10) << value;
  if (trailing_comma) {
    output << ',';
  }
  output << '\n';
}

double finite_or_zero(double value)
{
  return std::isfinite(value) ? value : 0.0;
}

void write_timing(std::ofstream& output)
{
  const double total = timer::cuda_time_total.elapsed();
  const double initialization = timer::cuda_time_initialize.elapsed();
  const double simulation_time =
    timer::cuda_time_inactive.elapsed() + timer::cuda_time_active.elapsed();
  const double inactive = timer::cuda_time_inactive.elapsed();
  const double active = timer::cuda_time_active.elapsed();
  const double source_update = timer::cuda_time_update_src.elapsed();
  const double random_ray_flux = timer::cuda_time_trrm_flux_treatment.elapsed();
  const double transport = timer::cuda_time_transport.elapsed();
  const double event_track = timer::cuda_time_event_track.elapsed();
  const double event_attenuate = timer::cuda_time_event_attenuate.elapsed();
  const double finalization = timer::cuda_time_finalize.elapsed();

  output << "  \"Timing\": {\n";
  output << "    \"unit\": \"seconds\",\n";
  output << "    \"cuda\": {\n";
  write_timer(output, "total", total, true);
  write_timer(output, "initialization", initialization, true);
  write_timer(output, "simulation", simulation_time, true);
  write_timer(output, "inactive_batches", inactive, true);
  write_timer(output, "active_batches", active, true);
  write_timer(output, "source_update", source_update, true);
  write_timer(output, "random_ray_flux", random_ray_flux, true);
  write_timer(output, "transport", transport, true);
  write_timer(output, "cuda_track_events", event_track, true);
  write_timer(output, "cuda_attenuate_flux_kernel", event_attenuate, true);
  write_timer(output, "finalization", finalization, false);
  output << "    }\n";
  output << "  }\n";
}

} // namespace

void begin_batch_info()
{
  const std::string& batch_info_path = simulation::out_file.batch_info;
  batch_info_file.open(batch_info_path, std::ios::out | std::ios::trunc);
  if (!batch_info_file.is_open()) {
    std::cerr << "Warning: cannot open " << batch_info_path << std::endl;
    return;
  }

  batch_info_file << "batch,event cycle,segments,keff,keff_std\n";
}

void record_batch(int64_t batch, int64_t event_cycles, uint64_t segments,
  double keff, double keff_std)
{
  if (batch_info_file.is_open()) {
    batch_info_file << batch << ',' << event_cycles << ',' << segments << ','
                    << std::setprecision(12) << finite_or_zero(keff) << ','
                    << finite_or_zero(keff_std) << '\n';
    batch_info_file.flush();
  }

  std::cout << " event cycles = " << event_cycles
            << " total segments = " << segments << ' ';
}

void end_batch_info()
{
  if (batch_info_file.is_open()) {
    batch_info_file.close();
  }
}

void output_run_info(double mean_keff, double std_dev_keff)
{
  const std::string& run_info_path = simulation::out_file.run_info;
  std::ofstream output(run_info_path, std::ios::out | std::ios::trunc);
  if (!output.is_open()) {
    std::cerr << "Warning: cannot open " << run_info_path << std::endl;
    return;
  }

  output << std::setprecision(10);
  output << "{\n";
  output << "  \"environment\": {\n";
  output << "    \"commit_id\": \"" << THETA_GIT_COMMIT_ID << "\",\n";
  output << "    \"build_precision\": \"" << THETA_BUILD_PRECISION
         << "\"\n";
  output << "  },\n";
  output << "  \"run\": {\n";
  output << "    \"input_file\": \"" << json_escape(settings::input_name)
         << "\",\n";
  output << "    \"solver\": \"" << solver_name() << "\",\n";
  output << "    \"transport_algorithm\": \"" << transport_name()
         << "\",\n";
  output << "    \"gpu_mode\": true,\n";
  output << "    \"n_batch\": " << settings::n_batch << ",\n";
  output << "    \"inactive_batch\": " << settings::inactive_batch << ",\n";
  output << "    \"n_track\": " << settings::n_track << ",\n";
  output << "    \"energy_groups\": " << settings::num_of_groups << ",\n";
  output << "    \"source_mode\": \"flat\",\n";
  output << "    \"run_mode\": \"" << run_mode_name() << "\",\n";
  output << "    \"event_based\": {\n";
  output << "      \"segments\": " << settings::event_seg_count << ",\n";
  output << "      \"cuda_event_track_block_size\": "
         << settings::cuda_track_block_size << ",\n";
  output << "      \"cuda_event_attenuation_block_size\": "
         << settings::cuda_attenuate_block_size << '\n';
  output << "    },\n";
  output << "    \"keff\": " << finite_or_zero(mean_keff) << ",\n";
  output << "    \"keff_std\": " << finite_or_zero(std_dev_keff) << '\n';
  output << "  },\n";
  output << "  \"output\": {\n";
  output << "    \"flux\": \"" << json_escape(simulation::out_file.flux)
         << "\",\n";
  output << "    \"flux_std\": \""
         << json_escape(simulation::out_file.flux_std) << "\",\n";
  output << "    \"binary\": "
         << (simulation::out_file.binary == "true" ? "true" : "false")
         << ",\n";
  output << "    \"binary_precision\": \""
         << simulation::out_file.bin_precision << "\",\n";
  output << "    \"run_info\": \"" << json_escape(run_info_path)
         << "\",\n";
  output << "    \"batch_info\": \""
         << json_escape(simulation::out_file.batch_info) << "\"\n";
  output << "  },\n";
  write_timing(output);
  output << "}\n";
}

} // namespace run_info
} // namespace theta
