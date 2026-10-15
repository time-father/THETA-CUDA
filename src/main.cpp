
#include <cstdlib>
#include <iostream>

#include "cuh/cuda_simulation.cuh"
#include "error.h"
#include "input.h"
#include "material.h"
#include "output.h"
#include "output_tally.h"
#include "random_ray_simulation.h"
#include "run_info.h"
#include "settings.h"
#include "simulation.h"
#include "timer.h"

namespace theta {
namespace {

void validate_publication_input()
{
  if (settings::hardware_type != HarewareType::GPU) {
    fatal_error("This publication repository supports CUDA GPU inputs only.");
  }
  if (settings::solver_type != SolverType::RandomRay) {
    fatal_error("Only the random_ray solver is supported.");
  }
  if (settings::run_mode == RunMode::PLOT) {
    fatal_error("Plot-only mode is not included in this repository.");
  }
}

} // namespace
} // namespace theta

int main(int argc, char* argv[])
{
  using namespace theta;

  parse_command_line(argc, argv);
  if (settings::input_name.empty()) {
    fatal_error("Specify one input XML file on the command line.");
  }

  timer::time_total.start();
  timer::time_initialize.start();
  theta_title();
  std::cout << "start Reading " << settings::input_name << std::endl;
  read_input(settings::input_name);
  validate_publication_input();
  gpu::initialize_cuda(settings::gpu_id);
  timer::time_initialize.stop();

  RandomRaySimulation worker(simulation::model.get());
  worker.simulation();

  timer::time_total.stop();
  Output gpu_output = simulation::out_file;
  gpu_output.flux = append_flux_suffix(gpu_output.flux, "_gpu");
  gpu_output.flux_std = append_flux_suffix(gpu_output.flux_std, "_gpu");
  if (worker.has_gpu_results_ && settings::enable_flux_output) {
    output_flux_with_data(worker, gpu_output, worker.gpu_final_scalar_flux_,
      worker.gpu_final_scalar_flux_sq_);
  }

  timer::time_finalize.start();
  free_memory_model();
  free_memory_material();
  free_memory_source();
  timer::time_finalize.stop();

  print_cuda_timers();
  output_cuda_timers(simulation::out_file.exce_time);
  run_info::output_run_info(worker.mean_keff, worker.std_dev_keff);
  std::cout << "Finished" << std::endl;
  return EXIT_SUCCESS;
}
