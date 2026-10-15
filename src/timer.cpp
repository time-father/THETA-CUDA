#include "timer.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include "settings.h"

namespace theta {
namespace timer {

Timer time_total;
Timer time_initialize;
Timer time_finalize;
// CUDA timers
Timer cuda_time_total;
Timer cuda_time_initialize;
Timer cuda_time_update_src;
Timer cuda_time_transport;
Timer cuda_time_event_track;
Timer cuda_time_event_attenuate;
Timer cuda_time_finalize;
Timer cuda_time_trrm_flux_treatment;
Timer cuda_time_active;
Timer cuda_time_inactive;

} // namespace timer

//==============================================================================
// Timer implementation
//==============================================================================

void Timer::start()
{
  running_ = true;
  start_ = clock::now();
}

void Timer::stop()
{
  elapsed_ = elapsed();
  running_ = false;
}

void Timer::reset()
{
  running_ = false;
  elapsed_ = 0.0;
}

void Timer::add(double seconds)
{
  elapsed_ += seconds;
}

double Timer::elapsed()
{
  if (running_) {
    std::chrono::duration<double> diff = clock::now() - start_;
    return elapsed_ + diff.count();
  } else {
    return elapsed_;
  }
}

//==============================================================================
// Non-member functions
//==============================================================================

void print_cuda_timers()
{
  show_time("Total cuda time", timer::cuda_time_total.elapsed());
  show_time("Time for initialization", timer::cuda_time_initialize.elapsed());
  show_time("Time for simulation",
    timer::cuda_time_inactive.elapsed() + timer::cuda_time_active.elapsed());
  if (settings::run_mode == RunMode::EIGENVALUE) {
    show_time("Time for inactive batches", timer::cuda_time_inactive.elapsed());
  }
  show_time("Time for active batches", timer::cuda_time_active.elapsed());
  if (settings::solver_type != SolverType::MGMC) {
    show_time("Time for source update", timer::cuda_time_update_src.elapsed());
  }
  if (settings::solver_type == SolverType::RandomRay) {
    show_time(
      "Time for random-ray flux", timer::cuda_time_trrm_flux_treatment.elapsed());
  }
  show_time("Time for transport", timer::cuda_time_transport.elapsed());
  if (settings::transport_algorithm == TransportAlgorithm::Event) {
    show_time(
      "Time for cuda_track_events", timer::cuda_time_event_track.elapsed());
    show_time(
      "Time for cuda_attenuate_flux_kernel",
      timer::cuda_time_event_attenuate.elapsed());
  }
  show_time("Time for finalization", timer::cuda_time_finalize.elapsed());
}

void output_cuda_timers(std::string filename)
{
  std::ofstream os(filename);
  os << "Total cuda time," << timer::cuda_time_total.elapsed() << "," << std::endl;
  os << "Time for initialization," << timer::cuda_time_initialize.elapsed() << ","
     << std::endl;
  os << "Time for simulation,"
     << timer::cuda_time_inactive.elapsed() + timer::cuda_time_active.elapsed() << ","
     << std::endl;
  if (settings::run_mode == RunMode::EIGENVALUE) {
    os << "Time for inactive batches," << timer::cuda_time_inactive.elapsed() << ","
       << std::endl;
  }
  os << "Time for active batches," << timer::cuda_time_active.elapsed() << ","
     << std::endl;
  if (settings::solver_type != SolverType::MGMC) {
    os << "Time for source update," << timer::cuda_time_update_src.elapsed() << ","
       << std::endl;
  }
  if (settings::solver_type == SolverType::RandomRay) {
    os << "Time for random-ray flux,"
       << timer::cuda_time_trrm_flux_treatment.elapsed() << "," << std::endl;
  }
  os << "Time for transport," << timer::cuda_time_transport.elapsed() << ","
     << std::endl;
  if (settings::transport_algorithm == TransportAlgorithm::Event) {
    os << "Time for cuda_track_events,"
       << timer::cuda_time_event_track.elapsed() << "," << std::endl;
    os << "Time for cuda_attenuate_flux_kernel,"
       << timer::cuda_time_event_attenuate.elapsed() << "," << std::endl;
  }
  os << "Time for finalization," << timer::cuda_time_finalize.elapsed() << ","
     << std::endl;
  os.close();
}

void show_time(const char* label, double secs, int indent_level)
{
  int width = 33 - indent_level * 2;

  std::cout << std::string(2 * indent_level, ' ') << std::left
            << std::setw(width) << label << " = " << std::right
            << std::setw(10) << std::scientific << std::setprecision(4)
            << secs << " seconds" << std::endl;
}

} // namespace theta
