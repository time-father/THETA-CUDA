#ifndef THETA_TIMER_H
#define THETA_TIMER_H

#include <chrono>
#include <string>

namespace theta {

//==============================================================================
// Global variables
//==============================================================================

class Timer;
namespace timer {

extern Timer time_total;
extern Timer time_initialize;
extern Timer time_finalize;
// CUDA timers
extern Timer cuda_time_total;
extern Timer cuda_time_update_src;
extern Timer cuda_time_initialize;
extern Timer cuda_time_transport;
extern Timer cuda_time_event_track;
extern Timer cuda_time_event_attenuate;
extern Timer cuda_time_trrm_flux_treatment;
extern Timer cuda_time_finalize;
extern Timer cuda_time_active;
extern Timer cuda_time_inactive;

} // namespace timer

//==============================================================================
//! Class for measuring time elapsed
//==============================================================================

class Timer {
public:
  using clock = std::chrono::high_resolution_clock;

  Timer() {};

  //! Start running the timer
  void start();

  //! Get total elapsed time in seconds
  //! \return Elapsed time in [s]
  double elapsed();

  //! Stop running the timer
  void stop();

  //! Stop the timer and reset its elapsed time
  void reset();

  //! Add externally measured elapsed seconds to this timer.
  void add(double seconds);

private:
  bool running_ {false};                 //!< is timer running?
  std::chrono::time_point<clock> start_; //!< starting point for clock
  double elapsed_ {0.0};                 //!< elapsed time in [s]
};

//==============================================================================
// Non-member functions
//==============================================================================

void print_cuda_timers();

void output_cuda_timers(std::string filename);

void show_time(const char* label, double secs, int indent_level = 0);

} // namespace theta

#endif // THETA_TIMER_H
