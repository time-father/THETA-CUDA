#ifndef THETA_RUN_INFO_H
#define THETA_RUN_INFO_H

#include <cstdint>

namespace theta {
namespace run_info {

void begin_batch_info();

void record_batch(int64_t batch, int64_t event_cycles, uint64_t segments,
  double keff, double keff_std);

void end_batch_info();

void output_run_info(double mean_keff, double std_dev_keff);

} // namespace run_info
} // namespace theta

#endif // THETA_RUN_INFO_H
