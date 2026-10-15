#ifndef CUDA_SETTINGS_H
#define CUDA_SETTINGS_H
#ifdef THETA_CUDA
#include "cuh/cuda_base.cuh"
#include "settings.h"

namespace theta {
namespace gpu {

struct DeviceSettings {
  geom_t active_length;
  geom_t inactive_length;
  int64_t n_batch;
  int64_t inactive_batch;
  int64_t n_track;
  int64_t event_seg_count;
  int n_groups;
  int n_order;
  int n_dim;
  int n_region;
  int n_source_region;
  int n_source_region_high_order;

  size_t volume_type;
  size_t x0_boundary;
  size_t x1_boundary;
  size_t y0_boundary;
  size_t y1_boundary;
  size_t z0_boundary;
  size_t z1_boundary;
};

extern __constant__ DeviceSettings d_settings;

void cuda_load_settings();

} // namespace gpu
} // namespace theta
#endif
#endif
