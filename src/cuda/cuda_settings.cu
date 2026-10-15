#include "cuh/cuda_settings.cuh"
#include "settings.h"
#include "simulation.h"
#include <cuda_runtime.h>

#ifdef THETA_CUDA
namespace theta {
namespace gpu {

__constant__ DeviceSettings d_settings;

void cuda_load_settings()
{
  DeviceSettings host_settings;
  host_settings.n_track = settings::n_track;
  host_settings.n_batch = settings::n_batch;
  host_settings.inactive_batch = settings::inactive_batch;
  host_settings.active_length = settings::active_length;
  host_settings.inactive_length = settings::inactive_length;
  host_settings.event_seg_count = settings::event_seg_count;

  host_settings.n_groups = simulation::model->ng();
  host_settings.n_order = simulation::model->n_order();
  host_settings.n_dim = simulation::model->n_dim();
  host_settings.n_region = simulation::model->n_region();
  host_settings.n_source_region =
    simulation::model->n_source_region();
  host_settings.n_source_region_high_order =
    simulation::model->n_source_region_high_order();

  host_settings.volume_type = static_cast<size_t>(settings::volume_type);

  host_settings.x0_boundary =
    static_cast<size_t>(simulation::model->geometry(0)->lower_boundary_type());
  host_settings.x1_boundary =
    static_cast<size_t>(simulation::model->geometry(0)->upper_boundary_type());
  if (simulation::model->n_dim() >= 2) {
    host_settings.y0_boundary = static_cast<size_t>(
      simulation::model->geometry(1)->lower_boundary_type());
    host_settings.y1_boundary = static_cast<size_t>(
      simulation::model->geometry(1)->upper_boundary_type());
  } else {
    host_settings.y0_boundary = 1;
    host_settings.y1_boundary = 1;
  }

  if (simulation::model->n_dim() >= 3) {
    host_settings.z0_boundary = static_cast<size_t>(
      simulation::model->geometry(2)->lower_boundary_type());
    host_settings.z1_boundary = static_cast<size_t>(
      simulation::model->geometry(2)->upper_boundary_type());
  } else {
    host_settings.z0_boundary = 1;
    host_settings.z1_boundary = 1;
  }

  cudaMemcpyToSymbol(d_settings, &host_settings, sizeof(DeviceSettings));
}

} // namespace gpu
} // namespace theta
#endif
