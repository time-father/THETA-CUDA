#include "settings.h"

#include <string>

#include "error.h"
#include "xml_interface.h"

namespace theta {

namespace {

// Check whether an event-tracking block size belongs to the planned
// warp-aligned candidate set.
//
// Parameter:
//          Candidate thread count read from XML.
//
// Return:
//   True when the candidate is supported; false otherwise.
bool is_valid_cuda_track_block_size(int value)
{
  return value == 32 || value == 64 || value == 96 || value == 128 ||
         value == 192 || value == 256;
}

// Check whether an event-attenuation block size belongs to the planned
// warp-aligned candidate set.
//
// Parameter:
//          Candidate thread count read from XML.
//
// Return:
//   True when the candidate is supported; false otherwise.
bool is_valid_cuda_attenuate_block_size(int value)
{
  return value == 64 || value == 128 || value == 256;
}

} // namespace

namespace settings {

//==============================================================================
// Hareware settings information for settings module
//==============================================================================
HarewareType hardware_type;

//==============================================================================
// Input file information for settings module
//==============================================================================
std::string input_name;
int gpu_id {0};

//==============================================================================
// Energy groups information for settings module
//==============================================================================
int64_t num_of_groups;

//==============================================================================
// The random characteristic ray information for settings module
//==============================================================================
int64_t n_batch;
int64_t inactive_batch;
int64_t n_track;
double active_length;
double inactive_length;
IterationType iteration_type;
RunMode run_mode;
RayMode ray_mode;
TransportAlgorithm transport_algorithm {TransportAlgorithm::Event};
int64_t event_seg_count {8};
int cuda_track_block_size {128};
int cuda_attenuate_block_size {128};
SolverType solver_type;
VolumeType volume_type;
bool shannon_entropy {true};
bool enable_ray_scan_output {false};
bool enable_flux_output {true};

//==============================================================================
// MGMC weight information for settings module
//==============================================================================
bool survival_biasing {true};
double weight_cutoff {0.25};
double weight_survive {1.0};
double ray_resample_rate {0.0};

} // namespace settings

void read_settings(pugi::xml_node node)
{
  // Read solver type
  if (check_for_node(node, "solver_type")) {
    if (get_node_value(node, "solver_type", true, true) == "random_ray") {
      settings::solver_type = SolverType::RandomRay;
    } else if (get_node_value(node, "solver_type", true, true) == "mgmc") {
      settings::solver_type = SolverType::MGMC;
    } else {
      fatal_error("Error ```solver_type``` node!");
    }
  } else {
    settings::solver_type = SolverType::RandomRay;
  }

  // Run mode
  if (settings::run_mode != RunMode::PLOT) {
    if (check_for_node(node, "run_mode")) {
      if (get_node_value(node, "run_mode", true, true) == "eigenvalue") {
        settings::run_mode = RunMode::EIGENVALUE;
      } else if (get_node_value(node, "run_mode", true, true) ==
                 "fixed_source") {
        settings::run_mode = RunMode::FIXED_SOURCE;
      } else if (get_node_value(node, "run_mode", true, true) ==
                 "fixed-source") {
        settings::run_mode = RunMode::FIXED_SOURCE;
      } else if (get_node_value(node, "run_mode", true, true) == "plot") {
        settings::run_mode = RunMode::PLOT;
      } else {
        fatal_error("Error ```run_mode``` node!");
      }
    } else {
      fatal_error("```run_mode``` node does not exist in input.xml!");
    }
  }

  if (check_for_node(node, "adjoint")) {
    if (get_node_value(node, "adjoint", true, true) == "true") {
      fatal_error("Adjoint calculations are not included in this repository.");
    }
  }

  // Read active length
  if (check_for_node(node, "active_length")) {
    settings::active_length = std::stod(get_node_value(node, "active_length"));
  } else {
    if (settings::solver_type == SolverType::RandomRay)
      fatal_error("```active_length``` node does not exist in input.xml!");
  }

  // Read inactive length
  if (check_for_node(node, "inactive_length")) {
    settings::inactive_length =
      std::stod(get_node_value(node, "inactive_length"));
  } else {
    if (settings::solver_type == SolverType::RandomRay) {
      fatal_error("```inactive_length``` node does not exist in input.xml!");
    }
  }

  if (check_for_node(node, "max_order")) {
    if (std::stod(get_node_value(node, "max_order")) != 0.0) {
      fatal_error("Only zero-order scattering is included in this repository.");
    }
  }

  // number of batches
  if (check_for_node(node, "n_batch")) {
    settings::n_batch = std::stoi(get_node_value(node, "n_batch"));
  } else {
    fatal_error("```n_batch``` node does not exist in input.xml!");
  }

  // number of inactive batches
  settings::inactive_batch = 0;
  if (check_for_node(node, "inactive_batch")) {
    settings::inactive_batch =
      std::stoi(get_node_value(node, "inactive_batch"));
    if (settings::run_mode == RunMode::FIXED_SOURCE &&
        settings::solver_type == SolverType::MGMC) {
      settings::inactive_batch = 0;
    }
  } else if (settings::run_mode == RunMode::EIGENVALUE) {
    fatal_error("```inactive_batch``` node does not exist in input.xml!");
  }

  // number of tracks
  if (check_for_node(node, "n_track")) {
    settings::n_track = std::stoi(get_node_value(node, "n_track"));
  } else {
    fatal_error("```n_track``` node does not exist in input.xml!");
  }

  // iteration type
  if (check_for_node(node, "iteration_type")) {
    if (get_node_value(node, "iteration_type", true, true) == "gauss_seidel") {
      settings::iteration_type = IterationType::GAUSS_SEIDEL;
    } else if (get_node_value(node, "iteration_type", true, true) == "jacobi") {
      settings::iteration_type = IterationType::JACOBI;
    } else {
      fatal_error("Error ```iteration_type``` node!");
    }
  } else {
    settings::iteration_type = IterationType::JACOBI;
  }

  if (check_for_node(node, "source_mode")) {
    if (get_node_value(node, "source_mode", true, true) != "flat") {
      fatal_error("Only flat-source inputs are included in this repository.");
    }
  }

  // Reject uncollided-source options because their transport path was removed.
  if (check_for_node(node, "use_uncollided_source")) {
    const std::string value =
      get_node_value(node, "use_uncollided_source", true, true);
    if (value == "true" || value == "yes") {
      fatal_error("Uncollided-source mode is not included in this repository.");
    }
    if (value != "false" && value != "no") {
      fatal_error("Error ```use_uncollided_source``` node!");
    }
  }

  if (check_for_node(node, "volume_type")) {
    if (get_node_value(node, "volume_type", true, true) == "adaptive") {
      settings::volume_type = VolumeType::Adaptive;
    } else if (get_node_value(node, "volume_type", true, true) == "analytic") {
      settings::volume_type = VolumeType::Analytic;
    } else if (get_node_value(node, "volume_type", true, true) ==
               "batch_estimate") {
      settings::volume_type = VolumeType::BatchEstimate;
    } else if (get_node_value(node, "volume_type", true, true) ==
               "average_estimate") {
      settings::volume_type = VolumeType::AverageEstimate;
    } else {
      fatal_error("Error ```volume_type``` node! Only `adaptive`, `analytic`, "
                  "`batch_estimate`, or `average_estimate` are supported!");
    }
  } else {
    settings::volume_type = VolumeType::Analytic;
  }

  if (check_for_node(node, "enable_ray_scan_output")) {
    const std::string value =
      get_node_value(node, "enable_ray_scan_output", true, true);
    if (value == "true" || value == "yes") {
      settings::enable_ray_scan_output = true;
    } else if (value == "false" || value == "no") {
      settings::enable_ray_scan_output = false;
    } else {
      fatal_error("Error ```enable_ray_scan_output``` node!");
    }
  } else {
    settings::enable_ray_scan_output = false;
  }

  // written. The default keeps the historical behavior.
  if (check_for_node(node, "enable_flux_output")) {
    const std::string value =
      get_node_value(node, "enable_flux_output", true, true);
    if (value == "true" || value == "yes") {
      settings::enable_flux_output = true;
    } else if (value == "false" || value == "no") {
      settings::enable_flux_output = false;
    } else {
      fatal_error("Error ```enable_flux_output``` node!");
    }
  } else {
    settings::enable_flux_output = true;
  }

  if (check_for_node(node, "shannon_entropy")) {
    if (get_node_value(node, "shannon_entropy", true, true) == "true") {
      settings::shannon_entropy = true;
    } else if (get_node_value(node, "shannon_entropy", true, true) == "false") {
      settings::shannon_entropy = false;
    } else {
      fatal_error("Error ```shannon_entropy``` node! Only `true` or `false` "
                  "are supported!");
    }
  } else {
    settings::shannon_entropy = true;
  }

  // Ray mode
  if (check_for_node(node, "ray_mode")) {
    if (get_node_value(node, "ray_mode", true, true) == "default") {
      settings::ray_mode = RayMode::DEFAULT;
    } else if (get_node_value(node, "ray_mode", true, true) == "immortal") {
      settings::ray_mode = RayMode::IMMORTAL;
    } else {
      settings::ray_mode = RayMode::DEFAULT;
    }
  } else {
    settings::ray_mode = RayMode::DEFAULT;
  }

  if (check_for_node(node, "transport_algorithm")) {
    const std::string algorithm =
      get_node_value(node, "transport_algorithm", true, true);
    if (algorithm == "event") {
      settings::transport_algorithm = TransportAlgorithm::Event;
      std::cout << "Event-based transport algorithm is selected." << std::endl;
    } else if (algorithm == "history") {
      settings::transport_algorithm = TransportAlgorithm::History;
      std::cout << "History-based transport algorithm is selected."
                << std::endl;
    } else {
      fatal_error("Error ```transport_algorithm``` node! Only `event` or "
                  "`history` are supported.");
    }
  } else {
    settings::transport_algorithm = TransportAlgorithm::Event;
    std::cout << "Event-based transport algorithm is selected." << std::endl;
  }

  if (check_for_node(node, "event_seg_count")) {
    settings::event_seg_count =
      std::stoll(get_node_value(node, "event_seg_count"));
    if (settings::event_seg_count <= 0) {
      fatal_error("Error ```event_seg_count``` node! It must be greater than "
                  "0.");
    }
    std::cout << "Event-based with " << settings::event_seg_count
              << " segments." << std::endl;
  } else {
    settings::event_seg_count = 8;
    std::cout << "Event-based with " << settings::event_seg_count
              << " segments." << std::endl;
  }

  // Read tracking and attenuation block sizes independently because the kernels
  // have different register pressure.
  if (check_for_node(node, "cuda_track_block_size")) {
    settings::cuda_track_block_size =
      std::stoi(get_node_value(node, "cuda_track_block_size"));
  } else {
    settings::cuda_track_block_size = 128;
  }
  if (!is_valid_cuda_track_block_size(settings::cuda_track_block_size)) {
    fatal_error("Error ```cuda_track_block_size``` node! Supported values are "
                "32, 64, 96, 128, 192, and 256.");
  }

  if (check_for_node(node, "cuda_attenuate_block_size")) {
    settings::cuda_attenuate_block_size =
      std::stoi(get_node_value(node, "cuda_attenuate_block_size"));
  } else {
    settings::cuda_attenuate_block_size = 128;
  }
  if (!is_valid_cuda_attenuate_block_size(
        settings::cuda_attenuate_block_size)) {
    fatal_error(
      "Error ```cuda_attenuate_block_size``` node! Supported values are 64, "
      "128, and 256.");
  }

  std::cout << "CUDA event track block size: "
            << settings::cuda_track_block_size << std::endl;
  std::cout << "CUDA event attenuation block size: "
            << settings::cuda_attenuate_block_size << std::endl;

  // Discard rate
  if (check_for_node(node, "ray_resample_rate")) {
    settings::ray_resample_rate =
      std::stod(get_node_value(node, "ray_resample_rate"));
  } else {
    settings::ray_resample_rate = 0.0;
  }

  if (check_for_node(node, "hardware")) {
    if (get_node_value(node, "hardware", true, true) == "gpu") {
#ifdef THETA_CUDA
      settings::hardware_type = HarewareType::GPU;
      std::cout << "GPU mode is selected." << std::endl;
#else
      fatal_error(
        "Error ```hardware``` node! `gpu` requires a CUDA-enabled build.");
#endif
    } else {
      fatal_error("Error ```hardware``` node! Only `gpu` is supported.");
    }
  } else {
    settings::hardware_type = HarewareType::GPU;
  }
}
} // namespace theta
