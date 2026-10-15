#ifndef THETA_SETTINGS_H
#define THETA_SETTINGS_H

#include <cstdint>
#include <iostream>

#include "pugixml.hpp"

/**
 * ==============================================================================
 * @file   settings.h
 *         Declares shared configuration items and the reader interface for the
 *         global settings namespace.
 *
 * This header only stores global variables in `theta::settings` and the
 * `read_settings` declaration used across translation units; new settings
 * should be declared here and defined/parsed in `settings.cpp`.
 *
 * @date    2025-07-15
 * @copyright Copyright (c) 2025 The Theta Authors.
 * ==============================================================================
 */

namespace theta {

enum class HarewareType { GPU };

enum class IterationType { GAUSS_SEIDEL, JACOBI };

enum class RunMode { EIGENVALUE, FIXED_SOURCE, PLOT };

enum class SolverType { RandomRay, MGMC };

enum class RayMode { DEFAULT, IMMORTAL };
enum class TransportAlgorithm { Event, History };

enum class VolumeType { Adaptive, Analytic, BatchEstimate, AverageEstimate };

namespace settings {

//==============================================================================
// Hareware settings information for settings module
//==============================================================================
extern HarewareType hardware_type;

//==============================================================================
// Input file information for settings module
//==============================================================================
extern std::string input_name; // Input file name
extern int gpu_id;             // CUDA GPU id selected from command line

//==============================================================================
// Energy groups information for settings module
//==============================================================================
extern int64_t num_of_groups; // Number of energy groups in claculation

//==============================================================================
// The random characteristic ray information for settings module
//==============================================================================
extern int64_t n_batch;          // Number of total batches in calculation
extern int64_t inactive_batch;   // Number of inactive batches in calculation
extern int64_t n_track;          // Number of random ray tracks in each batch
extern double active_length;     // Number of active length of each ray
extern double inactive_length;   // Number of inactive length of each ray
extern IterationType
  iteration_type; // Type of iteration method used in calculation
extern RunMode
  run_mode; // Type of run mode used in calculation (eigenvalue or fixed-source)
extern RayMode ray_mode; // Introducing Immortal Ray
extern TransportAlgorithm
  transport_algorithm; // GPU transport algorithm, event or history based
extern int64_t event_seg_count; // Number of segments recorded per event pass
extern int cuda_track_block_size;
extern int cuda_attenuate_block_size;
extern SolverType
  solver_type;       // Type of solver used in calculation (random_ray or mgmc)
extern VolumeType volume_type; // Type of volume calculation method
extern bool shannon_entropy;   // Flag indicating if shannon entropy is computed
extern bool enable_ray_scan_output; // Flag indicating if ray scan segment
                                    // output is dumped
extern bool enable_flux_output;
                                // whether flux/flux_std files are written

//==============================================================================
// MGMC weight information for settings module
//==============================================================================
extern bool survival_biasing;    // Flag indicating if survival biasing is used
extern double weight_cutoff;     // Weight cutoff for survival biasing
extern double weight_survive;    // Weight survive for survival biasing
extern double ray_resample_rate; // For Immortal Ray, set the proportion of
                                 // resampling required for each generation

} // namespace settings

void read_settings(pugi::xml_node node);

} // namespace theta

#endif // THETA_SETTINGS_H
