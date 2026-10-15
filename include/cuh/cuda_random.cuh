#ifndef THETA_CUDA_RANDOM_H
#define THETA_CUDA_RANDOM_H

#include <cstdint>
#include "cuh/cuda_functions.cuh"
#include "cuh/cuda_base.cuh"

namespace theta {
namespace gpu {

//==============================================================================
// Module constants.
//==============================================================================

__device__ __constant__ constexpr int N_STREAMS {4}; // Number of random number streams
// Tracking stream for both random ray and mgmc
// Used for random position, direction for tracking (random ray mode)
// Used for generating random flying distance for particles (mgmc mode)
__device__ __constant__ constexpr int STREAM_TRACKING {0};

// Source stream for mgmc
// Used for generating random position, direction and energy group for source
// particles (mgmc mode)
__device__ __constant__ constexpr int STREAM_SOURCE {1};
__device__ __constant__ constexpr int STREAM_URR_PTABLE {2};
__device__ __constant__ constexpr int STREAM_VOLUME {3};
__device__ __constant__ constexpr int64_t DEFAULT_SEED {1};

//==============================================================================
//! Generate a pseudo-random number using a linear congruential generator.
//! @param seed Pseudorandom number seed pointer
//! @return A random number between 0 and 1
//==============================================================================

HDD real_t cu_prn(uint64_t* seed);

//==============================================================================
//! Generate a random number which is 'n' times ahead from the current seed.
//!
//! The result of this function will be the same as the result from calling
//! `prn()` 'n' times, though without the side effect of altering the RNG
//! state.
//! @param n The number of RNG seeds to skip ahead by
//! @param seed Pseudorandom number seed
//! @return A random number between 0 and 1
//==============================================================================

HDD real_t cu_future_prn(int64_t n, uint64_t seed);

//==============================================================================
//! Set a RNG seed to a unique value based on a unique particle ID by striding
//! the seed.
//! @param id The particle ID
//! @param offset The offset from the master seed to be used (e.g., for creating
//! different streams)
//! @return The initialized seed value
//==============================================================================

HDD uint64_t cu_init_seed(int64_t id, int offset);

//==============================================================================
//! Set the RNG seeds to unique values based on the ID of the particle. This
//! function initializes the seeds for all RNG streams of the particle via
//! striding.
//! @param seeds Pseudorandom number seed array
//! @param id The particle ID
//==============================================================================

HDD void cu_init_particle_seeds(int64_t id, uint64_t* seeds);

//==============================================================================
//! Advance the random number seed 'n' times from the current seed. This
//! differs from the future_prn() function in that this function does alter
//! the RNG state.
//! @param seed Pseudorandom number seed pointer
//! @param n The number of RNG seeds to skip ahead by
//==============================================================================

HDD void cu_advance_prn_seed(int64_t n, uint64_t* seed);

//==============================================================================
//! Advance a random number seed 'n' times.
//!
//! This is usually used to skip a fixed number of random numbers (the stride)
//! so that a given particle always has the same starting seed regardless of
//! how many processors are used.
//! @param n The number of RNG seeds to skip ahead by
//! @param seed The starting to seed to advance from
//==============================================================================

HDD uint64_t cu_future_seed(uint64_t n, uint64_t seed);
} // namespace gpu
} // namespace theta

#endif // THETA_RANDOM_LCG_H
