#ifndef THETA_CUDA_BASE
#define THETA_CUDA_BASE

#ifdef THETA_CUDA
#include "cuda_runtime.h"
#include "stdio.h"
#include <memory>
#include <vector>

#define cugo(ans) gpu::cuda_execute((ans), __FILE__, __LINE__);

#ifndef DEBUG
#define DEBUG false
#endif

// English: CUDA numerical types are separated by physical-data semantics.
// Uniform modes use one scalar type; MIXED keeps transport XS/flux in float.
#if defined(THETA_CUDA_PRECISION_SINGLE)
using geom_t = float;
using flux_t = float;
using iter_t = float;
#elif defined(THETA_CUDA_PRECISION_MIXED)
using geom_t = double;
using flux_t = float;
using iter_t = double;
#else
using geom_t = double;
using flux_t = double;
using iter_t = double;
#endif

// English: real_t is a transitional alias for legacy CUDA transport code and
// is equivalent to the flux type.
using real_t = flux_t;

namespace theta {
namespace gpu {

class CudaDevice {
public:
  int id_;             // Unique ID
  int driver_version;  // Driver version
  int runtime_version; // Runtime version
  cudaDeviceProp prop; // Device properties
  int total_threads;   // Total number of threads

  // Constructor
  CudaDevice();
  explicit CudaDevice(int device_id);

  ~CudaDevice() = default;

  // Print device info
  void print_device_info();
};

extern int n_devices;
extern int current_device;
extern std::vector<std::unique_ptr<CudaDevice>> devices;

// Executor when CUDA basic function is called
void cuda_execute(cudaError_t code, const char* fname, int line);

// Initialize cuda
void initialize_cuda(int gpu_id);

} // namespace gpu
} // namespace theta
#endif // THETA_CUDA
#endif // THETA_CUDA_BASE
