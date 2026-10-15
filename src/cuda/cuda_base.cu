#ifdef THETA_CUDA
#include "cuh/cuda_base.cuh"

#include <iomanip>
#include <iostream>
#include <memory>
#include <stdio.h>

#include "error.h"

namespace theta {
namespace gpu {

int n_devices {0};
int current_device {0};
std::vector<std::unique_ptr<CudaDevice>> devices;

//==============================================================================
// CudaDevice implementation
//==============================================================================

// Empty constructor
CudaDevice::CudaDevice() {}

// Explict constructor by device ID
CudaDevice::CudaDevice(int device_id)
{
  // Set id
  id_ = device_id;

  // Set driver_version
  cudaDriverGetVersion(&driver_version);

  // Set runtime version
  cudaRuntimeGetVersion(&runtime_version);

  // Set properties
  cudaGetDeviceProperties(&prop, id_);

  // Get total number of threads
  total_threads = prop.multiProcessorCount * prop.maxThreadsPerMultiProcessor;

} // CudaDevice constructor

// Check if the CUDA API call is successful
void cuda_execute(cudaError_t code, const char* fname, int line)
{
  if (code != cudaSuccess) {
    fprintf(stderr, "CUDA error at %s:%d: %s\n", fname, line,
      cudaGetErrorString(code));
    exit(EXIT_FAILURE);
  }
}

// Initialize CUDA for every device
void initialize_cuda(int gpu_id)
{
  // Get number of devices
  cugo(cudaGetDeviceCount(&n_devices));
  if (n_devices <= 0) {
    fatal_error("No available CUDA devices.");
  }


  // Reserve memory for devices
  devices.reserve(n_devices);

  // Construct every device
  for (int i = 0; i < n_devices; ++i) {
    devices.push_back(std::make_unique<CudaDevice>(i));
    devices[i]->print_device_info();
  }

  if (gpu_id < 0 || gpu_id >= n_devices) {
    fatal_error("Invalid GPU id " + std::to_string(gpu_id) + ", available "
                "range is [0, " + std::to_string(n_devices - 1) + "].");
  }

  cugo(cudaSetDevice(gpu_id));
  current_device = gpu_id;

  std::cout << "Selected CUDA device " << current_device << ": "
            << devices[current_device]->prop.name << std::endl;
}

void CudaDevice::print_device_info()
{
  cudaDeviceProp prop;
  cugo(cudaGetDeviceProperties(&prop, this->id_));

  std::cout << "\n" << std::string(50, '=') << std::endl;
  std::cout << "CUDA DEVICE INFORMATION" << std::endl;
  std::cout << std::string(50, '=') << std::endl;
  std::cout << "Device " << this->id_ << ": " << prop.name << std::endl;
  std::cout << std::string(50, '-') << std::endl;
  std::cout << std::left << std::setw(30) << "Compute Capability:" << prop.major
            << "." << prop.minor << std::endl;
  std::cout << std::left << std::setw(30) << "Total Global Memory:"
            << (prop.totalGlobalMem / (1024 * 1024 * 1024)) << " GB"
            << std::endl;
  std::cout << std::left << std::setw(30)
            << "Memory Bus Width:" << prop.memoryBusWidth << " bits"
            << std::endl;
  std::cout << std::left << std::setw(30)
            << "Memory Clock Rate:" << prop.memoryClockRate << " KHz"
            << std::endl;
  std::cout << std::left << std::setw(30)
            << "Number of SMs:" << prop.multiProcessorCount << std::endl;
  std::cout << std::left << std::setw(30)
            << "Max Threads Per Block:" << prop.maxThreadsPerBlock << std::endl;
  std::cout << std::left << std::setw(30)
            << "Max Threads Per SM:" << prop.maxThreadsPerMultiProcessor
            << std::endl;
  std::cout << std::left << std::setw(30)
            << "Max registers per SM: " << prop.regsPerMultiprocessor
            << std::endl;
  std::cout << std::left << std::setw(30) << "Warp Size:" << prop.warpSize
            << " threads" << std::endl;
  std::cout << std::left << std::setw(30) << "Max Grid Size:"
            << "(" << prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", "
            << prop.maxGridSize[2] << ")" << std::endl;
  std::cout << std::left << std::setw(30) << "Max Block Size:"
            << "(" << prop.maxThreadsDim[0] << ", " << prop.maxThreadsDim[1]
            << ", " << prop.maxThreadsDim[2] << ")" << std::endl;
  std::cout << std::string(50, '=') << std::endl;
}

} // namespace gpu
} // namespace theta
#endif
