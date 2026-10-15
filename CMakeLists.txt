cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_CURRENT_BINARY_DIR)
  message(FATAL_ERROR ": Building in the source directory is not allowed!")
endif()

set(CMAKE_WARN_DEPRECATED OFF CACHE BOOL "Suppress deprecation warnings from vendored CMake projects." FORCE)

project(theta)

# Capture the source revision during configuration so the executable can report
# its provenance even when it is later moved away from the build directory.
find_package(Git QUIET)
set(THETA_GIT_COMMIT_ID "unknown")
if(Git_FOUND)
  execute_process(
    COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    RESULT_VARIABLE THETA_GIT_RESULT
    OUTPUT_VARIABLE THETA_GIT_COMMIT_ID
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT THETA_GIT_RESULT EQUAL 0 OR THETA_GIT_COMMIT_ID STREQUAL "")
    set(THETA_GIT_COMMIT_ID "unknown")
  endif()
endif()
message(STATUS "THETA Git commit: ${THETA_GIT_COMMIT_ID}")

# CPU-only builds have no GPU transport precision; the CUDA block below
# replaces this value after normalizing DOUBLE/SINGLE/MIXED.
set(THETA_BUILD_PRECISION "CPU")

option(THETA_WITH_CUDA "Enable CUDA support" ON)
if(NOT THETA_WITH_CUDA)
  message(FATAL_ERROR "This repository requires CUDA; CPU-only builds are not supported.")
endif()
# English: Select CUDA numerical precision. DOUBLE and SINGLE are comparison
# baselines; MIXED uses double precision for geometry/ray tracking, volumes,
# and keff while transport flux calculations use single precision.
set(THETA_CUDA_PRECISION "MIXED" CACHE STRING
    "CUDA precision mode used for the manuscript reproduction")
set_property(CACHE THETA_CUDA_PRECISION PROPERTY STRINGS MIXED)
if(NOT THETA_CUDA_PRECISION STREQUAL "MIXED")
  message(FATAL_ERROR "This repository only builds the manuscript Mixed-precision GPU path.")
endif()
option(THETA_CUDA_SEPARABLE_COMPILATION "Enable CUDA separable compilation" ON)
option(THETA_CUDA_PROFILE_INFO "Enable CUDA profiling compile info" OFF)
option(THETA_ENABLE_COMPILER_CACHE "Use ccache or sccache when available" ON)

# Set C++ standard to C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set default build type
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "No build type selected, defaulting to Release")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build" FORCE)
endif()

# Set compile flags based on build type
if(NOT MSVC)
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    list(APPEND cxxflags -g -O0 -Wall -fbounds-check) #-pedantic
  elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    list(APPEND cxxflags -O3)
  else()
    message(FATAL_ERROR "Unrecognized build type = ${CMAKE_BUILD_TYPE}")
  endif()
endif()

message(STATUS "THETA C++ flags: ${cxxflags}")

# English: Enable a compiler cache automatically to reuse repeated builds.
if(THETA_ENABLE_COMPILER_CACHE)
  find_program(THETA_COMPILER_CACHE sccache)
  if(NOT THETA_COMPILER_CACHE)
    find_program(THETA_COMPILER_CACHE ccache)
  endif()

  if(THETA_COMPILER_CACHE)
    message(STATUS "THETA compiler cache: ${THETA_COMPILER_CACHE}")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${THETA_COMPILER_CACHE}")
    set(CMAKE_CUDA_COMPILER_LAUNCHER "${THETA_COMPILER_CACHE}")
  else()
    message(STATUS "THETA compiler cache: not found")
  endif()
endif()

# Find CUDA library (optional)
if(THETA_WITH_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

  # Get OptiX library (optional)
  if(NOT DEFINED ENV{OPTIX_ROOT_DIR})
    message(STATUS "Environment variable OPTIX_ROOT_DIR missed")
  else()
    message(STATUS "OptiX root directory: $ENV{OPTIX_ROOT_DIR}")
    set(OPTIX_ROOT_DIR $ENV{OPTIX_ROOT_DIR})
    find_path(OPTIX_INCLUDE_DIR
      NAMES optix.h
      PATHS ${OPTIX_ROOT_DIR}/include
    )
    if(NOT OPTIX_INCLUDE_DIR)
      message(STATUS "Include directory for OptiX not found!")
    endif()
  endif()

  # English: Detect local CUDA architectures and compile only the unique targets.
  include(FindCUDA/select_compute_arch)

  CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
  string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
  string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
  # English: The fallback list may contain 8.6+PTX. CMAKE_CUDA_ARCHITECTURES
  # accepts numeric architecture IDs only, so remove the PTX suffix before
  # removing decimal points to avoid passing compute_86+PTX to nvcc.
  string(REPLACE "+PTX" "" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_3}")
  string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
  # English: CUDA supports double-precision atomicAdd from sm_60 onward. All
  # precision modes may accumulate double geometry volumes, so sm_53 cannot
  # remain in the fallback architecture list.
  list(REMOVE_ITEM CUDA_ARCH_LIST 53)
  list(REMOVE_DUPLICATES CUDA_ARCH_LIST)

  if(NOT CUDA_ARCH_LIST)
    message(WARNING "No CUDA GPU architecture detected; falling back to native.")
    set(CUDA_ARCH_LIST native)
  endif()

  # English: An explicitly supplied architecture takes precedence over auto
  # detection so release builds can control compile time and binary size for
  # their target cluster. Use the detected fallback list when it is omitted.
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "${CUDA_ARCH_LIST}" CACHE STRING
      "CUDA architectures compiled by nvcc.")
  endif()
  message(STATUS "THETA CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

  # English: Keep separable compilation by default for cross-TU __device__ symbols.
  set(CMAKE_CUDA_SEPARABLE_COMPILATION ${THETA_CUDA_SEPARABLE_COMPILATION})
  message(STATUS "THETA CUDA separable compilation: ${CMAKE_CUDA_SEPARABLE_COMPILATION}")

  # Set compile flags for single-config generators. Multi-config generators
  # such as Visual Studio must use per-config target options below.
  if(NOT CMAKE_CONFIGURATION_TYPES)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
      list(APPEND cudaflags -g -O0)
    elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
      list(APPEND cudaflags -O3)
    else()
      message(FATAL_ERROR "Unrecognized build type = ${CMAKE_BUILD_TYPE}")
    endif()
  else()
    list(REMOVE_ITEM CUDA_NVCC_FLAGS -O3 -O0 -g)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" CACHE STRING "Semi-colon delimit multiple arguments." FORCE)
  endif()

  # Append to nvcc flags
  foreach(flag IN LISTS cudaflags)
    list(FIND CUDA_NVCC_FLAGS ${flag} index)
    if(index EQUAL -1)
      list(APPEND CUDA_NVCC_FLAGS ${flag})
      set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" CACHE STRING "Semi-colon delimit multiple arguments." FORCE)
    endif()
  endforeach(flag)

  message(STATUS "THETA CUDA flags: ${cudaflags}")
endif()

include_directories(./include)
include_directories(./include/xs)
include_directories(./include/cuh)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(./include/utils)

# Add pugixml library
add_subdirectory(pugixml)

# Find source files and create executable
file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS "./src/*.cpp")
# Set CUDA source files
if(THETA_WITH_CUDA)
  file(GLOB_RECURSE CUDA_FILES CONFIGURE_DEPENDS "./src/cuda/*.cu")
  set_source_files_properties(${CUDA_FILES} PROPERTIES LANGUAGE CUDA)
  add_executable(theta ${SRC_FILES} ${CUDA_FILES})
else()
  add_executable(theta ${SRC_FILES})
endif()

target_compile_definitions(theta PRIVATE
  $<$<CONFIG:Debug>:DEBUG=true>
  $<$<NOT:$<CONFIG:Debug>>:DEBUG=false>
)

# Link libraries
if(TARGET pugixml::pugixml)
  target_link_libraries(theta pugixml::pugixml)
else()
  target_link_libraries(theta pugixml)
endif()

if(THETA_WITH_CUDA)
  target_compile_definitions(theta PRIVATE -DTHETA_CUDA)
  target_compile_definitions(theta PRIVATE -DOPENMC_CUDA)

  string(TOUPPER "${THETA_CUDA_PRECISION}" THETA_CUDA_PRECISION_NORMALIZED)
  if(NOT THETA_CUDA_PRECISION_NORMALIZED STREQUAL "MIXED")
    message(FATAL_ERROR "Only CUDA Mixed precision is supported by this repository.")
  endif()
  message(STATUS "THETA CUDA precision mode: ${THETA_CUDA_PRECISION_NORMALIZED}")
  set(THETA_BUILD_PRECISION "${THETA_CUDA_PRECISION_NORMALIZED}")
  target_compile_definitions(theta PRIVATE
    -DTHETA_CUDA_PRECISION_${THETA_CUDA_PRECISION_NORMALIZED})

  target_include_directories(
    theta PRIVATE ${CUDA_INCLUDE_DIRS} ${OPTIX_INCLUDE_DIR})

  target_link_libraries(theta CUDA::cudart)

endif()

# Embed the revision and final numerical precision as string constants instead
# of requiring Git at program runtime.
target_compile_definitions(theta PRIVATE
  THETA_GIT_COMMIT_ID="${THETA_GIT_COMMIT_ID}"
  THETA_BUILD_PRECISION="${THETA_BUILD_PRECISION}")

# Apply compile flags
target_compile_options(theta PRIVATE ${cxxflags})

if(MSVC)
  target_compile_options(theta PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/wd4819>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4819>
  )
endif()

# target_compile_options(theta PRIVATE -Wno-sign-compare)
if(THETA_WITH_CUDA)
  if(NOT CMAKE_CONFIGURATION_TYPES AND CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(theta PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-G>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe=--diag_suppress=177>
    )
    # English: CUDA compiler options used only for Debug builds.
  elseif(NOT CMAKE_CONFIGURATION_TYPES)
    target_compile_options(theta PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:
      -O3 # English: Force maximum optimization.
      -Xcudafe "--diag_suppress=177" # English: Suppress unused variable warnings.
      >
    )

    if(THETA_CUDA_PROFILE_INFO)
      target_compile_options(theta PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
        -Xptxas=-v # English: Print ptxas details such as register usage.
        --generate-line-info # English: Emit line info for Nsight Compute.
        >
      )
    endif()
  endif()
endif()

if(THETA_WITH_CUDA AND CMAKE_CONFIGURATION_TYPES)
  target_compile_options(theta PRIVATE
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-Xcudafe=--diag_suppress=177>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<CONFIG:Debug>>>:-O3>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<CONFIG:Debug>>>:-Xcudafe=--diag_suppress=177>
  )

  if(THETA_CUDA_PROFILE_INFO)
    target_compile_options(theta PRIVATE
      $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<CONFIG:Debug>>>:-Xptxas=-v>
      $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<CONFIG:Debug>>>:--generate-line-info>
    )
  endif()
endif()
