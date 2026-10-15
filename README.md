# Theta GPU Paper Validation Code

This repository is a trimmed snapshot of the THETA CUDA code for validating the accompanying paper. It builds only the CUDA Mixed-precision path. It includes GPU History- and Event-based random-ray transport, mixed-precision computation, the Immortal Ray method, and three benchmarks: IAEA, C5G7, and PCA.

## Scope

- CUDA random-ray solver on Cartesian meshes, using forward calculations, flat sources, and zeroth-order moments.
- Mixed precision: ray geometry, positions, and volume calculations use double precision; transport flux updates use single precision.
- History-based and segment-buffered Event-based transport, selected by the `transport_algorithm` XML field.
- Standard ray and Immortal Ray modes, selected by the `ray_mode` XML field.
- IAEA and C5G7 use History transport; PCA Replica uses Event transport with a fixed source.

## Build Requirements

### Dependencies

- CMake 3.17 or later
- A C++ compiler with C++14 support
- NVIDIA CUDA Toolkit, including `nvcc` and the CUDA Runtime
- An NVIDIA GPU that supports CUDA

By default, CMake detects the CUDA architecture on the build machine. For cross-compilation or a specific target GPU, set `CMAKE_CUDA_ARCHITECTURES` explicitly.

### Verified Build Environment

- Ubuntu Linux
- GCC/G++ 11.4
- CMake 3.22.1
- NVIDIA CUDA Toolkit 12.1
- CUDA architecture 89

### Build

Run these commands from the repository root:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DTHETA_CUDA_PRECISION=MIXED
cmake --build build -j4
```

The executable is created at `build/theta`. Adjust `-j4` to match the number of parallel jobs suitable for your machine.

To specify a CUDA architecture, for example compute capability 8.9:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DTHETA_CUDA_PRECISION=MIXED \
  -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j4
```

The CMake configuration in this repository requires CUDA `MIXED` precision mode; building without CUDA is not supported.

At startup, the program reports the source revision and build precision. Each run also records them in the XML-configured `run_info` file. `run_info.commit_id` identifies the local validation repository revision;

## Repository Layout

```text
.
├── CMakeLists.txt          # CUDA/C++ build configuration
├── examples/
│   ├── gpu/                # Standard ray cases: IAEA, C5G7, and PCA
│   └── immortal/           # Immortal Ray cases for the same benchmarks
├── include/                # C++ and CUDA headers
│   ├── cuh/                # CUDA interfaces and device-side declarations
│   └── xs/                 # Multigroup cross-section data structures
├── pugixml/                # XML parsing dependency
├── src/                    # Host-side C++ implementation
│   ├── cuda/               # CUDA transport and computation
│   ├── utils/              # General utilities
│   └── xs/                 # Cross-section data processing
└── tools/
    └── read_flux.py        # Raw flux reader and CSV converter
```

Runtime output is written to the paths specified in each XML file. The examples use the `output/` directory, which must be created before running a case. Generated output files are ignored by Git.

## Benchmark Inputs

| Input                   | Benchmark   | Transport | Ray mode  |
| ----------------------- | ----------- | --------- | --------- |
| `examples/gpu/iaea.xml` | IAEA        | History   | `default` |
| `examples/gpu/c5g7.xml` | C5G7        | History   | `default` |
| `examples/gpu/pca.xml`  | PCA Replica | Event     | `default` |

The precision cases use the Mixed build, `max_order=0`, `source_mode=flat`, `n_track=64000`, and `active_length=200 cm`. They use 2,500 batches, including 500 inactive batches. The Immortal Ray XML files provide matching cases with `ray_mode=immortal` and `inactive_length=0`. Their sample operating point uses 64,000 rays with an active length of 200 cm, for a total active ray length of `1.28e7 cm` per batch.

For the concurrency scan described in the paper, vary the ray count and inversely adjust the active length to keep `n_track × active_length = 1.28e7 cm`:

| `n_track` | `active_length` (cm) |
| --------: | -------------------: |
|     4,000 |                3,200 |
|     8,000 |                1,600 |
|    16,000 |                  800 |
|    32,000 |                  400 |
|    64,000 |                  200 |
|   128,000 |                  100 |
|   256,000 |                   50 |

Use distinct output filenames for each scan point to retain all results.

## Run the Benchmarks

Run the executable from the repository root so relative XML paths resolve correctly. Create the output directory first. These commands run each case in the background and save its console log:

```bash
mkdir -p output
nohup ./build/theta -g 0 examples/gpu/iaea.xml > output/iaea.log 2>&1 &
```

The `-g` option selects the CUDA device. Run the other standard ray cases with:

```bash
nohup ./build/theta -g 0 examples/gpu/c5g7.xml > output/c5g7.log 2>&1 &
nohup ./build/theta -g 0 examples/gpu/pca.xml > output/pca.log 2>&1 &
```

For example, start an Immortal Ray case with:

```bash
nohup ./build/theta -g 0 examples/immortal/iaea.xml > output/immortal_iaea.log 2>&1 &
```

Use the corresponding `examples/immortal/` XML file for C5G7 or PCA. Inspect a log with `tail -f output/iaea.log`, replacing the filename with the selected case's log. Each XML file configures the number of batches, number of rays, transport algorithm, ray mode, and output filenames. The executable inserts the `_gpu` suffix into the configured flux filename. For example, `output/iaea_flux.bin` is written as `output/iaea_flux_gpu.bin`. Running the standard and Immortal Ray versions of the same benchmark with identical output paths will overwrite the earlier result; change the XML output paths or back up the results to keep both.

## Read Flux Data

The three GPU examples write flux and `flux_std` as double-precision binary files by default. The raw binary files have no header. Values are stored contiguously in region-major order: all energy groups for one mesh region are written before the next region. Within the mesh, the x index varies fastest, followed by y and z.

`tools/read_flux.py` uses the Python 3 standard library to read the case XML, infer the number of energy groups and mesh regions, determine the binary precision and default input path, and convert the data to CSV. By default, it reads the XML `flux` path and writes a CSV file with the same base name:

```bash
python3 tools/read_flux.py examples/gpu/iaea.xml
```

To read the standard deviation file instead:

```bash
python3 tools/read_flux.py examples/gpu/iaea.xml --kind flux_std
```

Input and output paths can also be specified explicitly:

```bash
python3 tools/read_flux.py examples/gpu/pca.xml \
  --input output/pca_flux_gpu.bin \
  --output output/pca_flux_matrix.csv
```

The generated CSV has one mesh region per row and one energy group per column, with no header row. The `flux_std` file contains the sample standard deviation of the flux over active batches. The script checks that the number of values in the file matches the matrix dimensions inferred from the XML. Binary values use the native byte order of the machine that wrote the file; convert the byte order before reading files transferred between machines with different byte orders.

For all supported script options, run:

```bash
python3 tools/read_flux.py --help
```
