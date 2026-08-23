# TensorRT 8.6.1 C++ compile check

- Result: **PASS**
- TensorRT headers: NVIDIA official `release/8.6`, `NV_TENSORRT_VERSION=8.6.1`
- Header revision: `a0215c1a16c6413c7ac566a498871a4fa36f6f62`
- TensorRT libraries/bindings: `8.6.1`
- CUDA runtime headers/library: pip package `nvidia-cuda-runtime-cu12==12.3.101`
- Build system: CMake `3.29.6`, GNU C++ `11.4.0`
- Compiler mode: C++17, `-Wall -Wextra -Wpedantic -Werror=return-type`
- Compiled sources: `main.cpp`, `config.cpp`, `npy.cpp`, `postprocess.cpp`, `trt_runner.cpp`
- Link result: complete x86-64 ELF produced successfully
- CMake-build binary SHA-256: `500f2114d4cb206052390da3d9b2f7a4d9086bd78675fa258ffe7200b6f031d6`

The binary was not run because this host has no working NVIDIA driver/GPU. This check proves API/header/link compatibility; it does not replace target-GPU inference validation.
