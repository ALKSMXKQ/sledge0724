cmake_minimum_required(VERSION 3.18)
project(sledge_rvae_tensorrt LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CUDA_RUNTIME_ROOT "" CACHE PATH "Optional CUDA 12.3 runtime-wheel prefix")
if(CUDA_RUNTIME_ROOT)
  find_path(CUDA_RUNTIME_INCLUDE_DIR cuda_runtime_api.h PATHS ${CUDA_RUNTIME_ROOT}/include NO_DEFAULT_PATH REQUIRED)
  find_file(CUDA_RUNTIME_LIBRARY libcudart.so.12 PATHS ${CUDA_RUNTIME_ROOT}/lib NO_DEFAULT_PATH REQUIRED)
  file(STRINGS "${CUDA_RUNTIME_INCLUDE_DIR}/cuda_runtime_api.h" CUDART_VERSION_LINE
       REGEX "^#define CUDART_VERSION[ ]+[0-9]+")
  string(REGEX MATCH "[0-9]+" CUDART_VERSION_NUMBER "${CUDART_VERSION_LINE}")
  if(NOT CUDART_VERSION_NUMBER EQUAL 12030)
    message(FATAL_ERROR "CUDA runtime 12.3 required (CUDART_VERSION=12030); found ${CUDART_VERSION_NUMBER}")
  endif()
  add_library(sledge_cuda_runtime SHARED IMPORTED)
  set_target_properties(sledge_cuda_runtime PROPERTIES
    IMPORTED_LOCATION "${CUDA_RUNTIME_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${CUDA_RUNTIME_INCLUDE_DIR}")
  set(CUDA_RUNTIME_TARGET sledge_cuda_runtime)
  set(SLEDGE_CUDA_VERSION "12.3 wheel runtime")
else()
  find_package(CUDAToolkit REQUIRED)
  if(CUDAToolkit_VERSION VERSION_LESS "12.3" OR NOT CUDAToolkit_VERSION VERSION_LESS "12.4")
    message(FATAL_ERROR "CUDA 12.3.x is required; found ${CUDAToolkit_VERSION}")
  endif()
  set(CUDA_RUNTIME_TARGET CUDA::cudart)
  set(SLEDGE_CUDA_VERSION "${CUDAToolkit_VERSION}")
endif()

set(TENSORRT_ROOT "" CACHE PATH "TensorRT 8.6.1 installation prefix")
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS ${TENSORRT_ROOT} $ENV{TENSORRT_ROOT}
  PATH_SUFFIXES include include/x86_64-linux-gnu)
find_library(TENSORRT_LIBRARY nvinfer
  HINTS ${TENSORRT_ROOT} $ENV{TENSORRT_ROOT}
  PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu)
if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY)
  message(FATAL_ERROR "TensorRT headers/library not found; set -DTENSORRT_ROOT=/path/to/TensorRT-8.6.1")
endif()

add_executable(sledge_rvae_trt
  src/main.cpp
  src/config.cpp
  src/npy.cpp
  src/postprocess.cpp
  src/trt_runner.cpp)
target_include_directories(sledge_rvae_trt PRIVATE include ${TENSORRT_INCLUDE_DIR})
target_link_libraries(sledge_rvae_trt PRIVATE ${TENSORRT_LIBRARY} ${CUDA_RUNTIME_TARGET})
target_compile_options(sledge_rvae_trt PRIVATE -Wall -Wextra -Wpedantic -Werror=return-type)

message(STATUS "SLEDGE deployment baseline: CUDA ${SLEDGE_CUDA_VERSION}, TensorRT headers ${TENSORRT_INCLUDE_DIR}")

install(TARGETS sledge_rvae_trt RUNTIME DESTINATION bin)
