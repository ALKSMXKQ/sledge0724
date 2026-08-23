#include "trt_runner.h"

#include <NvInferVersion.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>

static_assert(NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR == 6 && NV_TENSORRT_PATCH == 1,
              "This deployment package must be compiled against TensorRT 8.6.1");

namespace {
void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

std::size_t volume(const nvinfer1::Dims& dims) {
  std::size_t result = 1;
  for (int i = 0; i < dims.nbDims; ++i) {
    if (dims.d[i] <= 0) throw std::runtime_error("Engine contains unresolved/non-positive dimensions");
    result *= static_cast<std::size_t>(dims.d[i]);
  }
  return result;
}

std::vector<std::int64_t> to_shape(const nvinfer1::Dims& dims) {
  std::vector<std::int64_t> shape;
  for (int i = 0; i < dims.nbDims; ++i) shape.push_back(dims.d[i]);
  return shape;
}

std::size_t element_bytes(nvinfer1::DataType type) {
  if (type == nvinfer1::DataType::kFLOAT) return sizeof(float);
  if (type == nvinfer1::DataType::kHALF) return sizeof(std::uint16_t);
  throw std::runtime_error("Only float32/float16 TensorRT bindings are supported");
}

float half_to_float(std::uint16_t half) {
  const std::uint32_t sign = static_cast<std::uint32_t>(half & 0x8000U) << 16U;
  const std::uint32_t exponent = (half >> 10U) & 0x1FU;
  std::uint32_t mantissa = half & 0x03FFU;
  std::uint32_t bits = 0;
  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      std::int32_t unbiased_exponent = -14;
      while ((mantissa & 0x0400U) == 0) {
        mantissa <<= 1U;
        --unbiased_exponent;
      }
      mantissa &= 0x03FFU;
      bits = sign | (static_cast<std::uint32_t>(unbiased_exponent + 127) << 23U) | (mantissa << 13U);
    }
  } else if (exponent == 31U) {
    bits = sign | 0x7F800000U | (mantissa << 13U);
  } else {
    bits = sign | ((exponent + 112U) << 23U) | (mantissa << 13U);
  }
  float value = 0.0F;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

std::string dims_string(const nvinfer1::Dims& dims) {
  std::ostringstream out;
  out << '[';
  for (int i = 0; i < dims.nbDims; ++i) {
    if (i) out << ',';
    out << dims.d[i];
  }
  out << ']';
  return out.str();
}

class Logger final : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* message) noexcept override {
    if (severity <= Severity::kWARNING) std::cerr << "[TensorRT] " << message << '\n';
  }
};

template <typename T>
struct TrtDeleter {
  void operator()(T* value) const { delete value; }
};

struct Binding {
  int index{-1};
  std::string name;
  nvinfer1::Dims dims{};
  nvinfer1::DataType type{};
  bool is_input{false};
  std::size_t elements{0};
  std::size_t bytes{0};
  void* device{nullptr};
  void* host{nullptr};
};
}  // namespace

class TrtRunner::Impl {
 public:
  Impl(const std::string& engine_path, const std::string& input_name,
       const std::vector<std::int64_t>& expected_input_shape) {
    std::ifstream stream(engine_path, std::ios::binary | std::ios::ate);
    if (!stream) throw std::runtime_error("Cannot open engine: " + engine_path);
    const auto size = stream.tellg();
    if (size <= 0) throw std::runtime_error("Engine file is empty: " + engine_path);
    stream.seekg(0);
    std::vector<char> serialized(static_cast<std::size_t>(size));
    stream.read(serialized.data(), size);
    if (!stream) throw std::runtime_error("Cannot read engine: " + engine_path);

    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if (!runtime_) throw std::runtime_error("createInferRuntime failed");
    engine_.reset(runtime_->deserializeCudaEngine(serialized.data(), serialized.size()));
    if (!engine_) throw std::runtime_error("deserializeCudaEngine failed; verify TensorRT/GPU compatibility");
    context_.reset(engine_->createExecutionContext());
    if (!context_) throw std::runtime_error("createExecutionContext failed");

    const int count = engine_->getNbBindings();
    if (count < 2) throw std::runtime_error("Expected at least one input and one output binding");
    bindings_.reserve(count);
    device_bindings_.resize(count, nullptr);
    for (int index = 0; index < count; ++index) {
      Binding binding;
      binding.index = index;
      binding.name = engine_->getBindingName(index);
      binding.is_input = engine_->bindingIsInput(index);
      binding.type = engine_->getBindingDataType(index);
      binding.dims = engine_->getBindingDimensions(index);
      if (binding.is_input) {
        if (binding.name != input_name) throw std::runtime_error("Unexpected input binding: " + binding.name);
        nvinfer1::Dims fixed = binding.dims;
        if (fixed.nbDims != static_cast<int>(expected_input_shape.size())) throw std::runtime_error("Input rank mismatch");
        for (int i = 0; i < fixed.nbDims; ++i) fixed.d[i] = static_cast<int>(expected_input_shape[i]);
        if (!context_->setBindingDimensions(index, fixed)) throw std::runtime_error("setBindingDimensions failed");
        binding.dims = fixed;
        input_index_ = index;
      }
      bindings_.push_back(binding);
    }
    if (input_index_ < 0) throw std::runtime_error("Configured input binding not found");
    if (!context_->allInputDimensionsSpecified()) throw std::runtime_error("Not all input dimensions are specified");

    for (auto& binding : bindings_) {
      binding.dims = context_->getBindingDimensions(binding.index);
      binding.elements = volume(binding.dims);
      binding.bytes = binding.elements * element_bytes(binding.type);
      check_cuda(cudaMalloc(&binding.device, binding.bytes), "cudaMalloc binding");
      check_cuda(cudaMallocHost(&binding.host, binding.bytes), "cudaMallocHost binding");
      device_bindings_[binding.index] = binding.device;
      if (binding.is_input && binding.type != nvinfer1::DataType::kFLOAT) {
        throw std::runtime_error("The raster input binding must be float32");
      }
    }
    check_cuda(cudaStreamCreate(&stream_), "cudaStreamCreate");
    for (auto& event : events_) check_cuda(cudaEventCreate(&event), "cudaEventCreate");
  }

  ~Impl() {
    for (auto& event : events_) if (event) cudaEventDestroy(event);
    if (stream_) cudaStreamDestroy(stream_);
    for (auto& binding : bindings_) {
      if (binding.host) cudaFreeHost(binding.host);
      if (binding.device) cudaFree(binding.device);
    }
  }

  InferenceResult infer(const std::vector<float>& input) {
    auto& input_binding = bindings_.at(static_cast<std::size_t>(input_index_));
    if (input.size() != input_binding.elements) throw std::runtime_error("Input element count mismatch");
    std::memcpy(input_binding.host, input.data(), input_binding.bytes);

    check_cuda(cudaEventRecord(events_[0], stream_), "record h2d start");
    check_cuda(cudaMemcpyAsync(input_binding.device, input_binding.host, input_binding.bytes,
                               cudaMemcpyHostToDevice, stream_), "input H2D");
    check_cuda(cudaEventRecord(events_[1], stream_), "record h2d end");
    if (!context_->enqueueV2(device_bindings_.data(), stream_, nullptr)) throw std::runtime_error("enqueueV2 failed");
    check_cuda(cudaEventRecord(events_[2], stream_), "record inference end");
    for (auto& binding : bindings_) {
      if (!binding.is_input) {
        check_cuda(cudaMemcpyAsync(binding.host, binding.device, binding.bytes,
                                   cudaMemcpyDeviceToHost, stream_), "output D2H");
      }
    }
    check_cuda(cudaEventRecord(events_[3], stream_), "record d2h end");
    check_cuda(cudaEventSynchronize(events_[3]), "synchronize inference");

    float h2d = 0.0F, engine = 0.0F, d2h = 0.0F;
    check_cuda(cudaEventElapsedTime(&h2d, events_[0], events_[1]), "measure h2d");
    check_cuda(cudaEventElapsedTime(&engine, events_[1], events_[2]), "measure engine");
    check_cuda(cudaEventElapsedTime(&d2h, events_[2], events_[3]), "measure d2h");
    InferenceResult result;
    result.timings = {h2d, engine, d2h};
    for (const auto& binding : bindings_) {
      if (binding.is_input) continue;
      TensorData tensor;
      tensor.shape = to_shape(binding.dims);
      tensor.values.resize(binding.elements);
      if (binding.type == nvinfer1::DataType::kFLOAT) {
        std::memcpy(tensor.values.data(), binding.host, binding.bytes);
      } else {
        const auto* source = static_cast<const std::uint16_t*>(binding.host);
        std::transform(source, source + binding.elements, tensor.values.begin(), half_to_float);
      }
      result.outputs.emplace(binding.name, std::move(tensor));
    }
    return result;
  }

  std::vector<std::string> summary() const {
    std::vector<std::string> result;
    for (const auto& binding : bindings_) {
      result.push_back(std::string(binding.is_input ? "input " : "output ") + binding.name + " " +
                       dims_string(binding.dims) + (binding.type == nvinfer1::DataType::kFLOAT ? " float32" : " float16"));
    }
    return result;
  }

 private:
  Logger logger_;
  std::unique_ptr<nvinfer1::IRuntime, TrtDeleter<nvinfer1::IRuntime>> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine, TrtDeleter<nvinfer1::ICudaEngine>> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext, TrtDeleter<nvinfer1::IExecutionContext>> context_;
  std::vector<Binding> bindings_;
  std::vector<void*> device_bindings_;
  int input_index_{-1};
  cudaStream_t stream_{nullptr};
  cudaEvent_t events_[4]{nullptr, nullptr, nullptr, nullptr};
};

TrtRunner::TrtRunner(const std::string& engine_path, const std::string& input_name,
                     const std::vector<std::int64_t>& expected_input_shape)
    : impl_(std::make_unique<Impl>(engine_path, input_name, expected_input_shape)) {}
TrtRunner::~TrtRunner() = default;
InferenceResult TrtRunner::infer(const std::vector<float>& input) { return impl_->infer(input); }
std::vector<std::string> TrtRunner::binding_summary() const { return impl_->summary(); }
