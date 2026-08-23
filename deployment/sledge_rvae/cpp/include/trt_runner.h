#pragma once

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

struct TensorData {
  std::vector<std::int64_t> shape;
  std::vector<float> values;
};

using OutputMap = std::unordered_map<std::string, TensorData>;

struct InferenceTimings {
  double h2d_ms{0.0};
  double engine_ms{0.0};
  double d2h_ms{0.0};
};

struct InferenceResult {
  OutputMap outputs;
  InferenceTimings timings;
};

class TrtRunner {
 public:
  TrtRunner(const std::string& engine_path, const std::string& input_name,
            const std::vector<std::int64_t>& expected_input_shape);
  ~TrtRunner();
  TrtRunner(const TrtRunner&) = delete;
  TrtRunner& operator=(const TrtRunner&) = delete;

  InferenceResult infer(const std::vector<float>& input);
  std::vector<std::string> binding_summary() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

