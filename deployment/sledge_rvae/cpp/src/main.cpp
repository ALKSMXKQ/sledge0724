#include "config.h"
#include "npy.h"
#include "postprocess.h"
#include "trt_runner.h"

#include <NvInferVersion.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

std::size_t count(const std::vector<std::int64_t>& shape) {
  return std::accumulate(shape.begin(), shape.end(), std::size_t{1},
                         [](std::size_t value, std::int64_t dim) { return value * static_cast<std::size_t>(dim); });
}

std::vector<float> preprocess(const FloatArray& raw, const std::vector<std::int64_t>& expected) {
  if (raw.values.size() != count(expected)) throw std::runtime_error("Input element count does not match contract");
  for (const float value : raw.values) {
    if (!std::isfinite(value)) throw std::runtime_error("Input contains NaN or Inf");
  }
  if (raw.shape == expected) return raw.values;
  const std::vector<std::int64_t> nhwc{1, expected[2], expected[3], expected[1]};
  const std::vector<std::int64_t> hwc{expected[2], expected[3], expected[1]};
  if (raw.shape != nhwc && raw.shape != hwc) throw std::runtime_error("Input must be NCHW, NHWC, or HWC with fixed dimensions");
  const std::size_t channels = static_cast<std::size_t>(expected[1]);
  const std::size_t height = static_cast<std::size_t>(expected[2]);
  const std::size_t width = static_cast<std::size_t>(expected[3]);
  std::vector<float> nchw(raw.values.size());
  for (std::size_t y = 0; y < height; ++y) {
    for (std::size_t x = 0; x < width; ++x) {
      for (std::size_t c = 0; c < channels; ++c) {
        nchw[(c * height + y) * width + x] = raw.values[(y * width + x) * channels + c];
      }
    }
  }
  return nchw;
}

double cuda_used_mib() {
  std::size_t free = 0, total = 0;
  const auto status = cudaMemGetInfo(&free, &total);
  if (status != cudaSuccess) throw std::runtime_error(std::string("cudaMemGetInfo: ") + cudaGetErrorString(status));
  return static_cast<double>(total - free) / (1024.0 * 1024.0);
}

struct Stats {
  double mean, p50, p95, p99, minimum, maximum;
};

Stats stats(std::vector<double> values) {
  if (values.empty()) throw std::runtime_error("No timing values");
  std::sort(values.begin(), values.end());
  const auto percentile = [&](double fraction) {
    const auto index = static_cast<std::size_t>(std::ceil(fraction * values.size())) - 1;
    return values[std::min(index, values.size() - 1)];
  };
  return {std::accumulate(values.begin(), values.end(), 0.0) / values.size(),
          percentile(0.50), percentile(0.95), percentile(0.99), values.front(), values.back()};
}

void write_stats(std::ostream& out, const Stats& value) {
  out << "{\"mean\":" << value.mean << ",\"p50\":" << value.p50 << ",\"p95\":" << value.p95
      << ",\"p99\":" << value.p99 << ",\"min\":" << value.minimum << ",\"max\":" << value.maximum << '}';
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 3 || std::string(argv[1]) != "--config") {
      std::cerr << "Usage: " << argv[0] << " --config deployment/sledge_rvae/configs/runtime.ini\n";
      return 2;
    }
    const RuntimeConfig config = load_config(argv[2]);
    const FloatArray raw = load_npy_f32(config.input_path);
    cudaFree(nullptr);  // Initialize the CUDA context before memory/load measurements.
    const double memory_before = cuda_used_mib();
    const auto load_start = Clock::now();
    TrtRunner runner(config.engine_path, config.input_name, config.expected_input_shape);
    const double engine_load_ms = elapsed_ms(load_start, Clock::now());
    const double memory_after_load = cuda_used_mib();
    double peak_memory = memory_after_load;
    std::cout << "Engine bindings:\n";
    for (const auto& line : runner.binding_summary()) std::cout << "  " << line << '\n';

    for (std::size_t i = 0; i < config.warmup_iterations; ++i) {
      auto input = preprocess(raw, config.expected_input_shape);
      auto result = runner.infer(input);
      (void)postprocess(result.outputs, config.threshold, false);
    }

    std::vector<double> preprocess_ms, h2d_ms, engine_ms, d2h_ms, postprocess_ms, end_to_end_ms;
    preprocess_ms.reserve(config.benchmark_iterations);
    h2d_ms.reserve(config.benchmark_iterations);
    engine_ms.reserve(config.benchmark_iterations);
    d2h_ms.reserve(config.benchmark_iterations);
    postprocess_ms.reserve(config.benchmark_iterations);
    end_to_end_ms.reserve(config.benchmark_iterations);
    InferenceResult final_result;
    for (std::size_t i = 0; i < config.benchmark_iterations; ++i) {
      const auto e2e_start = Clock::now();
      const auto pre_start = Clock::now();
      auto input = preprocess(raw, config.expected_input_shape);
      const auto pre_end = Clock::now();
      final_result = runner.infer(input);
      const auto post_start = Clock::now();
      (void)postprocess(final_result.outputs, config.threshold, false);
      const auto post_end = Clock::now();
      preprocess_ms.push_back(elapsed_ms(pre_start, pre_end));
      h2d_ms.push_back(final_result.timings.h2d_ms);
      engine_ms.push_back(final_result.timings.engine_ms);
      d2h_ms.push_back(final_result.timings.d2h_ms);
      postprocess_ms.push_back(elapsed_ms(post_start, post_end));
      end_to_end_ms.push_back(elapsed_ms(e2e_start, post_end));
      peak_memory = std::max(peak_memory, cuda_used_mib());
    }

    std::filesystem::create_directories(config.output_dir);
    if (config.save_raw_outputs) {
      for (const auto& [name, tensor] : final_result.outputs) {
        save_npy_f32((std::filesystem::path(config.output_dir) / (name + ".npy")).string(), tensor.shape, tensor.values);
      }
    }
    if (config.save_postprocessed) {
      const auto processed = postprocess(final_result.outputs, config.threshold, true);
      std::ofstream output(std::filesystem::path(config.output_dir) / "postprocessed.json");
      output << processed.json;
    }

    const double throughput = 1000.0 / stats(end_to_end_ms).mean;
    const std::filesystem::path metrics_path(config.metrics_path);
    if (!metrics_path.parent_path().empty()) std::filesystem::create_directories(metrics_path.parent_path());
    std::ofstream metrics(metrics_path);
    if (!metrics) throw std::runtime_error("Cannot write metrics: " + config.metrics_path);
    metrics << std::fixed << std::setprecision(6);
    metrics << "{\n  \"software\": {\"tensorrt\": \"" << NV_TENSORRT_MAJOR << '.' << NV_TENSORRT_MINOR << '.'
            << NV_TENSORRT_PATCH << "\", \"cuda_runtime\": \"" << CUDART_VERSION << "\"},\n";
    metrics << "  \"input_shape\": [1,12,256,256],\n  \"precision\": \"" << config.precision << "\",\n";
    metrics << "  \"warmup_iterations\": " << config.warmup_iterations << ",\n  \"benchmark_iterations\": " << config.benchmark_iterations << ",\n";
    metrics << "  \"engine_load_ms\": " << engine_load_ms << ",\n  \"throughput_fps\": " << throughput << ",\n";
    metrics << "  \"cuda_memory_mib\": {\"device_used_before_engine_load\": " << memory_before
            << ", \"device_used_after_engine_load\": " << memory_after_load
            << ", \"device_used_peak_observed\": " << peak_memory
            << ", \"runner_stable_delta\": " << std::max(0.0, memory_after_load - memory_before)
            << ", \"runner_peak_delta\": " << std::max(0.0, peak_memory - memory_before) << "},\n";
    metrics << "  \"latency_ms\": {\n";
    const std::vector<std::pair<std::string, std::vector<double>*>> stages = {
        {"preprocess", &preprocess_ms}, {"h2d", &h2d_ms}, {"engine", &engine_ms},
        {"d2h", &d2h_ms}, {"postprocess", &postprocess_ms}, {"end_to_end", &end_to_end_ms}};
    for (std::size_t i = 0; i < stages.size(); ++i) {
      metrics << "    \"" << stages[i].first << "\": ";
      write_stats(metrics, stats(*stages[i].second));
      metrics << (i + 1 == stages.size() ? "\n" : ",\n");
    }
    metrics << "  }\n}\n";
    std::cout << "Performance metrics: " << config.metrics_path << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "ERROR: " << error.what() << '\n';
    return 1;
  }
}
