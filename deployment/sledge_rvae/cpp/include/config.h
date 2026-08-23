#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct RuntimeConfig {
  std::string engine_path;
  std::string input_path;
  std::string output_dir;
  std::string metrics_path;
  std::string input_name{"raster"};
  std::string precision{"fp32"};
  std::vector<std::int64_t> expected_input_shape{1, 12, 256, 256};
  float threshold{0.3F};
  std::size_t warmup_iterations{20};
  std::size_t benchmark_iterations{100};
  bool save_raw_outputs{true};
  bool save_postprocessed{true};
};

RuntimeConfig load_config(const std::string& path);
