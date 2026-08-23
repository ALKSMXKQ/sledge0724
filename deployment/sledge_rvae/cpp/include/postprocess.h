#pragma once

#include "trt_runner.h"

#include <cstddef>
#include <string>
#include <unordered_map>

struct PostprocessResult {
  std::unordered_map<std::string, std::size_t> active_counts;
  std::string json;
};

PostprocessResult postprocess(const OutputMap& outputs, float threshold, bool include_states);

