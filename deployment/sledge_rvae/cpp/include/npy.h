#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct FloatArray {
  std::vector<std::int64_t> shape;
  std::vector<float> values;
};

FloatArray load_npy_f32(const std::string& path);
void save_npy_f32(const std::string& path, const std::vector<std::int64_t>& shape,
                  const std::vector<float>& values);

