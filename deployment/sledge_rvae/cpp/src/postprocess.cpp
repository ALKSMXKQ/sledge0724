#include "postprocess.h"

#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {
float sigmoid(float value) {
  if (value >= 0.0F) return 1.0F / (1.0F + std::exp(-value));
  const float exponential = std::exp(value);
  return exponential / (1.0F + exponential);
}

const TensorData& require(const OutputMap& outputs, const std::string& name) {
  const auto it = outputs.find(name);
  if (it == outputs.end()) throw std::runtime_error("Missing engine output: " + name);
  return it->second;
}
}  // namespace

PostprocessResult postprocess(const OutputMap& outputs, float threshold, bool include_states) {
  static const std::vector<std::string> prefixes = {
      "lines", "vehicles", "pedestrians", "static_objects", "green_lights", "red_lights"};
  PostprocessResult result;
  std::ostringstream json;
  json << std::setprecision(9) << "{\n  \"threshold\": " << threshold << ",\n  \"elements\": {\n";
  for (std::size_t category = 0; category < prefixes.size(); ++category) {
    const auto& prefix = prefixes[category];
    const auto& logits = require(outputs, prefix + "_logits");
    const auto& states = require(outputs, prefix + "_states");
    if (logits.values.empty() || states.values.size() % logits.values.size() != 0) {
      throw std::runtime_error("Invalid output sizes for " + prefix);
    }
    const std::size_t state_stride = states.values.size() / logits.values.size();
    std::size_t active = 0;
    json << "    \"" << prefix << "\": [";
    bool first_item = true;
    for (std::size_t query = 0; query < logits.values.size(); ++query) {
      const float probability = sigmoid(logits.values[query]);
      if (probability < threshold) continue;
      ++active;
      if (include_states) {
        if (!first_item) json << ',';
        json << "\n      {\"query\": " << query << ", \"probability\": " << probability << ", \"states\": [";
        for (std::size_t i = 0; i < state_stride; ++i) {
          if (i) json << ',';
          json << states.values[query * state_stride + i];
        }
        json << "]}";
        first_item = false;
      }
    }
    result.active_counts[prefix] = active;
    if (include_states && !first_item) json << '\n' << "    ";
    json << ']';
    if (category + 1 != prefixes.size()) json << ',';
    json << '\n';
  }
  const auto& ego = require(outputs, "ego_states");
  json << "  },\n  \"ego_states\": [";
  for (std::size_t i = 0; i < ego.values.size(); ++i) {
    if (i) json << ',';
    json << ego.values[i];
  }
  json << "]\n}\n";
  result.json = json.str();
  return result;
}

