#include "config.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace {
std::string trim(std::string value) {
  const auto not_space = [](unsigned char c) { return !std::isspace(c); };
  value.erase(value.begin(), std::find_if(value.begin(), value.end(), not_space));
  value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(), value.end());
  return value;
}

bool parse_bool(const std::string& value) {
  if (value == "true" || value == "1") return true;
  if (value == "false" || value == "0") return false;
  throw std::runtime_error("Invalid boolean: " + value);
}

std::vector<std::int64_t> parse_shape(const std::string& value) {
  std::vector<std::int64_t> shape;
  std::stringstream stream(value);
  std::string token;
  while (std::getline(stream, token, ',')) shape.push_back(std::stoll(trim(token)));
  return shape;
}
}  // namespace

RuntimeConfig load_config(const std::string& path) {
  std::ifstream stream(path);
  if (!stream) throw std::runtime_error("Cannot open config: " + path);
  std::unordered_map<std::string, std::string> values;
  std::string line;
  std::size_t line_number = 0;
  while (std::getline(stream, line)) {
    ++line_number;
    line = trim(line);
    if (line.empty() || line[0] == '#') continue;
    const auto equals = line.find('=');
    if (equals == std::string::npos) {
      throw std::runtime_error("Invalid config line " + std::to_string(line_number));
    }
    values[trim(line.substr(0, equals))] = trim(line.substr(equals + 1));
  }
  const auto required = [&](const std::string& key) -> const std::string& {
    const auto it = values.find(key);
    if (it == values.end() || it->second.empty()) throw std::runtime_error("Missing config key: " + key);
    return it->second;
  };
  RuntimeConfig config;
  config.engine_path = required("engine_path");
  config.input_path = required("input_path");
  config.output_dir = required("output_dir");
  config.metrics_path = required("metrics_path");
  if (values.count("input_name")) config.input_name = values.at("input_name");
  if (values.count("precision")) config.precision = values.at("precision");
  if (values.count("expected_input_shape")) config.expected_input_shape = parse_shape(values.at("expected_input_shape"));
  if (values.count("threshold")) config.threshold = std::stof(values.at("threshold"));
  if (values.count("warmup_iterations")) config.warmup_iterations = std::stoull(values.at("warmup_iterations"));
  if (values.count("benchmark_iterations")) config.benchmark_iterations = std::stoull(values.at("benchmark_iterations"));
  if (values.count("save_raw_outputs")) config.save_raw_outputs = parse_bool(values.at("save_raw_outputs"));
  if (values.count("save_postprocessed")) config.save_postprocessed = parse_bool(values.at("save_postprocessed"));
  if (!(config.threshold >= 0.0F && config.threshold <= 1.0F)) throw std::runtime_error("threshold must be in [0,1]");
  if (config.precision != "fp32" && config.precision != "fp16") throw std::runtime_error("precision must be fp32 or fp16");
  if (config.benchmark_iterations == 0) throw std::runtime_error("benchmark_iterations must be positive");
  return config;
}
