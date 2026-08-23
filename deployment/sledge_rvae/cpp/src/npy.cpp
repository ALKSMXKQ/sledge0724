#include "npy.h"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace {
std::size_t element_count(const std::vector<std::int64_t>& shape) {
  return std::accumulate(shape.begin(), shape.end(), std::size_t{1},
                         [](std::size_t a, std::int64_t b) {
                           if (b <= 0) throw std::runtime_error("NPY dimensions must be positive");
                           return a * static_cast<std::size_t>(b);
                         });
}

std::vector<std::int64_t> parse_shape(const std::string& header) {
  std::smatch match;
  const std::regex shape_re(R"('shape'\s*:\s*\(([^\)]*)\))");
  if (!std::regex_search(header, match, shape_re)) throw std::runtime_error("NPY shape missing");
  std::vector<std::int64_t> shape;
  std::stringstream stream(match[1].str());
  std::string token;
  while (std::getline(stream, token, ',')) {
    if (token.find_first_not_of(" \t") == std::string::npos) continue;
    shape.push_back(std::stoll(token));
  }
  if (shape.empty()) throw std::runtime_error("Scalar NPY is unsupported");
  return shape;
}

template <typename T>
T read_little_endian(std::istream& stream) {
  T value{};
  stream.read(reinterpret_cast<char*>(&value), sizeof(T));
  if (!stream) throw std::runtime_error("Truncated NPY header");
  return value;
}
}  // namespace

FloatArray load_npy_f32(const std::string& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) throw std::runtime_error("Cannot open NPY: " + path);
  char magic[6];
  stream.read(magic, sizeof(magic));
  const char expected[6] = {'\x93', 'N', 'U', 'M', 'P', 'Y'};
  if (!stream || std::memcmp(magic, expected, sizeof(magic)) != 0) throw std::runtime_error("Invalid NPY magic: " + path);
  const auto major = read_little_endian<std::uint8_t>(stream);
  (void)read_little_endian<std::uint8_t>(stream);
  std::uint32_t header_length = 0;
  if (major == 1) header_length = read_little_endian<std::uint16_t>(stream);
  else if (major == 2 || major == 3) header_length = read_little_endian<std::uint32_t>(stream);
  else throw std::runtime_error("Unsupported NPY version");
  std::string header(header_length, '\0');
  stream.read(header.data(), static_cast<std::streamsize>(header.size()));
  if (!stream) throw std::runtime_error("Truncated NPY header");
  if (header.find("'fortran_order': False") == std::string::npos &&
      header.find("\"fortran_order\": False") == std::string::npos) {
    throw std::runtime_error("Only C-order NPY arrays are supported");
  }
  if (header.find("'<f4'") == std::string::npos && header.find("'|f4'") == std::string::npos &&
      header.find("\"<f4\"") == std::string::npos) {
    throw std::runtime_error("Only little-endian float32 NPY arrays are supported");
  }
  FloatArray array;
  array.shape = parse_shape(header);
  array.values.resize(element_count(array.shape));
  stream.read(reinterpret_cast<char*>(array.values.data()),
              static_cast<std::streamsize>(array.values.size() * sizeof(float)));
  if (!stream) throw std::runtime_error("Truncated NPY data: " + path);
  return array;
}

void save_npy_f32(const std::string& path, const std::vector<std::int64_t>& shape,
                  const std::vector<float>& values) {
  if (element_count(shape) != values.size()) throw std::runtime_error("NPY shape/data size mismatch");
  std::filesystem::create_directories(std::filesystem::path(path).parent_path());
  std::ostringstream shape_text;
  shape_text << '(';
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (i) shape_text << ", ";
    shape_text << shape[i];
  }
  if (shape.size() == 1) shape_text << ',';
  shape_text << ')';
  std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': " + shape_text.str() + ", }";
  const std::size_t prefix = 10;
  const std::size_t padding = (64 - ((prefix + header.size() + 1) % 64)) % 64;
  header.append(padding, ' ');
  header.push_back('\n');
  if (header.size() > 65535) throw std::runtime_error("NPY v1 header too large");
  std::ofstream stream(path, std::ios::binary);
  if (!stream) throw std::runtime_error("Cannot write NPY: " + path);
  const char magic[6] = {'\x93', 'N', 'U', 'M', 'P', 'Y'};
  stream.write(magic, sizeof(magic));
  const std::uint8_t version[2] = {1, 0};
  stream.write(reinterpret_cast<const char*>(version), sizeof(version));
  const auto length = static_cast<std::uint16_t>(header.size());
  stream.write(reinterpret_cast<const char*>(&length), sizeof(length));
  stream.write(header.data(), static_cast<std::streamsize>(header.size()));
  stream.write(reinterpret_cast<const char*>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(float)));
}

