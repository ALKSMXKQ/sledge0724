#include "config.h"
#include "npy.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "Usage: test_io runtime.ini input.npy output.npy\n";
    return 2;
  }
  try {
    const auto config = load_config(argv[1]);
    const auto input = load_npy_f32(argv[2]);
    if (input.shape != config.expected_input_shape) throw std::runtime_error("Unexpected test input shape");
    if (input.values.size() != 1U * 12U * 256U * 256U) throw std::runtime_error("Unexpected test input size");
    save_npy_f32(argv[3], input.shape, input.values);
    std::cout << "C++ config/NPY I/O test passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

