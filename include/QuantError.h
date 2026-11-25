#pragma once

// Error metrics between a float32 reference and its approximation. Used both
// for quantize/dequantize round trips and for comparing the INT8 operator's
// output against the float32 reference Y = X * W^T.

#include <cstddef>
#include <vector>

namespace qlinear {

struct QuantError {
    double max_abs = 0.0;       // max_i |ref_i - approx_i|
    double mean_abs = 0.0;      // mean_i |ref_i - approx_i|
    double rmse = 0.0;          // sqrt(mean_i (ref_i - approx_i)^2)
    double relative_fro = 0.0;  // ||ref - approx||_F / ||ref||_F (0 if ref == 0)
    double max_relative = 0.0;  // max_i |ref_i - approx_i| / |ref_i|, ref_i != 0
    std::size_t count = 0;
};

// Throws std::invalid_argument if the sizes differ.
QuantError measure_error(const std::vector<float>& reference,
                         const std::vector<float>& approx);

}  // namespace qlinear
