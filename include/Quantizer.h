#pragma once

// Symmetric per-tensor INT8 quantization for the Y = X * W^T operator.
//
// A float tensor x is represented as q = clamp(round(x / s), -127, 127) with a
// single scale s per tensor. Matrices are passed as flat row-major
// std::vector<float> buffers so this component does not depend on Matrix<T>;
// callers pass Matrix<float>::data() (or equivalent) directly.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace qlinear {

// Symmetric range: -128 is excluded so that +x and -x map to +q and -q.
inline constexpr int kQMax = 127;
inline constexpr int kQMin = -kQMax;

struct QuantizedTensor {
    std::vector<std::int8_t> values;
    float scale = 1.0f;
};

class Quantizer {
public:
    Quantizer() = default;

    // Scale such that the largest |x| maps to 127. An all-zero tensor gets
    // scale 1.0 so that quantization never divides by zero.
    // Throws std::invalid_argument on NaN or infinite input.
    float compute_scale(const std::vector<float>& data) const;

    // Quantizes the whole tensor with compute_scale(data).
    QuantizedTensor quantize(const std::vector<float>& data) const;

    // Quantizes with a caller-supplied scale (must be finite and > 0).
    // Values outside [-127 * scale, 127 * scale] saturate.
    static QuantizedTensor quantize_with_scale(const std::vector<float>& data,
                                               float scale);

    // Rounds half away from zero, then saturates to [-127, 127].
    static std::int8_t quantize_value(float x, float scale);

    // x_hat = q * scale.
    static std::vector<float> dequantize(const QuantizedTensor& q);

    // Converts the INT32 output of an INT8 kernel back to float:
    // y_hat[i] = acc[i] * scale_x * scale_w. The product is formed in double
    // because |acc| can exceed float's 24-bit exact-integer range.
    static std::vector<float> dequantize_accumulator(
        const std::vector<std::int32_t>& acc, float scale_x, float scale_w);

    // Largest inner dimension K for which a sum of K products of two values
    // in [-127, 127] is guaranteed to fit in int32: floor(INT32_MAX / 127^2).
    static constexpr std::size_t max_safe_inner_dim() {
        return static_cast<std::size_t>(INT32_MAX) /
               static_cast<std::size_t>(kQMax * kQMax);
    }

    // Throws std::invalid_argument if K could overflow the INT32 accumulator.
    static void check_inner_dim(std::size_t k);
};

}  // namespace qlinear
