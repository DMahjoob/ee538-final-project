#include "Quantizer.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace qlinear {

namespace {

void require_finite(const std::vector<float>& data) {
    for (std::size_t i = 0; i < data.size(); ++i) {
        if (!std::isfinite(data[i])) {
            throw std::invalid_argument("quantize: non-finite value at index " +
                                        std::to_string(i));
        }
    }
}

void require_valid_scale(float scale) {
    if (!std::isfinite(scale) || scale <= 0.0f) {
        throw std::invalid_argument("quantize: scale must be finite and > 0");
    }
}

float max_abs(const std::vector<float>& data) {
    float result = 0.0f;
    for (float x : data) {
        result = std::max(result, std::fabs(x));
    }
    return result;
}

}  // namespace

float Quantizer::compute_scale(const std::vector<float>& data) const {
    require_finite(data);
    const float amax = max_abs(data);
    if (amax == 0.0f) {
        return 1.0f;
    }
    return amax / static_cast<float>(kQMax);
}

QuantizedTensor Quantizer::quantize(const std::vector<float>& data) const {
    return quantize_with_scale(data, compute_scale(data));
}

QuantizedTensor Quantizer::quantize_with_scale(const std::vector<float>& data,
                                               float scale) {
    require_valid_scale(scale);
    require_finite(data);
    QuantizedTensor out;
    out.scale = scale;
    out.values.resize(data.size());
    for (std::size_t i = 0; i < data.size(); ++i) {
        out.values[i] = quantize_value(data[i], scale);
    }
    return out;
}

std::int8_t Quantizer::quantize_value(float x, float scale) {
    const float q = std::round(x / scale);
    const float clamped = std::clamp(q, static_cast<float>(kQMin),
                                     static_cast<float>(kQMax));
    return static_cast<std::int8_t>(clamped);
}

std::vector<float> Quantizer::dequantize(const QuantizedTensor& q) {
    std::vector<float> out(q.values.size());
    for (std::size_t i = 0; i < q.values.size(); ++i) {
        out[i] = static_cast<float>(q.values[i]) * q.scale;
    }
    return out;
}

std::vector<float> Quantizer::dequantize_accumulator(
    const std::vector<std::int32_t>& acc, float scale_x, float scale_w) {
    require_valid_scale(scale_x);
    require_valid_scale(scale_w);
    const double combined = static_cast<double>(scale_x) * scale_w;
    std::vector<float> out(acc.size());
    for (std::size_t i = 0; i < acc.size(); ++i) {
        out[i] = static_cast<float>(static_cast<double>(acc[i]) * combined);
    }
    return out;
}

void Quantizer::check_inner_dim(std::size_t k) {
    if (k > max_safe_inner_dim()) {
        throw std::invalid_argument(
            "inner dimension " + std::to_string(k) +
            " exceeds INT32-safe limit " + std::to_string(max_safe_inner_dim()));
    }
}

}  // namespace qlinear
