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

}  // namespace qlinear
