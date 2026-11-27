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

// P-th percentile of |x| using the nearest-rank method.
float abs_percentile(const std::vector<float>& data, double percentile) {
    std::vector<float> mags(data.size());
    std::transform(data.begin(), data.end(), mags.begin(),
                   [](float x) { return std::fabs(x); });
    const double rank = std::ceil(percentile / 100.0 * static_cast<double>(mags.size()));
    const std::size_t idx =
        std::min(mags.size() - 1, static_cast<std::size_t>(std::max(rank, 1.0)) - 1);
    std::nth_element(mags.begin(), mags.begin() + static_cast<std::ptrdiff_t>(idx),
                     mags.end());
    return mags[idx];
}

}  // namespace

Quantizer::Quantizer(ScaleMethod method, double percentile)
    : method_(method), percentile_(percentile) {
    if (!(percentile > 0.0 && percentile <= 100.0)) {
        throw std::invalid_argument("Quantizer: percentile must be in (0, 100]");
    }
}

float Quantizer::compute_scale(const std::vector<float>& data) const {
    require_finite(data);
    const float amax = max_abs(data);
    if (amax == 0.0f) {
        return 1.0f;
    }
    float clip = amax;
    if (method_ == ScaleMethod::Percentile) {
        clip = abs_percentile(data, percentile_);
        // A mostly-zero tensor can have a zero percentile; fall back to max|x|
        // rather than collapsing every nonzero value to saturation.
        if (clip == 0.0f) {
            clip = amax;
        }
    }
    return clip / static_cast<float>(kQMax);
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
