#include "DataGenerator.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace qlinear {

namespace {

constexpr double kTwoPi = 6.283185307179586476925286766559;
constexpr float kTwoPow24Inv = 1.0f / 16777216.0f;  // 2^-24

}  // namespace

DistributionSpec DistributionSpec::defaults(Distribution kind) {
    DistributionSpec spec;
    spec.kind = kind;
    switch (kind) {
        case Distribution::Uniform:
            spec.a = -1.0f;
            spec.b = 1.0f;
            break;
        case Distribution::Normal:
        case Distribution::NormalOutliers:
        case Distribution::ChannelOutliers:
            spec.a = 0.0f;
            spec.b = 1.0f;
            break;
        case Distribution::Zeros:
        case Distribution::Constant:
            break;
    }
    return spec;
}

DataGenerator::DataGenerator(std::uint32_t seed) : seed_(seed), engine_(seed) {}

float DataGenerator::uniform01() {
    // mt19937 yields 32 uniform bits; keep the top 24 so the result is an
    // exactly representable float in [0, 1).
    return static_cast<float>(engine_() >> 8) * kTwoPow24Inv;
}

double DataGenerator::uniform01_open_left() {
    // (0, 1]: never 0, so log() is finite.
    return 1.0 - static_cast<double>(uniform01());
}

float DataGenerator::normal() {
    if (has_spare_) {
        has_spare_ = false;
        return static_cast<float>(spare_);
    }
    const double u1 = uniform01_open_left();
    const double u2 = static_cast<double>(uniform01());
    const double radius = std::sqrt(-2.0 * std::log(u1));
    const double angle = kTwoPi * u2;
    spare_ = radius * std::sin(angle);
    has_spare_ = true;
    return static_cast<float>(radius * std::cos(angle));
}

std::vector<float> DataGenerator::generate(const DistributionSpec& spec,
                                           std::size_t rows, std::size_t cols) {
    if (spec.kind == Distribution::Uniform && !(spec.b > spec.a)) {
        throw std::invalid_argument("uniform distribution needs b > a");
    }
    if ((spec.kind == Distribution::Normal || spec.kind == Distribution::NormalOutliers ||
         spec.kind == Distribution::ChannelOutliers) &&
        !(spec.b >= 0.0f)) {
        throw std::invalid_argument("normal distribution needs a standard deviation >= 0");
    }
    if (!(spec.outlier_rate >= 0.0 && spec.outlier_rate <= 1.0)) {
        throw std::invalid_argument("outlier_rate must be in [0, 1]");
    }
    std::vector<bool> outlier_column(cols, false);
    if (spec.kind == Distribution::ChannelOutliers) {
        for (std::size_t c : spec.outlier_channels) {
            if (c >= cols) {
                throw std::invalid_argument("outlier channel " + std::to_string(c) +
                                            " is out of range for " +
                                            std::to_string(cols) + " columns");
            }
            outlier_column[c] = true;
        }
    }

    std::vector<float> out(rows * cols, 0.0f);
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
            float& x = out[r * cols + c];
            switch (spec.kind) {
                case Distribution::Uniform:
                    x = spec.a + (spec.b - spec.a) * uniform01();
                    // Guard against rounding up to b when (b - a) * u rounds.
                    if (x >= spec.b) {
                        x = std::nextafter(spec.b, spec.a);
                    }
                    break;
                case Distribution::Normal:
                    x = spec.a + spec.b * normal();
                    break;
                case Distribution::NormalOutliers: {
                    // Draw the coin and the value every time so the sequence
                    // does not depend on earlier outcomes.
                    const bool outlier =
                        static_cast<double>(uniform01()) < spec.outlier_rate;
                    x = spec.a + spec.b * normal();
                    if (outlier) {
                        x *= spec.outlier_scale;
                    }
                    break;
                }
                case Distribution::ChannelOutliers:
                    x = spec.a + spec.b * normal();
                    if (outlier_column[c]) {
                        x *= spec.outlier_scale;
                    }
                    break;
                case Distribution::Zeros:
                    x = 0.0f;
                    break;
                case Distribution::Constant:
                    x = spec.constant;
                    break;
            }
        }
    }
    return out;
}

Distribution parse_distribution(std::string_view name) {
    if (name == "uniform") return Distribution::Uniform;
    if (name == "normal") return Distribution::Normal;
    if (name == "normal_outliers") return Distribution::NormalOutliers;
    if (name == "channel_outliers") return Distribution::ChannelOutliers;
    if (name == "zeros") return Distribution::Zeros;
    if (name == "constant") return Distribution::Constant;
    throw std::invalid_argument(
        "unknown distribution '" + std::string(name) +
        "' (expected uniform, normal, normal_outliers, channel_outliers, zeros, constant)");
}

std::string_view to_string(Distribution kind) {
    switch (kind) {
        case Distribution::Uniform: return "uniform";
        case Distribution::Normal: return "normal";
        case Distribution::NormalOutliers: return "normal_outliers";
        case Distribution::ChannelOutliers: return "channel_outliers";
        case Distribution::Zeros: return "zeros";
        case Distribution::Constant: return "constant";
    }
    return "unknown";
}

}  // namespace qlinear
