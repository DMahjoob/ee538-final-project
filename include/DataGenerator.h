#pragma once

// Reproducible float32 test-matrix generation.
//
// Only std::mt19937 is taken from <random>: its output sequence is fixed by
// the C++ standard. The std::*_distribution classes are NOT portable (libc++
// and libstdc++ map the same engine output to different values), so the
// uniform and normal transforms are implemented here. The same seed therefore
// produces the same matrices on every teammate's machine.

#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <string_view>
#include <vector>

namespace qlinear {

enum class Distribution {
    Uniform,          // U[a, b)
    Normal,           // N(a, b^2): a = mean, b = standard deviation
    NormalOutliers,   // N(a, b^2); each value is scaled by outlier_scale with
                      // probability outlier_rate
    ChannelOutliers,  // N(a, b^2); every value in the listed columns is scaled
                      // by outlier_scale (LLM-style activation outlier channels)
    Zeros,            // all 0
    Constant,         // all equal to `constant`
};

struct DistributionSpec {
    Distribution kind = Distribution::Uniform;
    float a = -1.0f;
    float b = 1.0f;
    double outlier_rate = 0.001;
    float outlier_scale = 50.0f;
    std::vector<std::size_t> outlier_channels;
    float constant = 0.0f;

    // Defaults used by the CLI and the experiments for each kind:
    // uniform [-1, 1), standard normal, 0.1% outliers at x50.
    static DistributionSpec defaults(Distribution kind);
};

class DataGenerator {
public:
    explicit DataGenerator(std::uint32_t seed);

    std::uint32_t seed() const { return seed_; }

    // Returns a row-major rows x cols matrix. Throws std::invalid_argument for
    // an invalid spec (b <= a for Uniform, negative standard deviation,
    // outlier_rate outside [0, 1], a channel index >= cols).
    std::vector<float> generate(const DistributionSpec& spec, std::size_t rows,
                                std::size_t cols);

    // Uniform on [0, 1) with 24 random bits (every value is an exact float).
    float uniform01();

    // Standard normal via the Box-Muller transform, computed in double.
    // Values are produced in pairs; the second one is cached.
    float normal();

private:
    double uniform01_open_left();  // (0, 1], safe for log()

    std::uint32_t seed_;
    std::mt19937 engine_;
    bool has_spare_ = false;
    double spare_ = 0.0;
};

// "uniform", "normal", "normal_outliers", "channel_outliers", "zeros",
// "constant". parse_distribution throws std::invalid_argument on other input.
Distribution parse_distribution(std::string_view name);
std::string_view to_string(Distribution kind);

}  // namespace qlinear
