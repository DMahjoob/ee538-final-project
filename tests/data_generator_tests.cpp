// Tests for DataGenerator: reproducibility, distribution shape, edge cases.

#include "DataGenerator.h"
#include "test_support.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

using namespace qlinear;

namespace {

struct Moments {
    double mean = 0.0;
    double stddev = 0.0;
};

Moments moments(const std::vector<float>& v) {
    double sum = 0.0;
    for (float x : v) sum += x;
    const double mean = sum / static_cast<double>(v.size());
    double sq = 0.0;
    for (float x : v) sq += (x - mean) * (x - mean);
    return {mean, std::sqrt(sq / static_cast<double>(v.size()))};
}

bool bitwise_equal(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() &&
           std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

}  // namespace

int main() {
    test::Runner runner;

    runner.add("same seed gives bitwise-identical matrices", [] {
        for (Distribution d : {Distribution::Uniform, Distribution::Normal,
                               Distribution::NormalOutliers}) {
            DataGenerator a(538);
            DataGenerator b(538);
            const auto spec = DistributionSpec::defaults(d);
            CHECK(bitwise_equal(a.generate(spec, 17, 31), b.generate(spec, 17, 31)));
        }
    });

    runner.add("different seeds give different matrices", [] {
        DataGenerator a(1);
        DataGenerator b(2);
        const auto spec = DistributionSpec::defaults(Distribution::Normal);
        CHECK(!bitwise_equal(a.generate(spec, 8, 8), b.generate(spec, 8, 8)));
    });

    runner.add("fixed snapshot of first values (cross-platform regression)", [] {
        // Recorded from this implementation. mt19937's sequence is fixed by
        // the standard and the transforms are our own, so these values must
        // be identical on every platform; a change means the generator (and
        // every experiment's inputs) changed.
        DataGenerator gen(538);
        const auto u = gen.generate(DistributionSpec::defaults(Distribution::Uniform), 1, 4);
        // Hex float literals are exact. Decimal: 0.56482625 0.370023251
        // 0.65842402 -0.523935199.
        const std::vector<float> expected_u = {0x1.2130e8p-1f, 0x1.7ae76p-2f,
                                               0x1.511cf4p-1f, -0x1.0c413cp-1f};
        CHECK(bitwise_equal(u, expected_u));

        DataGenerator gen2(538);
        const auto n = gen2.generate(DistributionSpec::defaults(Distribution::Normal), 1, 4);
        // Decimal: -0.693507612 -1.6029228 0.141238034 1.87475765.
        const std::vector<float> expected_n = {-0x1.63136ep-1f, -0x1.9a5926p+0f,
                                               0x1.214168p-3f, 0x1.dff01ep+0f};
        CHECK(bitwise_equal(n, expected_n));
    });

    runner.add("mt19937 engine matches the C++ standard's checkpoint", [] {
        // [rand.predef]: the 10000th output of a default-constructed mt19937
        // is 4123659995. Guards against a non-conforming engine.
        std::mt19937 engine;
        engine.discard(9999);
        CHECK(engine() == 4123659995u);
    });

    runner.add("uniform values stay in [a, b) and have the right moments", [] {
        DataGenerator gen(7);
        DistributionSpec spec = DistributionSpec::defaults(Distribution::Uniform);
        spec.a = -2.0f;
        spec.b = 3.0f;
        const auto v = gen.generate(spec, 400, 250);  // 100k samples
        for (float x : v) {
            CHECK(x >= -2.0f && x < 3.0f);
        }
        const Moments mo = moments(v);
        // U(-2, 3): mean 0.5, stddev 5/sqrt(12) = 1.443. Tolerances are
        // > 4 standard errors for 100k samples.
        CHECK_NEAR(mo.mean, 0.5, 0.02);
        CHECK_NEAR(mo.stddev, 5.0 / std::sqrt(12.0), 0.01);
    });

    runner.add("normal values have the requested mean and stddev", [] {
        DataGenerator gen(11);
        DistributionSpec spec = DistributionSpec::defaults(Distribution::Normal);
        spec.a = 3.0f;
        spec.b = 2.0f;
        const auto v = gen.generate(spec, 400, 250);
        const Moments mo = moments(v);
        CHECK_NEAR(mo.mean, 3.0, 0.03);
        CHECK_NEAR(mo.stddev, 2.0, 0.03);
        // Roughly 68.3% within one stddev.
        const auto inside = std::count_if(v.begin(), v.end(), [](float x) {
            return std::fabs(x - 3.0f) <= 2.0f;
        });
        CHECK_NEAR(static_cast<double>(inside) / static_cast<double>(v.size()), 0.6827, 0.01);
    });

    runner.add("normal_outliers injects roughly outlier_rate large values", [] {
        DataGenerator gen(13);
        DistributionSpec spec = DistributionSpec::defaults(Distribution::NormalOutliers);
        spec.outlier_rate = 0.01;
        spec.outlier_scale = 100.0f;
        const auto v = gen.generate(spec, 400, 250);
        // A non-outlier N(0,1) value above 20 essentially never happens.
        const auto big = std::count_if(v.begin(), v.end(),
                                       [](float x) { return std::fabs(x) > 20.0f; });
        const double rate = static_cast<double>(big) / static_cast<double>(v.size());
        // ~85% of x100 outliers exceed 20 in magnitude (|z| > 0.2).
        CHECK(rate > 0.006 && rate < 0.010);
    });

    runner.add("channel_outliers scales exactly the listed columns", [] {
        DataGenerator gen(17);
        DistributionSpec spec = DistributionSpec::defaults(Distribution::ChannelOutliers);
        spec.outlier_channels = {1, 4};
        spec.outlier_scale = 1000.0f;
        const std::size_t rows = 200;
        const std::size_t cols = 6;
        const auto v = gen.generate(spec, rows, cols);
        for (std::size_t c = 0; c < cols; ++c) {
            double max_abs = 0.0;
            for (std::size_t r = 0; r < rows; ++r) {
                max_abs = std::max(max_abs, static_cast<double>(std::fabs(v[r * cols + c])));
            }
            const bool outlier = c == 1 || c == 4;
            CHECK(outlier ? max_abs > 1000.0 : max_abs < 10.0);
        }
    });

    runner.add("zeros and constant", [] {
        DataGenerator gen(1);
        const auto z = gen.generate(DistributionSpec::defaults(Distribution::Zeros), 3, 4);
        CHECK(std::all_of(z.begin(), z.end(), [](float x) { return x == 0.0f; }));
        DistributionSpec c = DistributionSpec::defaults(Distribution::Constant);
        c.constant = -2.5f;
        const auto k = gen.generate(c, 2, 5);
        CHECK(k.size() == 10);
        CHECK(std::all_of(k.begin(), k.end(), [](float x) { return x == -2.5f; }));
    });

    runner.add("degenerate sizes", [] {
        DataGenerator gen(1);
        const auto spec = DistributionSpec::defaults(Distribution::Normal);
        CHECK(gen.generate(spec, 1, 1).size() == 1);
        CHECK(gen.generate(spec, 0, 5).empty());
    });

    runner.add("invalid specs throw", [] {
        DataGenerator gen(1);
        DistributionSpec u = DistributionSpec::defaults(Distribution::Uniform);
        u.a = 1.0f;
        u.b = 1.0f;
        CHECK_THROWS(gen.generate(u, 2, 2), std::invalid_argument);
        DistributionSpec n = DistributionSpec::defaults(Distribution::Normal);
        n.b = -1.0f;
        CHECK_THROWS(gen.generate(n, 2, 2), std::invalid_argument);
        DistributionSpec o = DistributionSpec::defaults(Distribution::NormalOutliers);
        o.outlier_rate = 1.5;
        CHECK_THROWS(gen.generate(o, 2, 2), std::invalid_argument);
        DistributionSpec ch = DistributionSpec::defaults(Distribution::ChannelOutliers);
        ch.outlier_channels = {3};
        CHECK_THROWS(gen.generate(ch, 2, 3), std::invalid_argument);
    });

    runner.add("distribution names round-trip", [] {
        for (Distribution d : {Distribution::Uniform, Distribution::Normal,
                               Distribution::NormalOutliers, Distribution::ChannelOutliers,
                               Distribution::Zeros, Distribution::Constant}) {
            CHECK(parse_distribution(to_string(d)) == d);
        }
        CHECK_THROWS(parse_distribution("gamma"), std::invalid_argument);
        CHECK_THROWS(parse_distribution(""), std::invalid_argument);
    });

    return runner.run();
}
