#include "QuantError.h"
#include "Quantizer.h"
#include "test_support.h"

#include <cmath>
#include <stdexcept>
#include <vector>

using qlinear::measure_error;
using qlinear::QuantError;

int main() {
    test::Runner tests;

    tests.add("identical inputs have zero error", [] {
        const QuantError e = measure_error({1.0f, -2.0f, 3.0f}, {1.0f, -2.0f, 3.0f});
        CHECK(e.max_abs == 0.0);
        CHECK(e.rmse == 0.0);
        CHECK(e.relative_fro == 0.0);
        CHECK(e.count == 3);
    });

    tests.add("hand-computed metrics", [] {
        // diffs = [1, 0, -2, 1]
        const QuantError e =
            measure_error({1.0f, 2.0f, -4.0f, 2.0f}, {0.0f, 2.0f, -2.0f, 3.0f});
        CHECK_NEAR(e.max_abs, 2.0, 1e-12);
        CHECK_NEAR(e.mean_abs, 1.0, 1e-12);
        CHECK_NEAR(e.rmse, std::sqrt(6.0 / 4.0), 1e-12);
        CHECK_NEAR(e.relative_fro, std::sqrt(6.0 / 25.0), 1e-12);
        CHECK_NEAR(e.max_relative, 1.0, 1e-12);  // |1-0| / |1|
    });

    tests.add("zero reference gives zero relative error, not NaN", [] {
        const QuantError e = measure_error({0.0f, 0.0f}, {0.1f, -0.1f});
        CHECK(e.relative_fro == 0.0);
        CHECK(e.max_relative == 0.0);
        CHECK(e.max_abs > 0.0);
    });

    tests.add("empty inputs give zero error", [] {
        const QuantError e = measure_error({}, {});
        CHECK(e.count == 0);
        CHECK(e.rmse == 0.0);
    });

    tests.add("size mismatch is rejected", [] {
        CHECK_THROWS(measure_error({1.0f}, {1.0f, 2.0f}), std::invalid_argument);
    });

    tests.add("round-trip max error is bounded by scale / 2", [] {
        std::vector<float> data;
        for (int i = 0; i < 1000; ++i) {
            data.push_back(std::sin(static_cast<float>(i)) * 4.0f);
        }
        const auto q = qlinear::Quantizer().quantize(data);
        const QuantError e = measure_error(data, qlinear::Quantizer::dequantize(q));
        CHECK(e.max_abs <= q.scale * 0.5 + 1e-6);
        CHECK(e.relative_fro < 0.01);
    });

    return tests.run();
}
