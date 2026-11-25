#include "Quantizer.h"
#include "test_support.h"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

using qlinear::Quantizer;
using qlinear::QuantizedTensor;

namespace {

void add_scale_tests(test::Runner& tests) {
    tests.add("scale maps max |x| to 127", [] {
        Quantizer quantizer;
        const float scale = quantizer.compute_scale({0.5f, -2.54f, 1.0f});
        CHECK_NEAR(scale, 2.54f / 127.0f, 1e-7f);
    });

    tests.add("all-zero tensor gets scale 1 and quantizes to zeros", [] {
        Quantizer quantizer;
        const QuantizedTensor q = quantizer.quantize({0.0f, -0.0f, 0.0f});
        CHECK(q.scale == 1.0f);
        for (std::int8_t v : q.values) {
            CHECK(v == 0);
        }
    });

    tests.add("empty tensor quantizes to empty", [] {
        Quantizer quantizer;
        const QuantizedTensor q = quantizer.quantize({});
        CHECK(q.values.empty());
        CHECK(q.scale == 1.0f);
    });

    tests.add("non-finite input is rejected", [] {
        Quantizer quantizer;
        const float nan = std::numeric_limits<float>::quiet_NaN();
        const float inf = std::numeric_limits<float>::infinity();
        CHECK_THROWS(quantizer.compute_scale({1.0f, nan}), std::invalid_argument);
        CHECK_THROWS(quantizer.quantize({inf}), std::invalid_argument);
    });

    tests.add("invalid explicit scale is rejected", [] {
        CHECK_THROWS(Quantizer::quantize_with_scale({1.0f}, 0.0f),
                     std::invalid_argument);
        CHECK_THROWS(Quantizer::quantize_with_scale({1.0f}, -1.0f),
                     std::invalid_argument);
    });
}

void add_quantize_tests(test::Runner& tests) {
    tests.add("extremes map to +/-127 and are symmetric", [] {
        Quantizer quantizer;
        const QuantizedTensor q = quantizer.quantize({-3.0f, 3.0f, 1.5f, -1.5f});
        CHECK(q.values[0] == -127);
        CHECK(q.values[1] == 127);
        CHECK(q.values[2] == -q.values[3]);
    });

    tests.add("rounding is half away from zero", [] {
        CHECK(Quantizer::quantize_value(0.5f, 1.0f) == 1);
        CHECK(Quantizer::quantize_value(-0.5f, 1.0f) == -1);
        CHECK(Quantizer::quantize_value(0.49f, 1.0f) == 0);
        CHECK(Quantizer::quantize_value(2.5f, 1.0f) == 3);
    });

    tests.add("out-of-range values saturate and -128 never appears", [] {
        const QuantizedTensor q =
            Quantizer::quantize_with_scale({1000.0f, -1000.0f, -127.6f}, 1.0f);
        CHECK(q.values[0] == 127);
        CHECK(q.values[1] == -127);
        CHECK(q.values[2] == -127);
    });

    tests.add("every quantized value stays in [-127, 127]", [] {
        std::vector<float> data;
        for (int i = -500; i <= 500; ++i) {
            data.push_back(static_cast<float>(i) * 0.013f);
        }
        const QuantizedTensor q = Quantizer().quantize(data);
        for (std::int8_t v : q.values) {
            CHECK(v >= -127 && v <= 127);
        }
    });
}

}  // namespace

int main() {
    test::Runner tests;
    add_scale_tests(tests);
    add_quantize_tests(tests);
    return tests.run();
}
