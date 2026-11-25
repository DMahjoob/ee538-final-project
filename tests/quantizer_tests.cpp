#include "Quantizer.h"
#include "test_support.h"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

using qlinear::Quantizer;
using qlinear::QuantizedTensor;
using qlinear::ScaleMethod;

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

void add_dequantize_tests(test::Runner& tests) {
    tests.add("round trip error is at most scale / 2 per element", [] {
        std::vector<float> data;
        for (int i = -200; i <= 200; ++i) {
            data.push_back(static_cast<float>(i) * 0.0173f);
        }
        const QuantizedTensor q = Quantizer().quantize(data);
        const std::vector<float> back = Quantizer::dequantize(q);
        CHECK(back.size() == data.size());
        for (std::size_t i = 0; i < data.size(); ++i) {
            CHECK(std::fabs(back[i] - data[i]) <= q.scale * 0.5f + 1e-6f);
        }
    });

    tests.add("accumulator dequantization applies scale_x * scale_w", [] {
        // x = [1, 2], w = [3, -4]  ->  x . w = -5
        const std::vector<float> x = {1.0f, 2.0f};
        const std::vector<float> w = {3.0f, -4.0f};
        Quantizer quantizer;
        const QuantizedTensor qx = quantizer.quantize(x);
        const QuantizedTensor qw = quantizer.quantize(w);
        std::int32_t acc = 0;
        for (std::size_t i = 0; i < x.size(); ++i) {
            acc += static_cast<std::int32_t>(qx.values[i]) * qw.values[i];
        }
        const std::vector<float> y =
            Quantizer::dequantize_accumulator({acc}, qx.scale, qw.scale);
        CHECK_NEAR(y[0], -5.0f, 0.05f);
    });

    tests.add("accumulator dequantization keeps precision for large int32", [] {
        const std::int32_t big = 2'000'000'001;  // not representable in float
        const std::vector<float> y =
            Quantizer::dequantize_accumulator({big, -big, 0}, 0.5f, 0.25f);
        CHECK_NEAR(y[0], 250'000'000.125, 32.0);
        CHECK(y[1] == -y[0]);
        CHECK(y[2] == 0.0f);
    });

    tests.add("accumulator dequantization rejects bad scales", [] {
        CHECK_THROWS(Quantizer::dequantize_accumulator({1}, 0.0f, 1.0f),
                     std::invalid_argument);
    });

    tests.add("max safe inner dim keeps worst case inside int32", [] {
        const std::size_t k = Quantizer::max_safe_inner_dim();
        CHECK(k == 133144);
        const std::int64_t worst = static_cast<std::int64_t>(k) * 127 * 127;
        CHECK(worst <= std::numeric_limits<std::int32_t>::max());
        CHECK((worst + 127 * 127) > std::numeric_limits<std::int32_t>::max());
        Quantizer::check_inner_dim(k);  // must not throw
        CHECK_THROWS(Quantizer::check_inner_dim(k + 1), std::invalid_argument);
    });
}

void add_outlier_tests(test::Runner& tests) {
    // 999 values in [-1, 1] plus one outlier at 100.
    auto outlier_data = [] {
        std::vector<float> data;
        for (int i = 0; i < 999; ++i) {
            data.push_back(static_cast<float>(i % 201 - 100) / 100.0f);
        }
        data.push_back(100.0f);
        return data;
    };

    tests.add("max-abs scale is stretched by a single outlier", [=] {
        const float scale = Quantizer().compute_scale(outlier_data());
        CHECK_NEAR(scale, 100.0f / 127.0f, 1e-6f);
    });

    tests.add("percentile scale ignores the outlier and saturates it", [=] {
        Quantizer quantizer(ScaleMethod::Percentile, 99.0);
        const std::vector<float> data = outlier_data();
        const QuantizedTensor q = quantizer.quantize(data);
        CHECK(q.scale <= 1.0f / 127.0f + 1e-6f);
        CHECK(q.values.back() == 127);
    });

    tests.add("percentile 100 matches max-abs", [=] {
        const std::vector<float> data = outlier_data();
        CHECK(Quantizer(ScaleMethod::Percentile, 100.0).compute_scale(data) ==
              Quantizer().compute_scale(data));
    });

    tests.add("sparse tensor falls back to max-abs when percentile is zero", [] {
        std::vector<float> data(1000, 0.0f);
        data[3] = 2.0f;
        const float scale = Quantizer(ScaleMethod::Percentile, 99.0).compute_scale(data);
        CHECK_NEAR(scale, 2.0f / 127.0f, 1e-7f);
    });

    tests.add("invalid percentile is rejected", [] {
        CHECK_THROWS(Quantizer(ScaleMethod::Percentile, 0.0), std::invalid_argument);
        CHECK_THROWS(Quantizer(ScaleMethod::Percentile, 100.5), std::invalid_argument);
    });
}

}  // namespace

int main() {
    test::Runner tests;
    add_scale_tests(tests);
    add_quantize_tests(tests);
    add_dequantize_tests(tests);
    add_outlier_tests(tests);
    return tests.run();
}
