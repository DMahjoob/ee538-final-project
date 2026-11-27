// Tests for the independent float32 reference Y = X * W^T.

#include "DataGenerator.h"
#include "MatrixIO.h"
#include "ReferenceKernel.h"
#include "test_support.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

using namespace qlinear;

int main() {
    test::Runner runner;

    runner.add("golden example from the proposal: Y = [[-1.5, -1.5]]", [] {
        const std::string dir = QLINEAR_TEST_DATA_DIR;
        const MatrixData x = load_matrix(dir + "/golden_x.txt");
        const MatrixData w = load_matrix(dir + "/golden_w.txt");
        const MatrixData expected = load_matrix(dir + "/golden_y_expected.txt");
        const auto y = reference_linear(x.values, w.values, x.rows, x.cols, w.rows);
        CHECK(expected.rows == 1 && expected.cols == 2);
        // Every intermediate is exactly representable, so the match is exact.
        CHECK(y == expected.values);
    });

    runner.add("hand-computed non-square case (2x3 * (4x3)^T)", [] {
        const std::vector<float> x = {1, 2, 3,
                                      -1, 0, 2};
        const std::vector<float> w = {1, 0, 0,
                                      0, 1, 0,
                                      0, 0, 1,
                                      1, 1, 1};
        const auto y = reference_linear(x, w, 2, 3, 4);
        CHECK((y == std::vector<float>{1, 2, 3, 6,
                                       -1, 0, 2, 1}));
    });

    runner.add("1 x 1 and single-row shapes", [] {
        CHECK(reference_linear({3}, {-4}, 1, 1, 1) == std::vector<float>{-12});
        const auto y = reference_linear({1, 1, 1, 1}, {1, 2, 3, 4, 5, 6, 7, 8}, 1, 4, 2);
        CHECK((y == std::vector<float>{10, 26}));
    });

    runner.add("zero matrix gives zero output", [] {
        const auto y = reference_linear(std::vector<float>(6, 0.0f), {1, 2, 3, 4, 5, 6}, 2,
                                        3, 2);
        CHECK((y == std::vector<float>(4, 0.0f)));
    });

    runner.add("matches an independent row/column formulation on random data", [] {
        // Y = X * W^T, recomputed as (W * X^T)^T with a float accumulator.
        DataGenerator gen(42);
        const auto spec = DistributionSpec::defaults(Distribution::Normal);
        const std::size_t m = 5, k = 37, n = 9;
        const auto x = gen.generate(spec, m, k);
        const auto w = gen.generate(spec, n, k);
        const auto y = reference_linear(x, w, m, k, n);
        for (std::size_t j = 0; j < n; ++j) {
            for (std::size_t i = 0; i < m; ++i) {
                float acc = 0.0f;
                for (std::size_t p = 0; p < k; ++p) acc += w[j * k + p] * x[i * k + p];
                CHECK_NEAR(y[i * n + j], acc, 1e-4f * (1.0f + std::fabs(acc)));
            }
        }
    });

    runner.add("double accumulation avoids float cancellation error", [] {
        // 1e8 + 1 - 1e8 = 1 exactly in double, but 0 in float.
        const auto y = reference_linear({1e8f, 1.0f, -1e8f}, {1.0f, 1.0f, 1.0f}, 1, 3, 1);
        CHECK(y[0] == 1.0f);
    });

    runner.add("size mismatches throw", [] {
        CHECK_THROWS(reference_linear({1, 2, 3}, {1, 2}, 1, 2, 1), std::invalid_argument);
        CHECK_THROWS(reference_linear({1, 2}, {1, 2, 3}, 1, 2, 1), std::invalid_argument);
    });

    return runner.run();
}
