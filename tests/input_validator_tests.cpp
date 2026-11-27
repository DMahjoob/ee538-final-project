// Tests for InputValidator: shape rules, the INT32 overflow bound, size cap,
// block size and command-line integer parsing.

#include "InputValidator.h"
#include "Quantizer.h"
#include "test_support.h"

#include <limits>
#include <stdexcept>
#include <string>

using namespace qlinear;

namespace {

MatrixData matrix(std::size_t rows, std::size_t cols) {
    MatrixData m;
    m.rows = rows;
    m.cols = cols;
    m.values.assign(rows * cols, 1.0f);
    return m;
}

bool message_contains(const std::function<void()>& fn, const std::string& fragment) {
    try {
        fn();
    } catch (const std::invalid_argument& e) {
        return std::string(e.what()).find(fragment) != std::string::npos;
    }
    return false;
}

}  // namespace

int main() {
    test::Runner runner;

    runner.add("accepts compatible shapes, including non-square and 1 x 1", [] {
        validate_linear_shapes(matrix(1, 1), matrix(1, 1));
        validate_linear_shapes(matrix(3, 5), matrix(7, 5));
        validate_linear_shapes(matrix(1, 64), matrix(64, 64));
        CHECK(true);
    });

    runner.add("rejects mismatched inner dimensions", [] {
        CHECK(message_contains([] { validate_linear_shapes(matrix(1, 2), matrix(2, 3)); },
                               "inner dimensions do not match: X is 1 x 2 and W is 2 x 3"));
    });

    runner.add("rejects empty matrices", [] {
        CHECK_THROWS(validate_linear_shapes(0, 4, 4, 4), std::invalid_argument);
        CHECK_THROWS(validate_linear_shapes(4, 0, 4, 0), std::invalid_argument);
        CHECK_THROWS(validate_linear_shapes(4, 4, 0, 4), std::invalid_argument);
    });

    runner.add("rejects a value count that disagrees with the declared shape", [] {
        MatrixData x = matrix(2, 2);
        x.values.pop_back();
        CHECK(message_contains([&] { validate_linear_shapes(x, matrix(2, 2)); },
                               "X holds 3 values but is declared 2 x 2"));
    });

    runner.add("INT32 overflow bound: K = 133144 accepted, 133145 rejected", [] {
        const std::size_t limit = Quantizer::max_safe_inner_dim();
        CHECK(limit == 133144);
        // Checked from dimensions only (M = N = 1), nothing is allocated.
        validate_linear_shapes(1, limit, 1, limit);
        CHECK(message_contains([&] { validate_linear_shapes(1, limit + 1, 1, limit + 1); },
                               "exceeds 133144"));
    });

    runner.add("rejects matrices above the element cap", [] {
        validate_dimensions(1, kMaxElements, "X");
        CHECK_THROWS(validate_dimensions(2, kMaxElements / 2 + 1, "X"), std::invalid_argument);
        // rows * cols would overflow size_t: must not wrap around and pass.
        const std::size_t huge = std::numeric_limits<std::size_t>::max() / 2;
        CHECK_THROWS(validate_dimensions(huge, 4, "X"), std::invalid_argument);
    });

    runner.add("rejects non-finite values with their position", [] {
        MatrixData m = matrix(2, 3);
        validate_finite(m, "X");
        m.values[4] = std::numeric_limits<float>::quiet_NaN();
        CHECK(message_contains([&] { validate_finite(m, "X"); }, "row 1, column 1"));
        m.values[4] = -std::numeric_limits<float>::infinity();
        CHECK_THROWS(validate_finite(m, "X"), std::invalid_argument);
    });

    runner.add("block size must be positive; larger than the matrix is allowed", [] {
        validate_block_size(1);
        validate_block_size(1 << 20);
        CHECK_THROWS(validate_block_size(0), std::invalid_argument);
    });

    runner.add("parse_size accepts positive integers only", [] {
        CHECK(parse_size("64", "--block") == 64);
        CHECK(parse_size("007", "--block") == 7);
        for (const char* bad : {"", "0", "-1", "+1", "1.5", " 1", "1 ", "abc", "1e3",
                                "99999999999999999999999"}) {
            CHECK_THROWS(parse_size(bad, "--block"), std::invalid_argument);
        }
        CHECK(message_contains([] { (void)parse_size("x", "--rows"); },
                               "--rows must be a positive integer, got 'x'"));
    });

    return runner.run();
}
