// Tests for the text matrix format: parsing, every rejection path, and exact
// write/read round trips.

#include "MatrixIO.h"
#include "test_support.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

using namespace qlinear;

namespace {

MatrixData parse(const std::string& text) {
    std::istringstream in(text);
    return read_matrix(in, "test");
}

// Parses `text`, expects an InputError on `line` whose message contains
// `fragment`.
void expect_error(const std::string& text, std::size_t line, const std::string& fragment) {
    try {
        (void)parse(text);
        CHECK(false && "expected InputError");
    } catch (const InputError& e) {
        CHECK(e.line() == line);
        CHECK(e.source() == "test");
        const bool has_fragment = std::string(e.what()).find(fragment) != std::string::npos;
        if (!has_fragment) {
            std::cerr << "  message was: " << e.what() << "\n";
        }
        CHECK(has_fragment);
    }
}

std::string data_path(const std::string& name) {
    return std::string(QLINEAR_TEST_DATA_DIR) + "/" + name;
}

}  // namespace

int main() {
    test::Runner runner;

    runner.add("parses a simple matrix row-major", [] {
        const MatrixData m = parse("2 3\n1 2 3\n4 5 6\n");
        CHECK(m.rows == 2 && m.cols == 3);
        CHECK((m.values == std::vector<float>{1, 2, 3, 4, 5, 6}));
    });

    runner.add("comments, blank lines and free whitespace are allowed", [] {
        const MatrixData m = parse("# leading comment\n\n  2\t2   # header\n1 2 3\n\n 4 # end\n");
        CHECK(m.rows == 2 && m.cols == 2);
        CHECK((m.values == std::vector<float>{1, 2, 3, 4}));
    });

    runner.add("scientific notation, signs and subnormals", [] {
        const MatrixData m = parse("1 4\n-1.5e2 +3 0.25 1e-40\n");
        CHECK(m.values[0] == -150.0f);
        CHECK(m.values[1] == 3.0f);
        CHECK(m.values[2] == 0.25f);
        CHECK(m.values[3] > 0.0f && m.values[3] < std::numeric_limits<float>::min());
    });

    runner.add("1 x 1 and single-row matrices", [] {
        CHECK(parse("1 1\n7\n").values == std::vector<float>{7});
        CHECK(parse("1 5\n1 2 3 4 5\n").values.size() == 5);
    });

    runner.add("rejects empty input and header problems", [] {
        expect_error("", 0, "empty input");
        expect_error("# only a comment\n", 0, "empty input");
        expect_error("2 x\n1 2\n", 1, "cols must be a positive integer, got 'x'");
        expect_error("2.5 2\n", 1, "rows must be a positive integer");
        expect_error("-2 2\n", 1, "rows must be a positive integer");
        expect_error("0 2\n", 1, "rows must be at least 1");
        expect_error("2 0\n", 1, "cols must be at least 1");
        expect_error("3\n1 2 3\n", 1, "expected '<rows> <cols>' on one line");
        expect_error("3", 1, "found only one number");
        expect_error("99999999999999999999999 2\n", 1, "too large");
        expect_error("100000 100000\n", 1, "exceeds the limit");
    });

    runner.add("rejects bad values with their line numbers", [] {
        expect_error("1 2\n1 abc\n", 2, "expected a number, got 'abc'");
        expect_error("1 2\n1 2x\n", 2, "expected a number, got '2x'");
        expect_error("1 2\n1\nnan\n", 3, "NaN is not allowed");
        expect_error("1 2\n-inf 1\n", 2, "infinity is not allowed");
        expect_error("1 2\n1 1e39\n", 2, "outside the float32 range");
        expect_error("1 2\n1 -1e39\n", 2, "outside the float32 range");
    });

    runner.add("rejects too few and too many values", [] {
        expect_error("2 2\n1 2\n3\n", 3, "expected 4 values");
        expect_error("2 2\n", 1, "ends after 0");
        expect_error("1 2\n1 2\n3\n", 3, "unexpected extra value '3'");
    });

    runner.add("write then read reproduces every float32 bit", [] {
        MatrixData m;
        m.rows = 2;
        m.cols = 4;
        m.values = {0.1f, -1.0f / 3.0f, 3.4028235e38f, 1e-45f,
                    123456.789f, -0.0f, 1.0f, std::nextafter(1.0f, 2.0f)};
        std::stringstream buffer;
        write_matrix(buffer, m);
        const MatrixData back = read_matrix(buffer, "roundtrip");
        CHECK(back.rows == 2 && back.cols == 4);
        CHECK(std::memcmp(back.values.data(), m.values.data(), m.values.size() * 4) == 0);
    });

    runner.add("load_matrix reads the golden files and reports missing files", [] {
        const MatrixData x = load_matrix(data_path("golden_x.txt"));
        CHECK(x.rows == 1 && x.cols == 2);
        CHECK((x.values == std::vector<float>{1.0f, -2.0f}));
        const MatrixData w = load_matrix(data_path("golden_w.txt"));
        CHECK(w.rows == 2 && w.cols == 2);
        CHECK((w.values == std::vector<float>{0.5f, 1.0f, -1.0f, 0.25f}));
        CHECK_THROWS(load_matrix(data_path("does_not_exist.txt")), InputError);
    });

    runner.add("every invalid fixture file is rejected", [] {
        for (const char* name : {"empty.txt", "bad_header.txt", "non_numeric.txt",
                                 "nan_value.txt", "inf_value.txt", "out_of_range.txt",
                                 "truncated.txt", "extra_values.txt", "zero_dims.txt"}) {
            bool threw = false;
            try {
                (void)load_matrix(data_path(name));
            } catch (const InputError& e) {
                threw = true;
                // Messages name the file so the user knows what to fix.
                CHECK(std::string(e.what()).find(name) != std::string::npos);
            }
            if (!threw) {
                std::cerr << "  not rejected: " << name << "\n";
            }
            CHECK(threw);
        }
    });

    return runner.run();
}
