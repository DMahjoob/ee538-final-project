#include "InputValidator.h"

#include "Quantizer.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace qlinear {

namespace {

std::string dims(std::size_t rows, std::size_t cols) {
    return std::to_string(rows) + " x " + std::to_string(cols);
}

}  // namespace

void validate_dimensions(std::size_t rows, std::size_t cols, std::string_view what) {
    const std::string name(what);
    if (rows == 0 || cols == 0) {
        throw std::invalid_argument(name + " must have at least one row and one column (got " +
                                    dims(rows, cols) + ")");
    }
    if (rows > kMaxElements / cols) {
        throw std::invalid_argument(name + " is too large: " + dims(rows, cols) +
                                    " exceeds the limit of " +
                                    std::to_string(kMaxElements) + " elements");
    }
}

void validate_linear_shapes(std::size_t m, std::size_t k_x, std::size_t n,
                            std::size_t k_w) {
    validate_dimensions(m, k_x, "X");
    validate_dimensions(n, k_w, "W");
    if (k_x != k_w) {
        throw std::invalid_argument(
            "inner dimensions do not match: X is " + dims(m, k_x) + " and W is " +
            dims(n, k_w) + ", but Y = X * W^T needs X.cols == W.cols");
    }
    if (k_x > Quantizer::max_safe_inner_dim()) {
        throw std::invalid_argument(
            "inner dimension K = " + std::to_string(k_x) + " exceeds " +
            std::to_string(Quantizer::max_safe_inner_dim()) +
            ", the largest K for which INT32 accumulation of 127 * 127 products cannot "
            "overflow");
    }
}

void validate_linear_shapes(const MatrixData& x, const MatrixData& w) {
    if (x.values.size() != x.rows * x.cols) {
        throw std::invalid_argument("X holds " + std::to_string(x.values.size()) +
                                    " values but is declared " + dims(x.rows, x.cols));
    }
    if (w.values.size() != w.rows * w.cols) {
        throw std::invalid_argument("W holds " + std::to_string(w.values.size()) +
                                    " values but is declared " + dims(w.rows, w.cols));
    }
    validate_linear_shapes(x.rows, x.cols, w.rows, w.cols);
}

void validate_finite(const MatrixData& m, std::string_view what) {
    for (std::size_t i = 0; i < m.values.size(); ++i) {
        if (!std::isfinite(m.values[i])) {
            const std::size_t cols = m.cols == 0 ? 1 : m.cols;
            throw std::invalid_argument(std::string(what) + " has a non-finite value at row " +
                                        std::to_string(i / cols) + ", column " +
                                        std::to_string(i % cols));
        }
    }
}

void validate_block_size(std::size_t block) {
    if (block == 0) {
        throw std::invalid_argument("block size must be at least 1");
    }
}

std::size_t parse_size(std::string_view text, std::string_view flag) {
    const std::string name(flag);
    if (text.empty()) {
        throw std::invalid_argument(name + " needs a positive integer value");
    }
    std::size_t value = 0;
    for (char ch : text) {
        if (ch < '0' || ch > '9') {
            throw std::invalid_argument(name + " must be a positive integer, got '" +
                                        std::string(text) + "'");
        }
        const auto digit = static_cast<std::size_t>(ch - '0');
        if (value > (std::numeric_limits<std::size_t>::max() - digit) / 10) {
            throw std::invalid_argument(name + " value '" + std::string(text) +
                                        "' is too large");
        }
        value = value * 10 + digit;
    }
    if (value == 0) {
        throw std::invalid_argument(name + " must be at least 1");
    }
    return value;
}

}  // namespace qlinear
