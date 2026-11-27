#pragma once

// Semantic checks on inputs, run before any computation so that the user gets
// one clear message instead of an exception from deep inside a kernel. Every
// function throws std::invalid_argument with a message that says what to fix.

#include "MatrixIO.h"

#include <cstddef>
#include <string_view>

namespace qlinear {

// Largest matrix (in elements) the program accepts: 2^26 floats = 256 MiB.
// Protects against allocating huge buffers from a typo in a file header.
inline constexpr std::size_t kMaxElements = std::size_t{1} << 26;

// rows and cols must be positive and rows * cols <= kMaxElements.
// `what` names the matrix in the message ("X", "W", "x.txt").
void validate_dimensions(std::size_t rows, std::size_t cols, std::string_view what);

// Shape and size checks for Y = X * W^T with X: M x K and W: N x K.
//  - both matrices non-empty and values.size() == rows * cols
//  - inner dimensions match (X.cols == W.cols)
//  - K <= Quantizer::max_safe_inner_dim() so INT32 accumulation cannot overflow
void validate_linear_shapes(std::size_t m, std::size_t k_x, std::size_t n,
                            std::size_t k_w);
void validate_linear_shapes(const MatrixData& x, const MatrixData& w);

// Every value is finite (no NaN or +/-Inf).
void validate_finite(const MatrixData& m, std::string_view what);

// Block size for the blocked kernel: must be > 0. A block larger than the
// matrix is allowed (the kernel then uses one partial block).
void validate_block_size(std::size_t block);

// Parses a positive decimal integer for a command-line flag, e.g.
// parse_size("64", "--block"). Rejects signs, spaces, fractions, zero and
// values that overflow size_t.
std::size_t parse_size(std::string_view text, std::string_view flag);

}  // namespace qlinear
