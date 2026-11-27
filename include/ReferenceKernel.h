#pragma once

// Independent float32 reference for Y = X * W^T.
//
// Deliberately the simplest correct algorithm (a naive triple loop) and shares
// no code with the quantized path, so it can serve as the ground truth for
// numerical error. Products are summed in double and rounded to float once,
// so the reference itself contributes negligible rounding error.

#include <cstddef>
#include <vector>

namespace qlinear {

// X is m x k, W is n x k, both row-major. Returns Y (m x n, row-major) with
// Y[i][j] = sum_p X[i][p] * W[j][p].
// Throws std::invalid_argument if x.size() != m*k or w.size() != n*k.
// Time O(m*n*k), extra space O(m*n).
std::vector<float> reference_linear(const std::vector<float>& x,
                                    const std::vector<float>& w, std::size_t m,
                                    std::size_t k, std::size_t n);

}  // namespace qlinear
