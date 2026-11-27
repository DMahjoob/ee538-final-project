#include "ReferenceKernel.h"

#include <stdexcept>
#include <string>

namespace qlinear {

std::vector<float> reference_linear(const std::vector<float>& x,
                                    const std::vector<float>& w, std::size_t m,
                                    std::size_t k, std::size_t n) {
    if (x.size() != m * k) {
        throw std::invalid_argument("reference_linear: X has " + std::to_string(x.size()) +
                                    " values, expected m*k = " + std::to_string(m * k));
    }
    if (w.size() != n * k) {
        throw std::invalid_argument("reference_linear: W has " + std::to_string(w.size()) +
                                    " values, expected n*k = " + std::to_string(n * k));
    }
    std::vector<float> y(m * n, 0.0f);
    for (std::size_t i = 0; i < m; ++i) {
        const float* x_row = x.data() + i * k;
        for (std::size_t j = 0; j < n; ++j) {
            const float* w_row = w.data() + j * k;
            double acc = 0.0;
            for (std::size_t p = 0; p < k; ++p) {
                acc += static_cast<double>(x_row[p]) * static_cast<double>(w_row[p]);
            }
            y[i * n + j] = static_cast<float>(acc);
        }
    }
    return y;
}

}  // namespace qlinear
