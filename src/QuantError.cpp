#include "QuantError.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace qlinear {

QuantError measure_error(const std::vector<float>& reference,
                         const std::vector<float>& approx) {
    if (reference.size() != approx.size()) {
        throw std::invalid_argument("measure_error: size mismatch");
    }
    QuantError err;
    err.count = reference.size();
    if (err.count == 0) {
        return err;
    }

    double sum_abs = 0.0;
    double sum_sq_diff = 0.0;
    double sum_sq_ref = 0.0;
    for (std::size_t i = 0; i < err.count; ++i) {
        const double ref = reference[i];
        const double diff = std::fabs(ref - static_cast<double>(approx[i]));
        err.max_abs = std::max(err.max_abs, diff);
        sum_abs += diff;
        sum_sq_diff += diff * diff;
        sum_sq_ref += ref * ref;
        if (ref != 0.0) {
            err.max_relative = std::max(err.max_relative, diff / std::fabs(ref));
        }
    }

    const double n = static_cast<double>(err.count);
    err.mean_abs = sum_abs / n;
    err.rmse = std::sqrt(sum_sq_diff / n);
    err.relative_fro = sum_sq_ref > 0.0 ? std::sqrt(sum_sq_diff / sum_sq_ref) : 0.0;
    return err;
}

}  // namespace qlinear
