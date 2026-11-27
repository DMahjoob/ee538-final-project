#include "LinearOperator.h"

#include "InputValidator.h"
#include "ReferenceKernel.h"
#include "Stopwatch.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace qlinear {

double StageTimes::overhead_ms() const {
    return std::max(0.0, total_ms - (quantize_x_ms + compute_ms + dequantize_ms));
}

void LinearOperator::prepare(const std::vector<float>& w, std::size_t n, std::size_t k) {
    // Reuse the X/W shape check with a 1-row X so the messages and the INT32
    // overflow bound are identical to the CLI's.
    validate_linear_shapes(1, k, n, k);
    if (w.size() != n * k) {
        throw std::invalid_argument("prepare: W holds " + std::to_string(w.size()) +
                                    " values, expected n*k = " + std::to_string(n * k));
    }
    prepared_ = false;
    n_ = n;
    k_ = k;
    times_ = StageTimes{};
    Stopwatch timer;
    do_prepare(w);
    times_.prepare_ms = timer.elapsed_ms();
    prepared_ = true;
}

std::vector<float> LinearOperator::forward(const std::vector<float>& x, std::size_t m) {
    if (!prepared_) {
        throw std::logic_error(name() + ": forward() called before prepare()");
    }
    if (m == 0) {
        throw std::invalid_argument(name() + ": X must have at least one row");
    }
    if (x.size() != m * k_) {
        throw std::invalid_argument(name() + ": X holds " + std::to_string(x.size()) +
                                    " values, expected m*k = " + std::to_string(m) + "*" +
                                    std::to_string(k_));
    }
    StageTimes stages;
    stages.prepare_ms = times_.prepare_ms;
    Stopwatch timer;
    std::vector<float> y = do_forward(x, m, stages);
    stages.total_ms = timer.elapsed_ms();
    times_ = stages;
    return y;
}

std::size_t ReferenceLinear::weight_storage_bytes() const {
    return w_.size() * sizeof(float);
}

void ReferenceLinear::do_prepare(const std::vector<float>& w) {
    w_ = w;
}

std::vector<float> ReferenceLinear::do_forward(const std::vector<float>& x, std::size_t m,
                                               StageTimes& times) {
    Stopwatch timer;
    std::vector<float> y = reference_linear(x, w_, m, in_features(), out_features());
    times.compute_ms = timer.elapsed_ms();
    return y;
}

std::unique_ptr<LinearOperator> make_operator(std::string_view kind, std::size_t block) {
    validate_block_size(block);
    if (kind == "reference") {
        return std::make_unique<ReferenceLinear>();
    }
    if (kind == "scalar" || kind == "blocked") {
        // Wired in by Int8Linear once Jenny's ScalarKernel/BlockedKernel land.
        throw std::runtime_error("the INT8 '" + std::string(kind) +
                                 "' operator is not integrated yet (waiting for the "
                                 "ScalarKernel/BlockedKernel implementations)");
    }
    throw std::invalid_argument("unknown operator '" + std::string(kind) +
                                "' (expected reference, scalar or blocked)");
}

std::vector<std::string> available_operators() {
    return {"reference"};
}

}  // namespace qlinear
