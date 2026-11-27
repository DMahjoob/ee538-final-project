#pragma once

// Common interface for every implementation of Y = X * W^T.
//
// All operators take float32 X and W and return float32 Y, so the float32
// reference and the INT8 operators are compared on exactly the same task, and
// the INT8 conversion costs (quantize, dequantize) are part of what is timed.
//
// Usage:
//     auto op = make_operator("reference", 64);
//     op->prepare(w, n, k);           // one-time weight setup (timed)
//     std::vector<float> y = op->forward(x, m);
//     const StageTimes& t = op->last_times();
//
// Design: non-virtual interface (NVI). The public prepare()/forward() check
// shapes and time the call once, for every implementation; subclasses only
// implement the protected do_prepare()/do_forward() hooks.

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace qlinear {

// Milliseconds spent in each stage of the most recent prepare()/forward().
// Stages an implementation does not have stay 0.
struct StageTimes {
    double prepare_ms = 0.0;     // weight preprocessing (quantize + pack W)
    double quantize_x_ms = 0.0;  // per-call activation quantization
    double compute_ms = 0.0;     // the matrix multiplication itself
    double dequantize_ms = 0.0;  // INT32 accumulator -> float32
    double total_ms = 0.0;       // wall time of the whole forward() call

    // total_ms minus the measured stages: validation, allocation, call
    // overhead. Never negative.
    double overhead_ms() const;
};

class LinearOperator {
public:
    virtual ~LinearOperator() = default;

    // Polymorphic base: copying would slice, so it is disabled.
    LinearOperator(const LinearOperator&) = delete;
    LinearOperator& operator=(const LinearOperator&) = delete;

    // Short identifier used in CSV output: "reference", "int8_scalar", ...
    virtual std::string name() const = 0;

    // Stores (and, for INT8, quantizes/packs) the weight matrix W (n x k,
    // row-major). Must be called before forward(). Throws
    // std::invalid_argument if w.size() != n * k, n or k is 0, or k exceeds
    // the INT32-safe inner dimension.
    void prepare(const std::vector<float>& w, std::size_t n, std::size_t k);

    // Computes Y = X * W^T for X (m x k, row-major); returns Y (m x n).
    // Throws std::logic_error if prepare() was not called, and
    // std::invalid_argument if x.size() != m * k or m is 0.
    std::vector<float> forward(const std::vector<float>& x, std::size_t m);

    bool prepared() const { return prepared_; }
    std::size_t out_features() const { return n_; }  // N
    std::size_t in_features() const { return k_; }   // K

    // Times of the last prepare() (prepare_ms) and forward() (the rest).
    const StageTimes& last_times() const { return times_; }

    // Bytes used to store the prepared weights, computed from the data
    // structure (an explicit estimate, not measured peak memory).
    virtual std::size_t weight_storage_bytes() const = 0;

protected:
    LinearOperator() = default;

    // w is already validated to be n_ x k_.
    virtual void do_prepare(const std::vector<float>& w) = 0;

    // x is already validated to be m x k_. Implementations record their
    // quantize/compute/dequantize stage times in `times`.
    virtual std::vector<float> do_forward(const std::vector<float>& x, std::size_t m,
                                          StageTimes& times) = 0;

private:
    std::size_t n_ = 0;
    std::size_t k_ = 0;
    bool prepared_ = false;
    StageTimes times_;
};

// The float32 baseline: stores W and calls reference_linear().
class ReferenceLinear final : public LinearOperator {
public:
    ReferenceLinear() = default;

    std::string name() const override { return "reference"; }
    std::size_t weight_storage_bytes() const override;

protected:
    void do_prepare(const std::vector<float>& w) override;
    std::vector<float> do_forward(const std::vector<float>& x, std::size_t m,
                                  StageTimes& times) override;

private:
    std::vector<float> w_;
};

// Creates an operator by name: "reference", "scalar" (INT8 scalar kernel) or
// "blocked" (INT8 blocked kernel with the given block size). Throws
// std::invalid_argument for an unknown name or a block size of 0.
std::unique_ptr<LinearOperator> make_operator(std::string_view kind, std::size_t block);

// Names accepted by make_operator(), in benchmark order.
std::vector<std::string> available_operators();

}  // namespace qlinear
