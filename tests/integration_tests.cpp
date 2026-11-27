// Integration tests: the LinearOperator interface (virtual dispatch, stage
// timing, error handling), the accuracy contract, and the benchmark driver.
//
// Phase B (when Jenny's kernels land) adds: golden INT32 accumulators
// [-12033, -12192], scalar == blocked bit-for-bit, and the accuracy bounds
// applied to the INT8 operators.

#include "Benchmark.h"
#include "DataGenerator.h"
#include "LinearOperator.h"
#include "QuantError.h"
#include "Quantizer.h"
#include "ReferenceKernel.h"
#include "test_support.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace qlinear;

namespace {

// Per-tensor symmetric INT8 round trip of X and W followed by an exact
// INT32-style product, built only from Darius's Quantizer. Used to check the
// accuracy contract before the real INT8 kernels are integrated; it is NOT a
// kernel under test.
std::vector<float> int8_emulation(const std::vector<float>& x, const std::vector<float>& w,
                                  std::size_t m, std::size_t k, std::size_t n) {
    const Quantizer quantizer;
    const QuantizedTensor qx = quantizer.quantize(x);
    const QuantizedTensor qw = quantizer.quantize(w);
    std::vector<std::int32_t> acc(m * n, 0);
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            std::int32_t sum = 0;
            for (std::size_t p = 0; p < k; ++p) {
                sum += static_cast<std::int32_t>(qx.values[i * k + p]) * qw.values[j * k + p];
            }
            acc[i * n + j] = sum;
        }
    }
    return Quantizer::dequantize_accumulator(acc, qx.scale, qw.scale);
}

double int8_relative_error(Distribution dist, std::size_t size) {
    DataGenerator gen(538);
    const auto spec = DistributionSpec::defaults(dist);
    const auto x = gen.generate(spec, size, size);
    const auto w = gen.generate(spec, size, size);
    const auto y_ref = reference_linear(x, w, size, size, size);
    return measure_error(y_ref, int8_emulation(x, w, size, size, size)).relative_fro;
}

}  // namespace

int main() {
    test::Runner runner;

    runner.add("operators are used through the LinearOperator base class", [] {
        std::unique_ptr<LinearOperator> op = make_operator("reference", 64);
        CHECK(op->name() == "reference");
        CHECK(!op->prepared());
        op->prepare({0.5f, 1.0f, -1.0f, 0.25f}, 2, 2);
        CHECK(op->prepared());
        CHECK(op->out_features() == 2 && op->in_features() == 2);
        const std::vector<float> y = op->forward({1.0f, -2.0f}, 1);
        CHECK((y == std::vector<float>{-1.5f, -1.5f}));
        CHECK(op->weight_storage_bytes() == 4 * sizeof(float));
    });

    runner.add("forward agrees with reference_linear on random non-square input", [] {
        DataGenerator gen(5);
        const auto spec = DistributionSpec::defaults(Distribution::Normal);
        const std::size_t m = 7, k = 33, n = 12;
        const auto x = gen.generate(spec, m, k);
        const auto w = gen.generate(spec, n, k);
        auto op = make_operator("reference", 16);
        op->prepare(w, n, k);
        CHECK(op->forward(x, m) == reference_linear(x, w, m, k, n));
        // The operator can be reused with a different batch size.
        CHECK(op->forward(gen.generate(spec, 1, k), 1).size() == n);
    });

    runner.add("stage times are recorded and consistent", [] {
        DataGenerator gen(9);
        const auto spec = DistributionSpec::defaults(Distribution::Uniform);
        auto op = make_operator("reference", 64);
        op->prepare(gen.generate(spec, 128, 128), 128, 128);
        (void)op->forward(gen.generate(spec, 64, 128), 64);
        const StageTimes& t = op->last_times();
        CHECK(t.prepare_ms >= 0.0);
        CHECK(t.compute_ms > 0.0);
        CHECK(t.total_ms >= t.compute_ms);
        CHECK(t.overhead_ms() >= 0.0);
        CHECK(t.quantize_x_ms == 0.0 && t.dequantize_ms == 0.0);  // float32 path
    });

    runner.add("misuse is rejected", [] {
        auto op = make_operator("reference", 64);
        CHECK_THROWS(op->forward({1.0f}, 1), std::logic_error);  // before prepare()
        CHECK_THROWS(op->prepare({1.0f, 2.0f, 3.0f}, 2, 2), std::invalid_argument);
        CHECK_THROWS(op->prepare({}, 0, 2), std::invalid_argument);
        op->prepare({1.0f, 2.0f}, 1, 2);
        CHECK_THROWS(op->forward({1.0f, 2.0f, 3.0f}, 1), std::invalid_argument);
        CHECK_THROWS(op->forward({}, 0), std::invalid_argument);
        CHECK_THROWS(op->prepare(std::vector<float>(Quantizer::max_safe_inner_dim() + 1),
                                 1, Quantizer::max_safe_inner_dim() + 1),
                     std::invalid_argument);
    });

    runner.add("make_operator validates its arguments", [] {
        CHECK_THROWS(make_operator("fast", 64), std::invalid_argument);
        CHECK_THROWS(make_operator("reference", 0), std::invalid_argument);
        for (const std::string& name : available_operators()) {
            CHECK(make_operator(name, 32)->name().size() > 0);
        }
    });

    runner.add("accuracy contract: uniform < 1%, normal < 2.5%", [] {
        // Per-tensor INT8 error grows with size for Gaussian inputs (the
        // largest |x| grows, so the quantization step does too); see README.
        for (std::size_t size : {64, 256}) {
            const double uniform = int8_relative_error(Distribution::Uniform, size);
            const double normal = int8_relative_error(Distribution::Normal, size);
            std::cout << "  size " << size << ": uniform " << uniform * 100 << "%, normal "
                      << normal * 100 << "%\n";
            CHECK(uniform < 0.01);
            CHECK(normal < 0.025);
        }
    });

    runner.add("all-zero input gives zero output and zero error", [] {
        const std::vector<float> zeros(16, 0.0f);
        const auto y = int8_emulation(zeros, zeros, 4, 4, 4);
        CHECK((y == std::vector<float>(16, 0.0f)));
    });

    runner.add("summarize computes order statistics", [] {
        const Summary s = summarize({5.0, 1.0, 3.0, 2.0, 4.0});
        CHECK(s.n == 5);
        CHECK(s.median == 3.0 && s.min == 1.0 && s.max == 5.0 && s.mean == 3.0);
        CHECK(s.q1 == 2.0 && s.q3 == 4.0 && s.iqr() == 2.0);
        const Summary even = summarize({1.0, 2.0, 3.0, 4.0});
        CHECK(even.median == 2.5);
        CHECK_THROWS(summarize({}), std::invalid_argument);
    });

    runner.add("collect_repeated runs warm-up then measured repetitions", [] {
        int calls = 0;
        const auto reps = collect_repeated([&](int rep) { ++calls; return rep; }, 2, 3);
        CHECK(calls == 5);
        CHECK((reps == std::vector<int>{0, 1, 2}));
        const auto times = time_repeated([] {}, 0, 4);
        CHECK(times.size() == 4);
    });

    runner.add("benchmark driver writes one CSV row per repetition", [] {
        ExperimentConfig config = quick_config();
        config.sizes = {4, 8};
        config.reps = 3;
        config.operators = {"reference"};
        std::ostringstream csv;
        std::ostringstream log;
        const std::size_t rows = run_experiments(config, csv, log);
        // 2 dists x 2 shapes x 2 sizes x 1 operator x 3 reps
        CHECK(rows == 24);
        std::istringstream lines(csv.str());
        std::string header;
        std::getline(lines, header);
        CHECK(header.rfind("seed,dist,shape,m,k,n,operator,block,rep,", 0) == 0);
        std::size_t data_lines = 0;
        std::string line;
        while (std::getline(lines, line)) {
            ++data_lines;
            CHECK(std::count(line.begin(), line.end(), ',') ==
                  static_cast<long>(csv_columns().size() - 1));
        }
        CHECK(data_lines == rows);
    });

    runner.add("benchmark driver rejects invalid configs", [] {
        std::ostringstream sink;
        ExperimentConfig c = quick_config();
        c.reps = 0;
        CHECK_THROWS(run_experiments(c, sink, sink), std::invalid_argument);
        c = quick_config();
        c.sizes.clear();
        CHECK_THROWS(run_experiments(c, sink, sink), std::invalid_argument);
    });

    return runner.run();
}
