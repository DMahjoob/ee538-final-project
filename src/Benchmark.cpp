#include "Benchmark.h"

#include "LinearOperator.h"
#include "QuantError.h"
#include "ReferenceKernel.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <ostream>
#include <stdexcept>
#include <string>

#ifndef QLINEAR_BUILD_FLAGS
#define QLINEAR_BUILD_FLAGS "unknown"
#endif

namespace qlinear {

namespace {

// Linear interpolation between closest ranks on a sorted sample.
double quantile(const std::vector<double>& sorted, double q) {
    const double pos = q * static_cast<double>(sorted.size() - 1);
    const auto lo = static_cast<std::size_t>(std::floor(pos));
    const auto hi = static_cast<std::size_t>(std::ceil(pos));
    const double frac = pos - static_cast<double>(lo);
    return sorted[lo] + (sorted[hi] - sorted[lo]) * frac;
}

// CSV fields never contain commas or quotes, but compiler strings might.
std::string csv_safe(std::string text) {
    std::replace(text.begin(), text.end(), ',', ';');
    std::replace(text.begin(), text.end(), '"', '\'');
    return text;
}

void validate_config(const ExperimentConfig& c) {
    if (c.sizes.empty() || c.shapes.empty() || c.dists.empty() || c.operators.empty()) {
        throw std::invalid_argument(
            "experiment config needs at least one size, shape, distribution and operator");
    }
    if (c.reps < 1 || c.warmup < 0) {
        throw std::invalid_argument("experiment config needs reps >= 1 and warmup >= 0");
    }
    for (std::size_t s : c.sizes) {
        if (s == 0) {
            throw std::invalid_argument("experiment sizes must be >= 1");
        }
    }
    const bool has_blocked =
        std::find(c.operators.begin(), c.operators.end(), "blocked") != c.operators.end();
    if (has_blocked && c.blocks.empty()) {
        throw std::invalid_argument("the blocked operator needs at least one block size");
    }
    for (std::size_t b : c.blocks) {
        if (b == 0) {
            throw std::invalid_argument("block sizes must be >= 1");
        }
    }
}

// Keeps the optimizer from discarding results that are otherwise unused.
volatile double g_sink = 0.0;

}  // namespace

Summary summarize(std::vector<double> samples) {
    if (samples.empty()) {
        throw std::invalid_argument("summarize: empty sample");
    }
    std::sort(samples.begin(), samples.end());
    Summary s;
    s.n = samples.size();
    s.min = samples.front();
    s.max = samples.back();
    s.median = quantile(samples, 0.5);
    s.q1 = quantile(samples, 0.25);
    s.q3 = quantile(samples, 0.75);
    double sum = 0.0;
    for (double v : samples) {
        sum += v;
    }
    s.mean = sum / static_cast<double>(s.n);
    return s;
}

std::string_view to_string(Shape shape) {
    switch (shape) {
        case Shape::Square: return "square";
        case Shape::SingleRow: return "single_row";
    }
    return "unknown";
}

Shape parse_shape(std::string_view name) {
    if (name == "square") return Shape::Square;
    if (name == "single_row") return Shape::SingleRow;
    throw std::invalid_argument("unknown shape '" + std::string(name) +
                                "' (expected square or single_row)");
}

ExperimentConfig default_config() {
    ExperimentConfig c;
    c.sizes = {64, 256, 1024, 2048};
    c.blocks = {16, 32, 64, 128};
    c.shapes = {Shape::Square, Shape::SingleRow};
    c.dists = {Distribution::Uniform, Distribution::Normal, Distribution::NormalOutliers};
    c.operators = available_operators();
    c.warmup = 1;
    c.reps = 5;
    return c;
}

ExperimentConfig quick_config() {
    ExperimentConfig c = default_config();
    c.sizes = {64, 128};
    c.blocks = {16, 64};
    c.dists = {Distribution::Uniform, Distribution::NormalOutliers};
    c.reps = 2;
    return c;
}

const std::vector<std::string>& csv_columns() {
    static const std::vector<std::string> columns = {
        "seed",          "dist",        "shape",      "m",           "k",
        "n",             "operator",    "block",      "rep",         "prepare_ms",
        "quantize_x_ms", "compute_ms",  "dequantize_ms", "total_ms", "overhead_ms",
        "weight_bytes",  "max_abs_err", "rel_fro_err", "compiler",   "build_flags"};
    return columns;
}

std::string compiler_description() {
#if defined(__clang__)
    return csv_safe("clang " __clang_version__);
#elif defined(__GNUC__)
    return csv_safe("gcc " __VERSION__);
#else
    return "unknown";
#endif
}

std::string build_flags_description() {
    return csv_safe(QLINEAR_BUILD_FLAGS);
}

std::size_t run_experiments(const ExperimentConfig& config, std::ostream& csv,
                            std::ostream& log) {
    validate_config(config);
    const std::string compiler = compiler_description();
    const std::string flags = build_flags_description();

    const auto& columns = csv_columns();
    for (std::size_t i = 0; i < columns.size(); ++i) {
        csv << (i == 0 ? "" : ",") << columns[i];
    }
    csv << '\n';
    csv << std::setprecision(9);

    std::size_t rows_written = 0;
    for (Distribution dist : config.dists) {
        for (Shape shape : config.shapes) {
            for (std::size_t size : config.sizes) {
                const std::size_t m = shape == Shape::Square ? size : 1;
                const std::size_t k = size;
                const std::size_t n = size;

                // Same seed for every configuration: a given (dist, size)
                // always produces the same matrices, whatever else runs.
                DataGenerator gen(config.seed);
                const DistributionSpec spec = DistributionSpec::defaults(dist);
                const std::vector<float> x = gen.generate(spec, m, k);
                const std::vector<float> w = gen.generate(spec, n, k);
                const std::vector<float> y_ref = reference_linear(x, w, m, k, n);

                for (const std::string& op_name : config.operators) {
                    // Block size only changes the blocked operator, so the
                    // others run once per configuration (block = 0 in CSV).
                    const std::vector<std::size_t> blocks =
                        op_name == "blocked" ? config.blocks : std::vector<std::size_t>{0};
                    for (std::size_t block : blocks) {
                        auto op = make_operator(op_name, block == 0 ? 1 : block);
                        log << to_string(dist) << ' ' << to_string(shape) << ' ' << m << 'x'
                            << k << 'x' << n << ' ' << op_name;
                        if (block != 0) {
                            log << " block=" << block;
                        }
                        log << " ..." << std::flush;

                        struct Sample {
                            StageTimes times;
                            QuantError error;
                        };
                        const auto samples = collect_repeated(
                            [&](int) {
                                op->prepare(w, n, k);
                                const std::vector<float> y = op->forward(x, m);
                                g_sink = g_sink + static_cast<double>(y.front()) +
                                         static_cast<double>(y.back());
                                return Sample{op->last_times(), measure_error(y_ref, y)};
                            },
                            config.warmup, config.reps);

                        std::vector<double> totals;
                        for (std::size_t rep = 0; rep < samples.size(); ++rep) {
                            const Sample& s = samples[rep];
                            totals.push_back(s.times.total_ms);
                            csv << config.seed << ',' << to_string(dist) << ','
                                << to_string(shape) << ',' << m << ',' << k << ',' << n << ','
                                << op_name << ',' << block << ',' << rep << ','
                                << s.times.prepare_ms << ',' << s.times.quantize_x_ms << ','
                                << s.times.compute_ms << ',' << s.times.dequantize_ms << ','
                                << s.times.total_ms << ',' << s.times.overhead_ms() << ','
                                << op->weight_storage_bytes() << ',' << s.error.max_abs << ','
                                << s.error.relative_fro << ',' << compiler << ',' << flags
                                << '\n';
                            ++rows_written;
                        }
                        const Summary total = summarize(totals);
                        log << " median " << total.median << " ms (IQR " << total.iqr()
                            << ")\n";
                    }
                }
            }
        }
    }
    csv.flush();
    return rows_written;
}

}  // namespace qlinear
