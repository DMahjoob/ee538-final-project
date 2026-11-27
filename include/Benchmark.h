#pragma once

// Benchmark and experiment driver.
//
// run_experiments() sweeps the configuration grid from the proposal (matrix
// size, shape, input distribution, operator, block size), runs every
// configuration `warmup + reps` times through the LinearOperator interface,
// and writes ONE CSV ROW PER MEASURED REPETITION (raw data). Medians and
// spread are computed afterwards by scripts/plot_results.py, so the raw
// measurements are always kept.

#include "DataGenerator.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "Stopwatch.h"

namespace qlinear {

// Runs fn(rep) `warmup` times (results discarded), then `reps` times, and
// returns the measured results in order. fn receives the measured repetition
// index (0-based), or -1 during warm-up.
template <typename Fn>
auto collect_repeated(Fn&& fn, int warmup, int reps)
    -> std::vector<std::invoke_result_t<Fn&, int>> {
    std::vector<std::invoke_result_t<Fn&, int>> results;
    results.reserve(static_cast<std::size_t>(std::max(reps, 0)));
    for (int i = 0; i < warmup; ++i) {
        (void)fn(-1);
    }
    for (int rep = 0; rep < reps; ++rep) {
        results.push_back(fn(rep));
    }
    return results;
}

// Wall time of fn() in milliseconds for each measured repetition.
template <typename Fn>
std::vector<double> time_repeated(Fn&& fn, int warmup, int reps) {
    return collect_repeated(
        [&fn](int) {
            Stopwatch timer;
            fn();
            return timer.elapsed_ms();
        },
        warmup, reps);
}

// Order statistics of a sample. Quartiles use linear interpolation between
// closest ranks (the same definition as numpy.percentile's default).
struct Summary {
    double median = 0.0;
    double q1 = 0.0;
    double q3 = 0.0;
    double min = 0.0;
    double max = 0.0;
    double mean = 0.0;
    std::size_t n = 0;

    double iqr() const { return q3 - q1; }
};

// Throws std::invalid_argument on an empty sample.
Summary summarize(std::vector<double> samples);

enum class Shape {
    Square,     // M = K = N = size
    SingleRow,  // M = 1, K = N = size (one token, as in LLM decoding)
};

std::string_view to_string(Shape shape);
Shape parse_shape(std::string_view name);  // "square" | "single_row"

struct ExperimentConfig {
    std::vector<std::size_t> sizes;
    std::vector<std::size_t> blocks;        // only used by the "blocked" operator
    std::vector<Shape> shapes;
    std::vector<Distribution> dists;
    std::vector<std::string> operators;     // names for make_operator()
    int warmup = 1;
    int reps = 5;
    std::uint32_t seed = 538;
};

// The proposal's grid: sizes 64/256/1024/2048, blocks 16/32/64/128, square and
// single-row shapes, uniform/normal/normal_outliers, 1 warm-up + 5 reps.
ExperimentConfig default_config();

// A fast smoke test of the same pipeline: sizes 64/128, blocks 16/64,
// 1 warm-up + 2 reps.
ExperimentConfig quick_config();

// Column names, in the order run_experiments() writes them.
const std::vector<std::string>& csv_columns();

// Runs the sweep, writing a header and one row per measured repetition to
// `csv` and progress lines to `log`. Returns the number of data rows.
// Throws std::invalid_argument for an invalid config (no sizes, reps < 1, ...).
std::size_t run_experiments(const ExperimentConfig& config, std::ostream& csv,
                            std::ostream& log);

// Compiler and build-flag strings recorded in every CSV row.
std::string compiler_description();
std::string build_flags_description();

}  // namespace qlinear
