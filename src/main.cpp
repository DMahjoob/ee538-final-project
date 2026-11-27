// qlinear: command-line driver for the quantized linear operator.
//
//   qlinear gen   --rows R --cols C --dist NAME [--seed S] --out FILE
//   qlinear run   --x X.txt --w W.txt [--kernel NAME|all] [--block B] [--out Y.txt]
//   qlinear bench [--config default|quick] [overrides...] --out FILE.csv
//
// Exit codes: 0 success, 1 invalid input or runtime error, 2 usage error.

#include "Benchmark.h"
#include "DataGenerator.h"
#include "InputValidator.h"
#include "LinearOperator.h"
#include "MatrixIO.h"
#include "QuantError.h"
#include "ReferenceKernel.h"

#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using namespace qlinear;

namespace {

constexpr int kExitOk = 0;
constexpr int kExitError = 1;
constexpr int kExitUsage = 2;

// Matrices up to this many elements are printed by `run`.
constexpr std::size_t kPrintLimit = 64;

const char* const kUsage =
    "usage:\n"
    "  qlinear gen   --rows R --cols C --dist NAME [--seed S] --out FILE\n"
    "                NAME: uniform | normal | normal_outliers | zeros\n"
    "  qlinear run   --x X.txt --w W.txt [--kernel reference|scalar|blocked|all]\n"
    "                [--block B] [--out Y.txt]\n"
    "                computes Y = X * W^T and reports its error against the float32\n"
    "                reference; --out needs a single --kernel\n"
    "  qlinear bench [--config default|quick] [--reps N] [--warmup N] [--seed S]\n"
    "                [--sizes 64,256] [--blocks 16,64] [--shapes square,single_row]\n"
    "                [--dists uniform,normal] [--operators reference,blocked]\n"
    "                --out FILE.csv   (use '-' for stdout; progress goes to stderr)\n"
    "  qlinear help\n"
    "\n"
    "matrix files: optional '#' comments, a '<rows> <cols>' header, then rows*cols\n"
    "numbers (row-major).\n";

class UsageError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// --flag value pairs after the subcommand.
class Args {
public:
    Args(int argc, char** argv, int first, const std::set<std::string>& allowed) {
        for (int i = first; i < argc; ++i) {
            const std::string flag = argv[i];
            if (flag.rfind("--", 0) != 0) {
                throw UsageError("unexpected argument '" + flag + "'");
            }
            if (allowed.count(flag) == 0) {
                throw UsageError("unknown option '" + flag + "'");
            }
            if (i + 1 >= argc) {
                throw UsageError(flag + " needs a value");
            }
            if (!values_.emplace(flag, argv[++i]).second) {
                throw UsageError(flag + " was given more than once");
            }
        }
    }

    bool has(const std::string& flag) const { return values_.count(flag) != 0; }

    const std::string& required(const std::string& flag) const {
        const auto it = values_.find(flag);
        if (it == values_.end()) {
            throw UsageError("missing required option " + flag);
        }
        return it->second;
    }

    std::string get(const std::string& flag, const std::string& fallback) const {
        const auto it = values_.find(flag);
        return it == values_.end() ? fallback : it->second;
    }

private:
    std::map<std::string, std::string> values_;
};

std::vector<std::string> split_list(const std::string& text, const std::string& flag) {
    std::vector<std::string> items;
    std::string current;
    for (char ch : text + ",") {
        if (ch == ',') {
            if (current.empty()) {
                throw std::invalid_argument(flag + " has an empty item in '" + text + "'");
            }
            items.push_back(current);
            current.clear();
        } else {
            current += ch;
        }
    }
    return items;
}

std::vector<std::size_t> parse_size_list(const std::string& text, const std::string& flag) {
    std::vector<std::size_t> out;
    for (const std::string& item : split_list(text, flag)) {
        out.push_back(parse_size(item, flag));
    }
    return out;
}

std::uint32_t parse_seed(const std::string& text) {
    if (text == "0") {
        return 0;
    }
    const std::size_t value = parse_size(text, "--seed");
    if (value > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("--seed must fit in 32 bits");
    }
    return static_cast<std::uint32_t>(value);
}

int cmd_gen(int argc, char** argv) {
    const Args args(argc, argv, 2, {"--rows", "--cols", "--dist", "--seed", "--out"});
    const std::size_t rows = parse_size(args.required("--rows"), "--rows");
    const std::size_t cols = parse_size(args.required("--cols"), "--cols");
    validate_dimensions(rows, cols, "generated matrix");
    const Distribution dist = parse_distribution(args.required("--dist"));
    if (dist == Distribution::ChannelOutliers || dist == Distribution::Constant) {
        throw std::invalid_argument("gen supports uniform, normal, normal_outliers and zeros");
    }
    const std::uint32_t seed = parse_seed(args.get("--seed", "538"));
    const std::string out = args.required("--out");

    DataGenerator gen(seed);
    MatrixData m;
    m.rows = rows;
    m.cols = cols;
    m.values = gen.generate(DistributionSpec::defaults(dist), rows, cols);
    if (out == "-") {
        write_matrix(std::cout, m);
    } else {
        save_matrix(out, m);
        std::cerr << "wrote " << rows << " x " << cols << ' ' << to_string(dist)
                  << " matrix (seed " << seed << ") to " << out << '\n';
    }
    return kExitOk;
}

void print_matrix(const std::vector<float>& values, std::size_t rows, std::size_t cols) {
    if (rows * cols > kPrintLimit) {
        std::cout << "  (" << rows << " x " << cols << " result not printed; use --out)\n";
        return;
    }
    for (std::size_t r = 0; r < rows; ++r) {
        std::cout << " ";
        for (std::size_t c = 0; c < cols; ++c) {
            std::cout << ' ' << std::setw(12) << values[r * cols + c];
        }
        std::cout << '\n';
    }
}

int cmd_run(int argc, char** argv) {
    const Args args(argc, argv, 2, {"--x", "--w", "--kernel", "--block", "--out"});
    const MatrixData x = load_matrix(args.required("--x"));
    const MatrixData w = load_matrix(args.required("--w"));
    validate_linear_shapes(x, w);
    const std::size_t block = parse_size(args.get("--block", "64"), "--block");
    validate_block_size(block);

    const std::string kernel = args.get("--kernel", "all");
    const std::vector<std::string> kernels =
        kernel == "all" ? available_operators() : std::vector<std::string>{kernel};
    if (args.has("--out") && kernels.size() != 1) {
        throw UsageError("--out needs a single --kernel, not 'all'");
    }

    const std::size_t m = x.rows;
    const std::size_t k = x.cols;
    const std::size_t n = w.rows;
    const std::vector<float> y_ref = reference_linear(x.values, w.values, m, k, n);

    std::cout << "Y = X * W^T with X " << m << " x " << k << ", W " << n << " x " << k
              << ", Y " << m << " x " << n << '\n';
    std::cout << std::setprecision(7);
    for (const std::string& name : kernels) {
        auto op = make_operator(name, block);
        op->prepare(w.values, n, k);
        const std::vector<float> y = op->forward(x.values, m);
        const QuantError err = measure_error(y_ref, y);
        const StageTimes& t = op->last_times();

        std::cout << '\n' << op->name();
        if (name == "blocked") {
            std::cout << " (block " << block << ")";
        }
        std::cout << ":\n";
        print_matrix(y, m, n);
        std::cout << "  error vs float32 reference: max_abs " << err.max_abs
                  << ", relative_fro " << err.relative_fro << '\n';
        std::cout << "  time (ms): prepare " << t.prepare_ms << ", quantize_x "
                  << t.quantize_x_ms << ", compute " << t.compute_ms << ", dequantize "
                  << t.dequantize_ms << ", total " << t.total_ms << '\n';
        std::cout << "  weight storage: " << op->weight_storage_bytes() << " bytes\n";

        if (args.has("--out")) {
            MatrixData out;
            out.rows = m;
            out.cols = n;
            out.values = y;
            save_matrix(args.required("--out"), out);
            std::cout << "  wrote Y to " << args.required("--out") << '\n';
        }
    }
    return kExitOk;
}

int cmd_bench(int argc, char** argv) {
    const Args args(argc, argv, 2,
                    {"--config", "--reps", "--warmup", "--seed", "--sizes", "--blocks",
                     "--shapes", "--dists", "--operators", "--out"});
    const std::string preset = args.get("--config", "default");
    ExperimentConfig config;
    if (preset == "default") {
        config = default_config();
    } else if (preset == "quick") {
        config = quick_config();
    } else {
        throw UsageError("--config must be default or quick, got '" + preset + "'");
    }
    if (args.has("--reps")) {
        config.reps = static_cast<int>(parse_size(args.required("--reps"), "--reps"));
    }
    if (args.has("--warmup")) {
        const std::string text = args.required("--warmup");
        config.warmup = text == "0" ? 0 : static_cast<int>(parse_size(text, "--warmup"));
    }
    if (args.has("--seed")) {
        config.seed = parse_seed(args.required("--seed"));
    }
    if (args.has("--sizes")) {
        config.sizes = parse_size_list(args.required("--sizes"), "--sizes");
    }
    if (args.has("--blocks")) {
        config.blocks = parse_size_list(args.required("--blocks"), "--blocks");
    }
    if (args.has("--shapes")) {
        config.shapes.clear();
        for (const std::string& s : split_list(args.required("--shapes"), "--shapes")) {
            config.shapes.push_back(parse_shape(s));
        }
    }
    if (args.has("--dists")) {
        config.dists.clear();
        for (const std::string& d : split_list(args.required("--dists"), "--dists")) {
            config.dists.push_back(parse_distribution(d));
        }
    }
    if (args.has("--operators")) {
        config.operators = split_list(args.required("--operators"), "--operators");
        for (const std::string& name : config.operators) {
            (void)make_operator(name, 1);  // fail fast on unknown/unavailable names
        }
    }

    const std::string out = args.required("--out");
    std::size_t rows = 0;
    if (out == "-") {
        rows = run_experiments(config, std::cout, std::cerr);
    } else {
        std::ofstream file(out);
        if (!file) {
            throw std::runtime_error(out + ": cannot open file for writing");
        }
        rows = run_experiments(config, file, std::cerr);
        if (!file) {
            throw std::runtime_error(out + ": write failed");
        }
    }
    std::cerr << "wrote " << rows << " measurements to " << out << '\n';
    return kExitOk;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << kUsage;
        return kExitUsage;
    }
    const std::string command = argv[1];
    try {
        if (command == "gen") return cmd_gen(argc, argv);
        if (command == "run") return cmd_run(argc, argv);
        if (command == "bench") return cmd_bench(argc, argv);
        if (command == "help" || command == "--help" || command == "-h") {
            std::cout << kUsage;
            return kExitOk;
        }
        throw UsageError("unknown command '" + command + "'");
    } catch (const UsageError& e) {
        std::cerr << "usage error: " << e.what() << "\n\n" << kUsage;
        return kExitUsage;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return kExitError;
    }
}
