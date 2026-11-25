// Quantization error analysis: sweeps input distributions, scale methods and
// sizes, and writes one CSV row per (distribution, method, size).
//
// Usage: quant_error_analysis [output.csv] [seed]
//
// The INT8 matmul here is a deliberately simple local loop so the analysis
// runs on its own; the scalar/blocked kernels produce identical INT32 sums.

#include "QuantError.h"
#include "Quantizer.h"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using qlinear::measure_error;
using qlinear::QuantError;
using qlinear::Quantizer;
using qlinear::ScaleMethod;

namespace {

enum class Dist { Uniform, Normal, NormalOutliers, Laplace, Sparse, Zeros };

const char* dist_name(Dist d) {
    switch (d) {
        case Dist::Uniform: return "uniform";
        case Dist::Normal: return "normal";
        case Dist::NormalOutliers: return "normal_outliers";
        case Dist::Laplace: return "laplace";
        case Dist::Sparse: return "sparse";
        case Dist::Zeros: return "zeros";
    }
    return "unknown";
}

std::vector<float> generate(Dist d, std::size_t n, std::mt19937& rng) {
    std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::exponential_distribution<float> expo(1.0f);
    std::bernoulli_distribution coin(0.5);
    std::bernoulli_distribution rare(0.001);   // outlier rate
    std::bernoulli_distribution dense(0.1);    // sparse: 10% nonzero
    std::vector<float> out(n, 0.0f);
    for (float& x : out) {
        switch (d) {
            case Dist::Uniform: x = uniform(rng); break;
            case Dist::Normal: x = normal(rng); break;
            case Dist::NormalOutliers:
                x = normal(rng) * (rare(rng) ? 50.0f : 1.0f);
                break;
            case Dist::Laplace: x = expo(rng) * (coin(rng) ? 1.0f : -1.0f); break;
            case Dist::Sparse: x = dense(rng) ? normal(rng) : 0.0f; break;
            case Dist::Zeros: break;
        }
    }
    return out;
}

// Y[m][n] = sum_k X[m][k] * W[n][k]  (row-major X: MxK, W: NxK)
std::vector<float> matmul_f32(const std::vector<float>& x, const std::vector<float>& w,
                              std::size_t m, std::size_t n, std::size_t k) {
    std::vector<float> y(m * n, 0.0f);
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            double acc = 0.0;
            for (std::size_t p = 0; p < k; ++p) {
                acc += static_cast<double>(x[i * k + p]) * w[j * k + p];
            }
            y[i * n + j] = static_cast<float>(acc);
        }
    }
    return y;
}

std::vector<std::int32_t> matmul_i8(const std::vector<std::int8_t>& x,
                                    const std::vector<std::int8_t>& w,
                                    std::size_t m, std::size_t n, std::size_t k) {
    std::vector<std::int32_t> y(m * n, 0);
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            std::int32_t acc = 0;
            for (std::size_t p = 0; p < k; ++p) {
                acc += static_cast<std::int32_t>(x[i * k + p]) * w[j * k + p];
            }
            y[i * n + j] = acc;
        }
    }
    return y;
}

struct Config {
    std::string label;
    Quantizer quantizer;
};

void write_header(std::ostream& os) {
    os << "distribution,method,M,N,K,scale_x,scale_w,"
          "x_max_abs,x_rmse,x_rel_fro,"
          "y_max_abs,y_mean_abs,y_rmse,y_rel_fro\n";
}

void run_case(std::ostream& os, Dist d, const Config& cfg, std::size_t size,
              unsigned seed) {
    const std::size_t m = size, n = size, k = size;
    Quantizer::check_inner_dim(k);
    std::mt19937 rng(seed);  // same data for every method at a given size
    const std::vector<float> x = generate(d, m * k, rng);
    const std::vector<float> w = generate(d, n * k, rng);

    const auto qx = cfg.quantizer.quantize(x);
    const auto qw = cfg.quantizer.quantize(w);
    const QuantError ex = measure_error(x, Quantizer::dequantize(qx));

    const std::vector<float> y_ref = matmul_f32(x, w, m, n, k);
    const std::vector<float> y_hat = Quantizer::dequantize_accumulator(
        matmul_i8(qx.values, qw.values, m, n, k), qx.scale, qw.scale);
    const QuantError ey = measure_error(y_ref, y_hat);

    os << dist_name(d) << ',' << cfg.label << ',' << m << ',' << n << ',' << k << ','
       << qx.scale << ',' << qw.scale << ',' << ex.max_abs << ',' << ex.rmse << ','
       << ex.relative_fro << ',' << ey.max_abs << ',' << ey.mean_abs << ','
       << ey.rmse << ',' << ey.relative_fro << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    const std::string path = argc > 1 ? argv[1] : "quant_error_analysis.csv";
    const unsigned seed = argc > 2 ? static_cast<unsigned>(std::stoul(argv[2])) : 538u;

    std::ofstream out(path);
    if (!out) {
        std::cerr << "Could not open " << path << " for writing.\n";
        return 1;
    }

    const std::vector<Dist> dists = {Dist::Uniform, Dist::Normal, Dist::NormalOutliers,
                                     Dist::Laplace, Dist::Sparse, Dist::Zeros};
    const std::vector<Config> configs = {
        {"maxabs", Quantizer()},
        {"p99.9", Quantizer(ScaleMethod::Percentile, 99.9)},
        {"p99", Quantizer(ScaleMethod::Percentile, 99.0)},
    };
    const std::vector<std::size_t> sizes = {64, 128, 256};

    write_header(out);
    for (Dist d : dists) {
        for (const Config& cfg : configs) {
            for (std::size_t size : sizes) {
                run_case(out, d, cfg, size, seed);
            }
        }
    }
    std::cout << "Wrote " << dists.size() * configs.size() * sizes.size()
              << " rows to " << path << "\n";
    return 0;
}
