# ee538-final-project

CPU-based symmetric INT8 quantized linear operator, `Y = X Wᵀ`.

| Component | Owner |
|---|---|
| Quantization / dequantization / error analysis | Darius |
| `Matrix<T>`, `PackedMatrix`, scalar and blocked INT8 kernels | Jenny |
| Data generation, input validation, float32 reference, CLI, benchmarks | Gerson |

## Build and test

```sh
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

## Quantization interface (`include/Quantizer.h`, `include/QuantError.h`)

Matrices are passed as flat **row-major** `std::vector<float>` buffers, for example `Matrix<float>::data()`.

```cpp
#include "Quantizer.h"
#include "QuantError.h"
using namespace qlinear;

Quantizer quantizer;                          // per-tensor max-abs (baseline)
QuantizedTensor qx = quantizer.quantize(x);   // qx.values: int8, qx.scale: float
QuantizedTensor qw = quantizer.quantize(w);

Quantizer::check_inner_dim(K);                // throws if INT32 could overflow
std::vector<int32_t> acc = /* kernel: qx.values * qw.values^T */;

std::vector<float> y_hat =
    Quantizer::dequantize_accumulator(acc, qx.scale, qw.scale);
QuantError err = measure_error(y_ref, y_hat); // max_abs, rmse, relative_fro, ...
```

### Contract

- **Scale:** `s = max|x| / 127`, one per tensor. All-zero and empty tensors get `s = 1`.
- **Quantize:** `q = clamp(round(x / s), -127, 127)`, rounding half away from zero. `-128` is never produced.
- **Kernels** take `int8` inputs and must accumulate into **`int32`**.
- **Overflow bound:** `K ≤ Quantizer::max_safe_inner_dim() = floor(INT32_MAX / 127²) = 133144`.
- **Dequantize:** `y = acc · s_x · s_w`, computed in double.
- **Invalid input:** NaN or Inf values, a non-positive scale, and a percentile outside `(0, 100]` throw `std::invalid_argument`.
- **Optional:** `Quantizer(ScaleMethod::Percentile, 99.9)` clips the scale at a percentile of `|x|`. It is used for analysis only.

## Quantization error analysis

```sh
mkdir -p results
./build/quant_error_analysis results/quant_error_analysis.csv [seed]
```

This sweeps 6 input distributions × 3 scale methods × 3 sizes (64, 128, 256). Each CSV row reports the round-trip error for X and the end-to-end error for Y against a float32 reference. The default seed is 538.

Relative Frobenius error of Y at 256³ with max-abs scaling:

| Distribution | `y_rel_fro` |
|---|---|
| uniform [-1, 1] | 0.0055 |
| normal | 0.014 |
| sparse (10% nonzero) | 0.014 |
| laplace | 0.027 |
| normal + 0.1% ×50 outliers | 0.30 |

On the outlier input, percentile clipping makes the output **worse** (0.86 at p99.9). The clipped outliers carry most of the energy in `Y`, so max-abs stays the default.

## Command-line driver (`qlinear`)

```sh
./build/qlinear gen   --rows 4 --cols 8 --dist normal --seed 538 --out x.txt
./build/qlinear run   --x tests/data/golden_x.txt --w tests/data/golden_w.txt [--kernel reference|scalar|blocked|all] [--block 64] [--out y.txt]
./build/qlinear bench --config quick --out results/raw/bench.csv
./build/qlinear help
```

`run` computes `Y = X Wᵀ` with each selected operator and reports its error against the float32 reference, its per-stage times and its weight storage. Exit codes: `0` success, `1` invalid input, `2` usage error. Every error message names the file and line to fix.

**Matrix file format:** optional `#` comments, a `<rows> <cols>` header, then `rows*cols` finite numbers in row-major order:

```
# X = [[1, -2]]
1 2
1 -2
```

**Rejected input:**
- a missing, empty or truncated file;
- a malformed, zero or oversized header (more than 2²⁶ elements);
- non-numeric, `nan`, `inf` or out-of-float32-range values;
- extra values;
- mismatched inner dimensions;
- `K > 133144`, the INT32 overflow bound;
- a block size of 0.

## Tests

```sh
scripts/run_tests.sh            # Release build + ctest, then an ASan/UBSan build + ctest
scripts/run_tests.sh --no-sanitize
```

| Test | Checks |
|---|---|
| `data_generator` | The same seed gives bit-identical matrices; a fixed snapshot of the first values; the `mt19937` standard checkpoint; mean/stddev; outlier rate; outlier channels; invalid specs |
| `matrix_io` | Parsing, every rejection path with its line number, write→read is bit-exact, fixture files |
| `input_validator` | Shape rules; `K = 133144` accepted and `133145` rejected; size cap; `size_t` overflow; non-finite values; CLI integers |
| `reference_kernel` | The proposal's golden example (exact), hand-computed non-square cases, 1×1, zeros, double accumulation |
| `integration` | Virtual dispatch through `LinearOperator`, stage timing, misuse errors, the accuracy contract (relative error < 1% uniform, < 2.5% Gaussian), the benchmark CSV |
| `cli_*` (20) | The end-to-end golden `run` and every bad-input fixture: exit code and message |

**Accuracy contract.** The proposal targeted < 1% relative error for Gaussian inputs. With one scale per tensor, measured error is 1.2–1.7% (it grows with size, because the largest |x| grows and so the quantization step does too). The tests therefore assert < 1% for uniform and < 2.5% for Gaussian inputs. Outlier-heavy inputs are measured and reported, not asserted.

**Memory checking.** The whole suite runs under AddressSanitizer + UBSan (`-DQLINEAR_SANITIZE=ON`). LeakSanitizer is not available in Apple clang on Apple Silicon. On macOS, run `leaks --atExit -- ./build/integration_tests` in a regular terminal instead. All owned memory is held by `std::vector` / `std::unique_ptr` (RAII); there is no raw `new`/`delete`.

## Experiments

```sh
scripts/run_benchmarks.sh quick      # smoke test, seconds
scripts/run_benchmarks.sh default    # the proposal's full grid, several minutes
python3 scripts/plot_results.py results/raw/bench_default_<timestamp>.csv
```

The default grid:
- sizes 64/256/1024/2048;
- square (`M = K = N`) and single-row (`M = 1`) shapes;
- uniform, normal and normal-with-outliers inputs;
- block sizes 16/32/64/128 (blocked operator only);
- 1 warm-up and 5 measured repetitions;
- seed 538.

The CSV has one row per measured repetition (raw data), with per-stage times:
- `prepare` (weights, once);
- `quantize_x`, `compute` and `dequantize`;
- `overhead` (total minus those stages).

Each row also has the weight storage in bytes (calculated from the data structure, not measured peak memory), the error against float32, the compiler and the build flags. A sidecar `.env.txt` records the machine and the git commit. `plot_results.py` writes the median/IQR tables to `results/tables/` and the plots to `results/plots/`.
