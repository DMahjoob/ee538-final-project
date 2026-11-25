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
