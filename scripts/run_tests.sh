#!/usr/bin/env bash
# Builds and runs the whole test suite twice:
#   1. build/       Release build (the configuration used for benchmarks)
#   2. build-asan/  Debug build with AddressSanitizer + UndefinedBehaviorSanitizer
#
# Usage: scripts/run_tests.sh [--no-sanitize]
# Exits non-zero if any build or test fails.

set -euo pipefail
cd "$(dirname "$0")/.."

jobs="$( (command -v nproc >/dev/null && nproc) || sysctl -n hw.ncpu 2>/dev/null || echo 2)"

echo "== Release build + tests (build/)"
cmake -S . -B build -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build build -j "$jobs"
ctest --test-dir build --output-on-failure

if [[ "${1:-}" == "--no-sanitize" ]]; then
    exit 0
fi

echo
echo "== ASan + UBSan build + tests (build-asan/)"
cmake -S . -B build-asan -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Debug \
      -DQLINEAR_SANITIZE=ON >/dev/null
cmake --build build-asan -j "$jobs"
# halt_on_error: any sanitizer report fails the test that triggered it.
ASAN_OPTIONS="halt_on_error=1:${ASAN_OPTIONS:-}" \
UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1:${UBSAN_OPTIONS:-}" \
    ctest --test-dir build-asan --output-on-failure

echo
echo "All tests passed in both builds."
