#!/usr/bin/env bash
# Builds in Release mode and runs the experiment sweep.
#
# Usage: scripts/run_benchmarks.sh [quick|default] [extra qlinear bench options...]
#   e.g. scripts/run_benchmarks.sh default
#        scripts/run_benchmarks.sh quick --operators reference
#
# Writes, under results/raw/:
#   bench_<config>_<timestamp>.csv       one row per measured repetition
#   bench_<config>_<timestamp>.env.txt   machine, compiler and git state
# Then plot with: python3 scripts/plot_results.py results/raw/bench_<...>.csv
#
# For stable timings: plug in the laptop and close other applications.

set -euo pipefail
cd "$(dirname "$0")/.."

config="${1:-quick}"
shift || true
case "$config" in
    quick|default) ;;
    *) echo "usage: $0 [quick|default] [qlinear bench options...]" >&2; exit 2 ;;
esac

jobs="$( (command -v nproc >/dev/null && nproc) || sysctl -n hw.ncpu 2>/dev/null || echo 2)"
cmake -S . -B build -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build build -j "$jobs" --target qlinear

mkdir -p results/raw
stamp="$(date +%Y%m%d-%H%M%S)"
base="results/raw/bench_${config}_${stamp}"

{
    echo "date:       $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "command:    qlinear bench --config ${config} $*"
    echo "git:        $(git rev-parse --short HEAD 2>/dev/null || echo unknown)$(git diff --quiet 2>/dev/null || echo ' (uncommitted changes)')"
    echo "os:         $(uname -srm)"
    if [[ "$(uname)" == "Darwin" ]]; then
        echo "cpu:        $(sysctl -n machdep.cpu.brand_string)"
        echo "cores:      $(sysctl -n hw.ncpu)"
        echo "memory:     $(( $(sysctl -n hw.memsize) / 1073741824 )) GiB"
    else
        echo "cpu:        $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //')"
        echo "cores:      $(nproc)"
        echo "memory:     $(awk '/MemTotal/ {printf "%.0f GiB", $2/1048576}' /proc/meminfo)"
    fi
    echo "compiler:   $(c++ --version | head -1)"
    echo "cmake:      $(cmake --version | head -1)"
    echo "build type: Release (flags are recorded in every CSV row)"
} > "${base}.env.txt"

./build/qlinear bench --config "$config" "$@" --out "${base}.csv"

echo
echo "raw data:    ${base}.csv"
echo "environment: ${base}.env.txt"
echo "next:        python3 scripts/plot_results.py ${base}.csv"
