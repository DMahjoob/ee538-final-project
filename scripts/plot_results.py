#!/usr/bin/env python3
"""Summarize and plot a raw benchmark CSV written by `qlinear bench`.

Usage:
    python3 scripts/plot_results.py results/raw/bench_<...>.csv [--out results]

Writes:
    <out>/tables/<name>_summary.md   median and IQR per configuration
    <out>/plots/<name>_*.png         time vs size, block-size sweep,
                                     stage breakdown, error vs size

Each CSV row is one measured repetition. Statistics: median, and the
interquartile range (quartiles by linear interpolation, as numpy.percentile).
Only the standard library and matplotlib are used.
"""

import argparse
import csv
import os
import statistics
import sys
from collections import defaultdict

GROUP_KEYS = ("dist", "shape", "m", "k", "n", "operator", "block")
TIME_FIELDS = ("prepare_ms", "quantize_x_ms", "compute_ms", "dequantize_ms",
               "overhead_ms", "total_ms")


def quartiles(values):
    """(q1, median, q3) with linear interpolation; works for one sample."""
    if len(values) == 1:
        return values[0], values[0], values[0]
    q1, med, q3 = statistics.quantiles(values, n=4, method="inclusive")
    return q1, med, q3


def load(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"{path}: no data rows")
    missing = [k for k in GROUP_KEYS + TIME_FIELDS + ("rel_fro_err",) if k not in rows[0]]
    if missing:
        sys.exit(f"{path}: missing columns {missing}")
    return rows


def summarize(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in GROUP_KEYS)].append(r)
    summary = []
    for key, reps in groups.items():
        entry = dict(zip(GROUP_KEYS, key))
        for k in ("m", "k", "n", "block"):
            entry[k] = int(entry[k])
        entry["reps"] = len(reps)
        for field in TIME_FIELDS:
            entry[field] = quartiles([float(r[field]) for r in reps])
        entry["rel_fro_err"] = float(reps[0]["rel_fro_err"])
        entry["max_abs_err"] = float(reps[0]["max_abs_err"])
        entry["weight_bytes"] = int(reps[0]["weight_bytes"])
        entry["label"] = entry["operator"] + (f" b={entry['block']}" if entry["block"] else "")
        summary.append(entry)
    summary.sort(key=lambda e: (e["dist"], e["shape"], e["k"], e["operator"], e["block"]))
    return summary


def write_table(summary, path, source):
    with open(path, "w") as f:
        f.write(f"# Benchmark summary\n\nSource: `{source}`. Times in ms: median "
                "[Q1, Q3] over the measured repetitions.\n\n")
        f.write("| dist | shape | M×K×N | operator | reps | total ms | compute ms | "
                "quantize X ms | dequantize ms | prepare ms | rel. error | weight bytes |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for e in summary:
            def cell(field):
                q1, med, q3 = e[field]
                return f"{med:.4g} [{q1:.4g}, {q3:.4g}]"
            f.write(f"| {e['dist']} | {e['shape']} | {e['m']}×{e['k']}×{e['n']} | "
                    f"{e['label']} | {e['reps']} | {cell('total_ms')} | {cell('compute_ms')} | "
                    f"{e['quantize_x_ms'][1]:.4g} | {e['dequantize_ms'][1]:.4g} | "
                    f"{e['prepare_ms'][1]:.4g} | {e['rel_fro_err']:.3%} | "
                    f"{e['weight_bytes']:,} |\n")


def plot_all(summary, plot_dir, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written = []
    dists = sorted({e["dist"] for e in summary})
    shapes = sorted({e["shape"] for e in summary})
    # Timing barely depends on the values, so time plots use one distribution.
    timing_dist = "uniform" if "uniform" in dists else dists[0]

    # 1. Total time vs size, one line per operator (and block size).
    for shape in shapes:
        sel = [e for e in summary if e["shape"] == shape and e["dist"] == timing_dist]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for label in sorted({e["label"] for e in sel}):
            pts = sorted((e for e in sel if e["label"] == label), key=lambda e: e["k"])
            xs = [e["k"] for e in pts]
            q1 = [e["total_ms"][0] for e in pts]
            med = [e["total_ms"][1] for e in pts]
            q3 = [e["total_ms"][2] for e in pts]
            ax.errorbar(xs, med, yerr=[[m - a for m, a in zip(med, q1)],
                                       [b - m for m, b in zip(med, q3)]],
                        marker="o", capsize=3, label=label)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("size (K = N" + (", M = K)" if shape == "square" else ", M = 1)"))
        ax.set_ylabel("total time per forward (ms, median, IQR bars)")
        ax.set_title(f"Time vs size: {shape}, {timing_dist} inputs")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
        path = os.path.join(plot_dir, f"{name}_time_vs_size_{shape}.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)

    # 2. Blocked operator: time vs block size, one line per matrix size.
    blocked = [e for e in summary if e["operator"] == "blocked" and e["dist"] == timing_dist]
    for shape in sorted({e["shape"] for e in blocked}):
        sel = [e for e in blocked if e["shape"] == shape]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for size in sorted({e["k"] for e in sel}):
            pts = sorted((e for e in sel if e["k"] == size), key=lambda e: e["block"])
            ax.plot([e["block"] for e in pts], [e["compute_ms"][1] for e in pts],
                    marker="o", label=f"size {size}")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("block size")
        ax.set_ylabel("compute time (ms, median)")
        ax.set_title(f"Blocked kernel: block-size sweep ({shape})")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
        path = os.path.join(plot_dir, f"{name}_block_sweep_{shape}.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)

    # 3. Stage breakdown at the largest size of each shape.
    stages = ("quantize_x_ms", "compute_ms", "dequantize_ms", "overhead_ms")
    for shape in shapes:
        sel = [e for e in summary if e["shape"] == shape and e["dist"] == timing_dist]
        largest = max(e["k"] for e in sel)
        sel = sorted((e for e in sel if e["k"] == largest), key=lambda e: e["label"])
        fig, ax = plt.subplots(figsize=(7, 4.5))
        bottom = [0.0] * len(sel)
        for stage in stages:
            vals = [e[stage][1] for e in sel]
            ax.bar([e["label"] for e in sel], vals, bottom=bottom,
                   label=stage.replace("_ms", ""))
            bottom = [b + v for b, v in zip(bottom, vals)]
        ax.set_ylabel("time per forward (ms, median)")
        ax.set_title(f"Stage breakdown at size {largest} ({shape})")
        ax.legend(fontsize=8)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        path = os.path.join(plot_dir, f"{name}_stages_{shape}.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)

    # 4. Numerical error vs size per distribution (non-reference operators).
    approx = [e for e in summary if e["operator"] != "reference" and e["shape"] == "square"]
    if approx:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for dist in dists:
            for op in sorted({e["operator"] for e in approx}):
                pts = sorted((e for e in approx if e["dist"] == dist and e["operator"] == op
                              and e["block"] in (0, min(x["block"] for x in approx
                                                        if x["operator"] == op))),
                             key=lambda e: e["k"])
                if pts:
                    ax.plot([e["k"] for e in pts], [e["rel_fro_err"] * 100 for e in pts],
                            marker="o", label=f"{op}, {dist}")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("size (square)")
        ax.set_ylabel("relative Frobenius error vs float32 (%)")
        ax.set_title("INT8 numerical error vs size")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
        path = os.path.join(plot_dir, f"{name}_error_vs_size.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv", help="raw CSV from `qlinear bench`")
    parser.add_argument("--out", default="results", help="output directory (default: results)")
    parser.add_argument("--no-plots", action="store_true", help="write the table only")
    args = parser.parse_args()

    name = os.path.splitext(os.path.basename(args.csv))[0]
    summary = summarize(load(args.csv))

    table_dir = os.path.join(args.out, "tables")
    os.makedirs(table_dir, exist_ok=True)
    table = os.path.join(table_dir, f"{name}_summary.md")
    write_table(summary, table, args.csv)
    print(f"table: {table}")

    if not args.no_plots:
        plot_dir = os.path.join(args.out, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        for path in plot_all(summary, plot_dir, name):
            print(f"plot:  {path}")


if __name__ == "__main__":
    main()
