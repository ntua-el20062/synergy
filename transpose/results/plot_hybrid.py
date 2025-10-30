#!/usr/bin/env python3

import argparse
import re
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

#RUN IT LIKE THIS:python3 plot_hybrid.py -i results_hybrid.txt -o plots --all



HEADER_RE = re.compile(r'^N=(\d+)\s+iters=(\d+)(?:\s+frac=([0-9.]+))?')
MODE_RE = re.compile(r'^mode=([a-zA-Z0-9_]+)')
HYB_KERNEL_RE = re.compile(r'^Hybrid Kernel avg:\s*([0-9.]+)\s*ms,\s*effective bandwidth:\s*([0-9.]+)\s*GB/s')
SECTION_HYBRID = re.compile(r'^HYBRID SWEEP')
ANSI_RE = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')

def parse_file(path: Path):
    """Parse only HYBRID SWEEP data from the file."""
    text = path.read_text(errors="ignore")
    text = ANSI_RE.sub('', text)
    lines = [l.strip() for l in text.splitlines()]

    section = None
    current_N = None
    current_iters = None
    current_frac = None
    current_mode = None
    hybrid_rows = []

    for line in lines:
        if SECTION_HYBRID.search(line):
            section = "HYBRID SWEEP"
            continue

        m = HEADER_RE.search(line)
        if m:
            current_N = int(m.group(1))
            current_iters = int(m.group(2))
            current_frac = float(m.group(3)) if m.group(3) else None
            current_mode = None
            continue

        m = MODE_RE.search(line)
        if m:
            current_mode = m.group(1)
            continue

        m = HYB_KERNEL_RE.search(line)
        if m and section == "HYBRID SWEEP" and current_mode is not None:
            hybrid_rows.append({
                "section": section,
                "N": current_N,
                "iters": current_iters,
                "frac": current_frac,
                "mode": current_mode,
                "kernel_ms": float(m.group(1)),
                "bandwidth_GBps": float(m.group(2)),
            })

    hybrid_df = pd.DataFrame(hybrid_rows)
    return hybrid_df

def save_csv(hybrid_df, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    hybrid_csv = outdir / "hybrid_sweep_metrics.csv"
    hybrid_df.to_csv(hybrid_csv, index=False)
    return hybrid_csv

def _pick_hybrid_pair(hybrid_df: pd.DataFrame, n: int = None, iters: int = None):
    if n is not None and iters is not None:
        return n, iters
    if hybrid_df.empty:
        return None, None
    common_pair = hybrid_df.groupby(["N", "iters"]).size().sort_values(ascending=False).index[0]
    return int(common_pair[0]), int(common_pair[1])

def _all_pairs(df: pd.DataFrame):
    if df.empty:
        return []
    pairs = df.drop_duplicates(["N", "iters"])[["N", "iters"]].sort_values(["N", "iters"])
    return [tuple(x) for x in pairs.to_records(index=False)]

def plot_hybrid_bandwidth_vs_frac(hybrid_df: pd.DataFrame, outdir: Path, n: int, iters: int, show=False):
    sub = hybrid_df[(hybrid_df["N"] == n) & (hybrid_df["iters"] == iters)].copy()
    if sub.empty:
        print(f"No HYBRID data for N={n}, iters={iters} — skipping.", file=sys.stderr)
        return None

    pivot = sub.pivot_table(index="frac", columns="mode", values="bandwidth_GBps", aggfunc="mean").sort_index()
    plt.figure(figsize=(8, 5))
    bar_width = 0.8 / max(1, len(pivot.columns))
    x = range(len(pivot.index))

    for i, col in enumerate(pivot.columns):
        plt.bar([xi + i * bar_width for xi in x], pivot[col], width=bar_width, label=col)

    plt.xticks([xi + bar_width * (len(pivot.columns) - 1) / 2 for xi in x],
               [f"{f:.2f}" for f in pivot.index])
    plt.title(f"Hybrid Bandwidth vs CPU Fraction (N={n}, iters={iters})")
    plt.xlabel("CPU fraction (frac)")
    plt.ylabel("Effective bandwidth (GB/s)")
    plt.legend()
    plt.tight_layout()

    outpath = outdir / f"hybrid_bandwidth_vs_frac_N{n}_iters{iters}.png"
    plt.savefig(outpath, dpi=200)
    if show: plt.show()
    else: plt.close()
    return outpath

def plot_hybrid_time_vs_frac(hybrid_df: pd.DataFrame, outdir: Path, n: int, iters: int, show=False):
    sub = hybrid_df[(hybrid_df["N"] == n) & (hybrid_df["iters"] == iters)].copy()
    if sub.empty:
        print(f"No HYBRID data for N={n}, iters={iters} — skipping.", file=sys.stderr)
        return None

    pivot_t = sub.pivot_table(index="frac", columns="mode", values="kernel_ms", aggfunc="mean").sort_index()
    plt.figure(figsize=(8, 5))
    bar_width = 0.8 / max(1, len(pivot_t.columns))
    x = range(len(pivot_t.index))

    for i, col in enumerate(pivot_t.columns):
        plt.bar([xi + i * bar_width for xi in x], pivot_t[col], width=bar_width, label=col)

    plt.xticks([xi + bar_width * (len(pivot_t.columns) - 1) / 2 for xi in x],
               [f"{f:.2f}" for f in pivot_t.index])
    plt.title(f"Hybrid Kernel Time vs CPU Fraction (N={n}, iters={iters})")
    plt.xlabel("CPU fraction (frac)")
    plt.ylabel("Kernel time (ms)")
    plt.legend()
    plt.tight_layout()

    outpath = outdir / f"hybrid_kernel_time_vs_frac_N{n}_iters{iters}.png"
    plt.savefig(outpath, dpi=200)
    if show: plt.show()
    else: plt.close()
    return outpath

def main():
    p = argparse.ArgumentParser(description="Parse and plot only HYBRID SWEEP metrics.")
    p.add_argument("--input", "-i", required=True, type=Path, help="Path to results_hybrid.txt")
    p.add_argument("--outdir", "-o", type=Path, default=Path("plots"), help="Output directory")
    p.add_argument("--show", action="store_true", help="Show plots interactively")
    p.add_argument("--hybrid-n", type=int, help="Hybrid plot N (default = most common)")
    p.add_argument("--hybrid-iters", type=int, help="Hybrid plot iters (default = most common)")
    p.add_argument("--all", action="store_true", help="Generate plots for ALL (N,iters) pairs")
    args = p.parse_args()

    hybrid_df = parse_file(args.input)
    hybrid_csv = save_csv(hybrid_df, args.outdir)
    print(f"Wrote CSV:\n  {hybrid_csv}")

    # === PLOTTING ===
    if args.all:
        if hybrid_df.empty:
            print("No HYBRID SWEEP records found.")
            return
        for n, iters in _all_pairs(hybrid_df):
            hb_path = plot_hybrid_bandwidth_vs_frac(hybrid_df, args.outdir, n=n, iters=iters, show=args.show)
            if hb_path:
                print(f"Saved HYBRID bandwidth plot: {hb_path}")
            ht_path = plot_hybrid_time_vs_frac(hybrid_df, args.outdir, n=n, iters=iters, show=args.show)
            if ht_path:
                print(f"Saved HYBRID kernel time plot: {ht_path}")
        return

    # default: single pair (most common or specified)
    n, iters = _pick_hybrid_pair(hybrid_df, args.hybrid_n, args.hybrid_iters)
    if n is None:
        print("No HYBRID SWEEP records found.", file=sys.stderr)
        return
    hb_path = plot_hybrid_bandwidth_vs_frac(hybrid_df, args.outdir, n=n, iters=iters, show=args.show)
    if hb_path:
        print(f"Saved HYBRID bandwidth plot: {hb_path}")
    ht_path = plot_hybrid_time_vs_frac(hybrid_df, args.outdir, n=n, iters=iters, show=args.show)
    if ht_path:
        print(f"Saved HYBRID kernel time plot: {ht_path}")

if __name__ == "__main__":
    main()

