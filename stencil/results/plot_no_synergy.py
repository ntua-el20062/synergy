#!/usr/bin/env python3
"""
Plot all metrics from a STENCIL RESULTS text file (e.g., 'no_synergy.txt').

What it does
------------
- Parses one or more result tables from the text file.
- Builds a tidy pandas DataFrame for every row/metric.
- Saves the parsed data to CSV for transparency.
- Produces Matplotlib figures (no seaborn) for *every* metric the file provides:
    * For each (N, Step) combo: grouped bar charts by Mode for
      - Total(ms), Cold(ms), Warm(ms), Read(ms), UMback(ms)
    * For each metric: line plots vs N, with one subplot per Step and a separate line per Mode
      (if multiple Steps present).
- Saves all figures into an output directory (default: ./figures next to the input file).

Usage
-----
python plot_no_synergy.py --input no_synergy.txt --outdir figures

You can pass multiple inputs; each will get its own figures subfolder:
python plot_no_synergy.py --input no_synergy.txt other.txt --outdir figures

Notes
-----
- Charts use pure Matplotlib, one chart per figure, no explicit colors/styles, per execution environment constraints.
- Missing values noted as "--" in the text will be treated as NaN.
"""

import argparse
import os
import re
import math
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


HEADER_RE = re.compile(r'^Mode\s+N\s+Step\s+Iters\s+Total\(ms\)\s+Cold\(ms\)\s+Warm\(ms\)\s+Read\(ms\)\s+UMback\(ms\)\s*$')
ROW_RE = re.compile(
    r'^(?P<Mode>[A-Za-z0-9_]+)\s+'
    r'(?P<N>\d+)\s+'
    r'(?P<Step>\d+)\s+'
    r'(?P<Iters>\d+)\s+'
    r'(?P<Total_ms>[\d.]+)\s+'
    r'(?P<Cold_ms>[\d.\-]+)\s+'
    r'(?P<Warm_ms>[\d.\-]+)\s+'
    r'(?P<Read_ms>[\d.]+)\s+'
    r'(?P<UMback_ms>[\d.\-]+)?\s*$'
)

SEPARATOR_RE = re.compile(r'^-+\s*$')


def _to_float(x: str):
    if x is None:
        return np.nan
    x = x.strip()
    if x == '--':
        return np.nan
    try:
        return float(x)
    except Exception:
        return np.nan


def parse_stencil_file(text_path: Path) -> pd.DataFrame:
    """
    Parse the input text file into a unified DataFrame.
    Returns columns: [Mode, N, Step, Iters, Total_ms, Cold_ms, Warm_ms, Read_ms, UMback_ms, block_id]
    """
    rows = []
    block_id = -1
    with open(text_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line.strip():
                continue
            if SEPARATOR_RE.match(line):
                # a row of dashes signals the start/end of a block
                block_id += 1
                continue
            if HEADER_RE.match(line):
                # header on first block does not increment block id;
                # ensure we have a block id to attribute rows
                if block_id < 0:
                    block_id = 0
                continue
            m = ROW_RE.match(line)
            if m:
                d = m.groupdict()
                rows.append({
                    "Mode": d["Mode"],
                    "N": int(d["N"]),
                    "Step": int(d["Step"]),
                    "Iters": int(d["Iters"]),
                    "Total_ms": _to_float(d["Total_ms"]),
                    "Cold_ms": _to_float(d["Cold_ms"]),
                    "Warm_ms": _to_float(d["Warm_ms"]),
                    "Read_ms": _to_float(d["Read_ms"]),
                    "UMback_ms": _to_float(d.get("UMback_ms")),
                    "block_id": block_id,
                })
            # else: ignore unrecognized lines (e.g., titles)
    if not rows:
        raise ValueError(f"No data rows parsed from {text_path}")
    df = pd.DataFrame(rows)
    # Normalize/clean types
    numeric_cols = ["N", "Step", "Iters", "Total_ms", "Cold_ms", "Warm_ms", "Read_ms", "UMback_ms", "block_id"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def grouped_bar_by_mode(df: pd.DataFrame, metric: str, N: int, Step: int, outdir: Path, title_prefix: str=''):
    sub = df[(df["N"] == N) & (df["Step"] == Step)].copy()
    if sub.empty:
        return
    sub = sub.sort_values("Mode")
    fig = plt.figure()
    ax = fig.gca()
    ax.bar(sub["Mode"].values, sub[metric].values)
    ax.set_xlabel("Mode")
    ax.set_ylabel(metric.replace('_', ' '))
    ax.set_title(f"{title_prefix}{metric} for N={N}, Step={Step}")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    fname = outdir / f"{metric}_N{N}_Step{Step}.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)


def line_vs_N_per_step(df: pd.DataFrame, metric: str, outdir: Path, title_prefix: str=''):
    # One figure per Step, lines by Mode
    for step in sorted(df["Step"].dropna().unique()):
        sub = df[df["Step"] == step].copy()
        if sub.empty:
            continue
        fig = plt.figure()
        ax = fig.gca()
        modes = sorted(sub["Mode"].unique())
        for mode in modes:
            mdf = sub[sub["Mode"] == mode].sort_values("N")
            ax.plot(mdf["N"].values, mdf[metric].values, marker='o', label=mode)
        ax.set_xlabel("N")
        ax.set_ylabel(metric.replace('_', ' '))
        ax.set_title(f"{title_prefix}{metric} vs N (Step={step})")
        ax.legend(loc='best', fontsize='small')
        plt.tight_layout()
        fname = outdir / f"{metric}_vsN_Step{step}.png"
        fig.savefig(fname, dpi=150)
        plt.close(fig)


def save_all_plots(df: pd.DataFrame, outdir: Path, title_prefix: str=''):
    outdir.mkdir(parents=True, exist_ok=True)
    metrics = ["Total_ms", "Cold_ms", "Warm_ms", "Read_ms", "UMback_ms"]
    # Per (N, Step)
    for N in sorted(df["N"].dropna().unique()):
        for step in sorted(df["Step"].dropna().unique()):
            for metric in metrics:
                if metric not in df.columns:
                    continue
                grouped_bar_by_mode(df, metric, int(N), int(step), outdir, title_prefix=title_prefix)
    # Trend vs N for each Step
    for metric in metrics:
        if metric in df.columns:
            line_vs_N_per_step(df, metric, outdir, title_prefix=title_prefix)


def main():
    ap = argparse.ArgumentParser(description="Plot all metrics from STENCIL RESULTS text files.")
    ap.add_argument("--input", "-i", nargs="+", required=True, help="Path(s) to input .txt file(s) containing STENCIL RESULTS tables.")
    ap.add_argument("--outdir", "-o", default=None, help="Directory to save figures (default: 'figures' next to each input).")
    ap.add_argument("--csv", action="store_true", help="Also save parsed CSV next to figures.")
    args = ap.parse_args()

    for inp in args.input:
        path = Path(inp).expanduser().resolve()
        if not path.exists():
            raise SystemExit(f"Input file not found: {path}")
        df = parse_stencil_file(path)

        # Decide output directory
        if args.outdir:
            outdir = Path(args.outdir).expanduser().resolve()
            outdir = outdir / path.stem  # namespace per input
        else:
            outdir = path.parent / "figures" / path.stem

        # optional CSV
        if args.csv:
            outdir.mkdir(parents=True, exist_ok=True)
            csv_path = outdir / f"{path.stem}_parsed.csv"
            df.to_csv(csv_path, index=False)

        title_prefix = f"{path.stem} — "
        save_all_plots(df, outdir, title_prefix=title_prefix)

        # Also print a compact summary to stdout
        print(f"Parsed {len(df)} rows from {path.name}.")
        print("Available metrics: Total_ms, Cold_ms, Warm_ms, Read_ms, UMback_ms")
        steps = ", ".join(str(s) for s in sorted(df['Step'].dropna().unique()))
        Ns = ", ".join(str(n) for n in sorted(df['N'].dropna().unique()))
        print(f"Steps: {steps}")
        print(f"N values: {Ns}")
        print(f"Figures saved under: {outdir}")

if __name__ == "__main__":
    main()
