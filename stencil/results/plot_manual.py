#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate grouped BAR PLOTS from manual.txt:

- step=1, Cold (ms) vs Threads vs K% for N=2048/4096/8192
- step=1, Read (ms)  vs Threads vs K% for N=2048/4096/8192
- step=2, Cold & Warm (same plot) vs Threads vs K% for N=2048/4096/8192
- step=2, Read (ms)  vs Threads vs K% for N=2048/4096/8192

Usage (default: manual.txt in current dir):
    python make_plots.py [--input manual.txt] [--out plots] [--mode MODE]

If --mode is provided, the script filters to that Mode exactly (e.g., "gh_hbm_shared").
Otherwise it averages across rows that share (N, Step, Threads, K%).
"""

import re
import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


THREADS_ORDER = [1, 2, 4, 8, 16, 32, 64]
K_ORDER       = [10, 15, 20, 25, 30]


def parse_manual(path: Path) -> pd.DataFrame:
    text = path.read_text(errors="ignore")

    row_re = re.compile(
        r"^(?P<Mode>\S+)\s+"
        r"(?P<N>\d+)\s+"
        r"(?P<Step>\d+)\s+"
        r"(?P<Iters>\d+)\s+"
        r"(?P<Kpct>\d+)\s+"
        r"(?P<Threads>\d+)\s+"
        r"(?P<Total>[.\d]+)\s+"
        r"(?P<Cold>[.\d-]+)\s+"
        r"(?P<Warm>[.\d-]+)\s+"
        r"(?P<Read>[.\d-]+)\s+"
        r"(?P<UMback>[.\d-]+)\s*$",
        re.MULTILINE
    )

    rows = [m.groupdict() for m in row_re.finditer(text)]
    if not rows:
        raise RuntimeError("No rows parsed — check the format of manual.txt.")

    df = pd.DataFrame(rows)
    num_cols = ["N","Step","Iters","Kpct","Threads","Total","Cold","Warm","Read","UMback"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace("--","nan"), errors="coerce")

    df = df[df["Kpct"].isin(K_ORDER) & df["Threads"].isin(THREADS_ORDER)]
    return df


def grouped_bar_plot(
    data: pd.DataFrame,
    title: str,
    ylabel: str,
    outfile: Path,
    metrics: List[str],
    legend_labels: Optional[List[str]] = None,
):
    """
    Grouped bars:
      X: Threads
      Groups within each Thread: K% (color-coded)
      Bars within each K%: metrics (hatch-coded) -> e.g., Cold (no hatch), Warm (//)
    """
    from matplotlib.patches import Patch

    threads = [t for t in THREADS_ORDER if t in data["Threads"].unique()]
    ks      = [k for k in K_ORDER       if k in data["Kpct"].unique()]
    if not threads or not ks:
        print(f"[skip] Nothing to plot for: {title}")
        return

    M, T, K = len(metrics), len(threads), len(ks)
    if legend_labels is None:
        legend_labels = metrics

    # values array [thread, k, metric]
    vals = np.full((T, K, M), np.nan, dtype=float)
    for ti, t in enumerate(threads):
        for ki, k in enumerate(ks):
            row = data[(data["Threads"]==t) & (data["Kpct"]==k)]
            if not row.empty:
                for mi, m in enumerate(metrics):
                    vals[ti, ki, mi] = row[m].mean(skipna=True)

    # Colors: one per K%
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0","C1","C2","C3","C4","C5","C6","C7","C8","C9"])
    k_colors = [color_cycle[i % len(color_cycle)] for i in range(K)]

    # Hatches for metrics
    metric_hatches = {0: "", 1: "//"}  # extend if >2 metrics

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(T)

    group_width = 0.8
    width_per_k = group_width / K
    width_per_bar = width_per_k / M

    # draw bars
    for ki, k in enumerate(ks):
        k_color = k_colors[ki]
        k_group_start = -group_width/2 + ki*width_per_k
        for mi, m in enumerate(metrics):
            offset = k_group_start + mi*width_per_bar + width_per_bar/2
            xpos = x + offset
            heights = vals[:, ki, mi]
            ax.bar(
                xpos,
                heights,
                width=width_per_bar*0.95,
                color=k_color,                 # distinct color per K%
                edgecolor="black",
                linewidth=0.5,
                hatch=metric_hatches.get(mi, ""),
            )

    # axes/labels
    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in threads])
    ax.set_xlabel("Threads")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    # Legend A (K% colors)
    k_handles = [Patch(facecolor=k_colors[ki], edgecolor="black") for ki in range(K)]
    k_labels  = [f"{k}%" for k in ks]
    leg1 = ax.legend(k_handles, k_labels, title="K%", loc="upper left", frameon=False)

    # Legend B (metrics hatches)
    m_handles = [
        Patch(facecolor="white", edgecolor="black", hatch=metric_hatches.get(mi, ""))
        for mi in range(M)
    ]
    ax.add_artist(leg1)
    ax.legend(m_handles, legend_labels, title="Metric", loc="upper right", frameon=False)

    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    print(f"[saved] {outfile}")


def make_plots_for_N(df: pd.DataFrame, N: int, out_dir: Path):
    dN = df[df["N"] == N]
    if dN.empty:
        print(f"[warn] No data for N={N}")
        return

    # STEP 1
    d1 = dN[dN["Step"] == 1]
    if not d1.empty and d1["Cold"].notna().any():
        grouped_bar_plot(
            d1,
            title=f"Step=1 | N={N} | Cold (ms) vs Threads vs K%",
            ylabel="Cold (ms)",
            outfile=out_dir / f"N{N}_step1_cold_bars.png",
            metrics=["Cold"],
        )
    if not d1.empty and d1["Read"].notna().any():
        grouped_bar_plot(
            d1,
            title=f"Step=1 | N={N} | Read (ms) vs Threads vs K%",
            ylabel="Read (ms)",
            outfile=out_dir / f"N{N}_step1_read_bars.png",
            metrics=["Read"],
        )

    # STEP 2
    d2 = dN[dN["Step"] == 2]
    if not d2.empty and d2["Cold"].notna().any() and d2["Warm"].notna().any():
        grouped_bar_plot(
            d2,
            title=f"Step=2 | N={N} | Cold & Warm (ms) vs Threads vs K%",
            ylabel="ms",
            outfile=out_dir / f"N{N}_step2_cold_warm_bars.png",
            metrics=["Cold", "Warm"],
            legend_labels=["Cold", "Warm"],
        )
    if not d2.empty and d2["Read"].notna().any():
        grouped_bar_plot(
            d2,
            title=f"Step=2 | N={N} | Read (ms) vs Threads vs K%",
            ylabel="Read (ms)",
            outfile=out_dir / f"N{N}_step2_read_bars.png",
            metrics=["Read"],
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("manual.txt"), help="Path to manual.txt")
    ap.add_argument("--out",   type=Path, default=Path("plots"),     help="Output directory for PNGs")
    ap.add_argument("--mode",  type=str,  default=None,              help="Filter to a specific Mode (exact match)")
    args = ap.parse_args()

    df = parse_manual(args.input)

    if args.mode:
        before = len(df)
        df = df[df["Mode"] == args.mode]
        if df.empty:
            raise SystemExit(f"No rows match Mode='{args.mode}'.")
        print(f"[info] Filtered by Mode='{args.mode}' ({len(df)}/{before} rows kept).")

    args.out.mkdir(parents=True, exist_ok=True)

    for N in (2048, 4096, 8192):
        make_plots_for_N(df, N, args.out)

    print(f"Done. Files written to: {args.out.resolve()}")


if __name__ == "__main__":
    main()

