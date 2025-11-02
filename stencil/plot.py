#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import argparse
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Keep original orders for consistent sorting of axes/legends
K_ORDER = [0, 10, 25, 30, 50, 70, 80]

def parse_manual(path: Path) -> pd.DataFrame:
    text = path.read_text(errors="ignore")

    numf = r'(?:[.\d+-]+|--)'
    row_re = re.compile(
        r'^(?P<Mode>\S+)\s+'
        r'(?P<N>\d+)\s+'
        r'(?P<Step>\d+)\s+'
        r'(?P<Iters>\d+)\s+'
        r'(?P<Kpct>\d+)\s+'
        r'(?P<Threads>\d+)\s+'
        r'(?P<H2D>'      + numf + r')\s+'
        r'(?P<UMDev>'    + numf + r')\s+'
        r'(?P<D2H>'      + numf + r')\s+'
        r'(?P<UMHost>'   + numf + r')\s+'
        r'(?P<GPUms>'    + numf + r')\s+'
        r'(?P<GPUgbps>'  + numf + r')\s+'
        r'(?P<CPUms>'    + numf + r')\s+'
        r'(?P<CPUgbps>'  + numf + r')\s+'
        r'(?:(?P<End2End>'+ numf + r')\s+)?'
        r'(?:(?P<FullE2E>'+ numf + r')\s+)?',
        re.MULTILINE
    )

    rows = [m.groupdict() for m in row_re.finditer(text)]
    if not rows:
        raise RuntimeError(
            "No rows parsed — check the file formatting.\n"
            "Expected the manual-section table from run_synergy.sh."
        )

    df = pd.DataFrame(rows)
    numeric_cols = [
        "N","Step","Iters","Kpct","Threads",
        "H2D","UMDev","D2H","UMHost",
        "GPUms","GPUgbps","CPUms","CPUgbps",
        "End2End", "FullE2E"
    ]
    for c in numeric_cols:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c].astype(str).str.replace("--", "nan"), errors="coerce")

    df["Mode"] = df["Mode"].astype(str)

    # Average duplicates for stability (same (Mode,N,Step,Threads,Kpct))
    key = ["Mode","N","Step","Threads","Kpct"]
    df = df.groupby(key, as_index=False).mean(numeric_only=True)

    return df

def grouped_bar_plot_by_kpct_modes(
    data: pd.DataFrame,
    title: str,
    ylabel: str,
    outfile: Path,
    metrics: List[str],
    legend_labels: Optional[List[str]] = None,
):
    """
    Grouped bars with X = Kpct; color = Mode; hatch = metric.
    Assumes data has been pre-filtered to a single Threads value.
    """
    from matplotlib.patches import Patch

    if legend_labels is None:
        legend_labels = metrics

    # Sort Kpct by preferred order, then any extras numerically
    k_unique = sorted(data["Kpct"].dropna().unique().tolist(), key=lambda x: (K_ORDER.index(x) if x in K_ORDER else len(K_ORDER), x))
    modes    = sorted(data["Mode"].unique().tolist())

    if not k_unique or not modes:
        print(f"[skip] Nothing to plot for: {title}")
        return

    K, S, M = len(k_unique), len(modes), len(metrics)
    vals = np.full((K, S, M), np.nan, dtype=float)

    # Aggregate metric per (Kpct, Mode)
    for ki, k in enumerate(k_unique):
        for si, mode in enumerate(modes):
            row = data[(data["Kpct"] == k) & (data["Mode"] == mode)]
            if not row.empty:
                for mi, m in enumerate(metrics):
                    if m in row.columns:
                        vals[ki, si, mi] = pd.to_numeric(row[m], errors="coerce").mean(skipna=True)

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get(
        "color", ["C0","C1","C2","C3","C4","C5","C6","C7","C8","C9"]
    )
    mode_colors = [color_cycle[i % len(color_cycle)] for i in range(S)]

    # Hatches distinguish metrics
    metric_hatches = {0: "", 1: "//", 2: "xx", 3: "++", 4: "..", 5: "\\\\"}

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(K, dtype=float)

    group_width   = 0.82
    width_per_s   = group_width / max(S, 1)
    width_per_bar = width_per_s / max(M, 1)

    for si, mode in enumerate(modes):
        m_color = mode_colors[si]
        s_group_start = -group_width / 2 + si * width_per_s
        for mi, m in enumerate(metrics):
            offset = s_group_start + mi * width_per_bar + width_per_bar / 2
            xpos = x + offset
            heights = vals[:, si, mi]
            ax.bar(
                xpos,
                heights,
                width=width_per_bar * 0.95,
                color=m_color,
                edgecolor="black",
                linewidth=0.5,
                hatch=metric_hatches.get(mi, ""),
            )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(k)}" for k in k_unique])
    ax.set_xlabel("K%")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    # Legends: modes (colors) + metrics (hatches)
    mode_handles = [Patch(facecolor=mode_colors[si], edgecolor="black") for si in range(S)]
    leg1 = ax.legend(mode_handles, modes, title="Mode", loc="upper left", frameon=False)

    metric_handles = [
        Patch(facecolor="white", edgecolor="black", hatch=metric_hatches.get(mi, ""))
        for mi in range(M)
    ]
    ax.add_artist(leg1)
    ax.legend(metric_handles, legend_labels, title="Metric", loc="upper right", frameon=False)

    fig.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    print(f"[saved] {outfile}")

def make_plots_for_N_merged_modes(
    df: pd.DataFrame, N: int, out_dir: Path, threads_value: int
):
    dN = df[(df["N"] == N) & (df["Threads"] == threads_value)]
    if dN.empty:
        print(f"[warn] No data for N={N} at Threads={threads_value}")
        return

    def any_notna(cols: List[str], frame: pd.DataFrame) -> bool:
        cols_present = [c for c in cols if c in frame.columns]
        return frame[cols_present].notna().any().any() if cols_present else False

    # Step 1
    d1 = dN[dN["Step"] == 1]
    if not d1.empty:
        if any_notna(["H2D","UMDev","D2H","UMHost"], d1):
            grouped_bar_plot_by_kpct_modes(
                d1,
                title=f"ALL MODES | Threads={threads_value} | N={N} | Step=1 | Transfers (ms): H2D / UM→Dev / D2H / UM→Host",
                ylabel="ms",
                outfile=out_dir / f"ALL_N{N}_T{threads_value}_step1_transfers_ms.png",
                metrics=["H2D","UMDev","D2H","UMHost"],
                legend_labels=["H2D","UM→Dev","D2H","UM→Host"],
            )
        if any_notna(["GPUms","CPUms"], d1):
            grouped_bar_plot_by_kpct_modes(
                d1,
                title=f"ALL MODES | Threads={threads_value} | N={N} | Step=1 | Compute (ms): GPU / CPU",
                ylabel="ms",
                outfile=out_dir / f"ALL_N{N}_T{threads_value}_step1_compute_ms.png",
                metrics=["GPUms","CPUms"],
                legend_labels=["GPU(ms)","CPU(ms)"],
            )
        if any_notna(["GPUgbps","CPUgbps"], d1):
            grouped_bar_plot_by_kpct_modes(
                d1,
                title=f"ALL MODES | Threads={threads_value} | N={N} | Step=1 | Compute (GB/s): GPU / CPU",
                ylabel="GB/s",
                outfile=out_dir / f"ALL_N{N}_T{threads_value}_step1_compute_gbps.png",
                metrics=["GPUgbps","CPUgbps"],
                legend_labels=["GPU(GB/s)","CPU(GB/s)"],
            )
        summary_metrics = ["End2End"]
        summary_labels  = ["End-to-End"]
        if "FullE2E" in d1.columns and d1["FullE2E"].notna().any():
            summary_metrics = ["End2End", "FullE2E"]
            summary_labels  = ["End-to-End", "FullE2E (excl. checksum)"]
        if any_notna(summary_metrics, d1):
            grouped_bar_plot_by_kpct_modes(
                d1,
                title=f"ALL MODES | Threads={threads_value} | N={N} | Step=1 | Summary (ms): " + " / ".join(summary_labels),
                ylabel="ms",
                outfile=out_dir / f"ALL_N{N}_T{threads_value}_step1_summary_ms.png",
                metrics=summary_metrics,
                legend_labels=summary_labels,
            )

    # Step 2
    d2 = dN[dN["Step"] == 2]
    if not d2.empty:
        if any_notna(["H2D","UMDev","D2H","UMHost"], d2):
            grouped_bar_plot_by_kpct_modes(
                d2,
                title=f"ALL MODES | Threads={threads_value} | N={N} | Step=2 | Transfers (ms): H2D / UM→Dev / D2H / UM→Host",
                ylabel="ms",
                outfile=out_dir / f"ALL_N{N}_T{threads_value}_step2_transfers_ms.png",
                metrics=["H2D","UMDev","D2H","UMHost"],
                legend_labels=["H2D","UM→Dev","D2H","UM→Host"],
            )
        if any_notna(["GPUms","CPUms"], d2):
            grouped_bar_plot_by_kpct_modes(
                d2,
                title=f"ALL MODES | Threads={threads_value} | N={N} | Step=2 | Compute (ms): GPU / CPU",
                ylabel="ms",
                outfile=out_dir / f"ALL_N{N}_T{threads_value}_step2_compute_ms.png",
                metrics=["GPUms","CPUms"],
                legend_labels=["GPU(ms)","CPU(ms)"],
            )
        if any_notna(["GPUgbps","CPUgbps"], d2):
            grouped_bar_plot_by_kpct_modes(
                d2,
                title=f"ALL MODES | Threads={threads_value} | N={N} | Step=2 | Compute (GB/s): GPU / CPU",
                ylabel="GB/s",
                outfile=out_dir / f"ALL_N{N}_T{threads_value}_step2_compute_gbps.png",
                metrics=["GPUgbps","CPUgbps"],
                legend_labels=["GPU(GB/s)","CPU(GB/s)"],
            )
        summary_metrics = ["End2End"]
        summary_labels  = ["End-to-End"]
        if "FullE2E" in d2.columns and d2["FullE2E"].notna().any():
            summary_metrics = ["End2End", "FullE2E"]
            summary_labels  = ["End-to-End", "FullE2E (excl. checksum)"]
        if any_notna(summary_metrics, d2):
            grouped_bar_plot_by_kpct_modes(
                d2,
                title=f"ALL MODES | Threads={threads_value} | N={N} | Step=2 | Summary (ms): " + " / ".join(summary_labels),
                ylabel="ms",
                outfile=out_dir / f"ALL_N{N}_T{threads_value}_step2_summary_ms.png",
                metrics=summary_metrics,
                legend_labels=summary_labels,
            )

def run_merged_across_modes(df_all: pd.DataFrame, out_root: Path, threads_value: int):
    # Create a single set of plots merging all modes, for each N
    subdir = out_root / "_ALL_MODES_"
    print("=== Merged across ALL modes ===")
    for N in (2048, 4096, 8192):
        make_plots_for_N_merged_modes(df_all, N, subdir, threads_value)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("manual.txt"), help="Path to manual.txt")
    ap.add_argument("--out",   type=Path, default=Path("plots"),     help="Output directory for PNGs")
    ap.add_argument("--threads", type=int, default=72,               help="Threads value to lock to (X axis becomes K%)")
    args = ap.parse_args()

    df = parse_manual(args.input)
    args.out.mkdir(parents=True, exist_ok=True)

    # We now only produce merged (all modes) plots for the specified Threads value
    run_merged_across_modes(df, args.out, args.threads)

    print(f"Done. Output: {args.out.resolve()}")

if __name__ == "__main__":
    main()

