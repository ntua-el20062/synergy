#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import argparse
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

THREADS_ORDER = [1, 2, 4, 8, 16, 32, 64]
K_ORDER       = [0, 10, 25, 30, 50, 70, 80]

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
        "FullE2E"
    ]
    for c in numeric_cols:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c].astype(str).str.replace("--", "nan"), errors="coerce")

    df["Mode"] = df["Mode"].astype(str)

    # Keep only K/Threads of interest (optional)
    df = df[df["Threads"].isin(THREADS_ORDER) & df["Kpct"].isin(K_ORDER)]

    # Average duplicates for stability (same (Mode,N,Step,Threads,Kpct))
    key = ["Mode","N","Step","Threads","Kpct"]
    df = df.groupby(key, as_index=False).mean(numeric_only=True)

    return df

def grouped_bar_plot(
    data: pd.DataFrame,
    title: str,
    ylabel: str,
    outfile: Path,
    metrics: List[str],
    legend_labels: Optional[List[str]] = None,
):
    """Grouped bars: X=Threads; color=K%; hatch=metric."""
    from matplotlib.patches import Patch

    threads = [t for t in THREADS_ORDER if t in data["Threads"].unique()]
    ks      = [k for k in K_ORDER       if k in data["Kpct"].unique()]
    if not threads or not ks:
        print(f"[skip] Nothing to plot for: {title}")
        return

    M, T, K = len(metrics), len(threads), len(ks)
    if legend_labels is None:
        legend_labels = metrics

    vals = np.full((T, K, M), np.nan, dtype=float)
    for ti, t in enumerate(threads):
        for ki, k in enumerate(ks):
            row = data[(data["Threads"] == t) & (data["Kpct"] == k)]
            if not row.empty:
                for mi, m in enumerate(metrics):
                    if m in row.columns:
                        vals[ti, ki, mi] = pd.to_numeric(row[m], errors="coerce").mean(skipna=True)

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get(
        "color", ["C0","C1","C2","C3","C4","C5","C6","C7","C8","C9"]
    )
    k_colors = [color_cycle[i % len(color_cycle)] for i in range(K)]

    # Support up to 4 metrics by default; extend if you add more
    metric_hatches = {0: "", 1: "//", 2: "xx", 3: "++"}

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(T, dtype=float)

    group_width   = 0.82
    width_per_k   = group_width / K
    width_per_bar = width_per_k / M if M > 0 else width_per_k

    for ki, k in enumerate(ks):
        k_color = k_colors[ki]
        k_group_start = -group_width / 2 + ki * width_per_k
        for mi, m in enumerate(metrics):
            offset = k_group_start + mi * width_per_bar + width_per_bar / 2
            xpos = x + offset
            heights = vals[:, ki, mi]
            ax.bar(
                xpos,
                heights,
                width=width_per_bar * 0.95,
                color=k_color,
                edgecolor="black",
                linewidth=0.5,
                hatch=metric_hatches.get(mi, ""),
            )

    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in threads])
    ax.set_xlabel("Threads")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    from matplotlib.patches import Patch
    k_handles = [Patch(facecolor=k_colors[ki], edgecolor="black") for ki in range(K)]
    k_labels  = [f"{k}%" for k in ks]
    leg1 = ax.legend(k_handles, k_labels, title="K%", loc="upper left", frameon=False)

    m_handles = [
        Patch(facecolor="white", edgecolor="black", hatch=metric_hatches.get(mi, ""))
        for mi in range(M)
    ]
    ax.add_artist(leg1)
    ax.legend(m_handles, legend_labels, title="Metric", loc="upper right", frameon=False)

    fig.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    print(f"[saved] {outfile}")


def make_plots_for_N(df: pd.DataFrame, N: int, out_dir: Path, mode_label: str):
    dN = df[df["N"] == N]
    if dN.empty:
        print(f"[warn] No data for N={N} in mode {mode_label}")
        return

    def any_notna(cols: List[str], frame: pd.DataFrame) -> bool:
        cols_present = [c for c in cols if c in frame.columns]
        return frame[cols_present].notna().any().any() if cols_present else False

    # Step 1
    d1 = dN[dN["Step"] == 1]
    if not d1.empty:
        if any_notna(["H2D","UMDev","D2H","UMHost"], d1):
            grouped_bar_plot(
                d1,
                title=f"{mode_label} | N={N} | Step=1 | Transfers (ms): H2D / UM→Dev / D2H / UM→Host",
                ylabel="ms",
                outfile=out_dir / f"{mode_label}_N{N}_step1_transfers_ms.png",
                metrics=["H2D","UMDev","D2H","UMHost"],
                legend_labels=["H2D","UM→Dev","D2H","UM→Host"],
            )
        if any_notna(["GPUms","CPUms"], d1):
            grouped_bar_plot(
                d1,
                title=f"{mode_label} | N={N} | Step=1 | Compute (ms): GPU / CPU",
                ylabel="ms",
                outfile=out_dir / f"{mode_label}_N{N}_step1_compute_ms.png",
                metrics=["GPUms","CPUms"],
                legend_labels=["GPU(ms)","CPU(ms)"],
            )
        if any_notna(["GPUgbps","CPUgbps"], d1):
            grouped_bar_plot(
                d1,
                title=f"{mode_label} | N={N} | Step=1 | Compute (GB/s): GPU / CPU",
                ylabel="GB/s",
                outfile=out_dir / f"{mode_label}_N{N}_step1_compute_gbps.png",
                metrics=["GPUgbps","CPUgbps"],
                legend_labels=["GPU(GB/s)","CPU(GB/s)"],
            )
        # Summary: include FullE2E when present; no "Read"
        summary_metrics = ["End2End"]
        summary_labels  = ["End-to-End"]
        if "FullE2E" in d1.columns and d1["FullE2E"].notna().any():
            summary_metrics = ["End2End", "FullE2E"]
            summary_labels  = ["End-to-End", "FullE2E (excl. checksum)"]
        if any_notna(summary_metrics, d1):
            grouped_bar_plot(
                d1,
                title=f"{mode_label} | N={N} | Step=1 | Summary (ms): " + " / ".join(summary_labels),
                ylabel="ms",
                outfile=out_dir / f"{mode_label}_N{N}_step1_summary_ms.png",
                metrics=summary_metrics,
                legend_labels=summary_labels,
            )

    # Step 2
    d2 = dN[dN["Step"] == 2]
    if not d2.empty:
        if any_notna(["H2D","UMDev","D2H","UMHost"], d2):
            grouped_bar_plot(
                d2,
                title=f"{mode_label} | N={N} | Step=2 | Transfers (ms): H2D / UM→Dev / D2H / UM→Host",
                ylabel="ms",
                outfile=out_dir / f"{mode_label}_N{N}_step2_transfers_ms.png",
                metrics=["H2D","UMDev","D2H","UMHost"],
                legend_labels=["H2D","UM→Dev","D2H","UM→Host"],
            )
        if any_notna(["GPUms","CPUms"], d2):
            grouped_bar_plot(
                d2,
                title=f"{mode_label} | N={N} | Step=2 | Compute (ms): GPU / CPU",
                ylabel="ms",
                outfile=out_dir / f"{mode_label}_N{N}_step2_compute_ms.png",
                metrics=["GPUms","CPUms"],
                legend_labels=["GPU(ms)","CPU(ms)"],
            )
        if any_notna(["GPUgbps","CPUgbps"], d2):
            grouped_bar_plot(
                d2,
                title=f"{mode_label} | N={N} | Step=2 | Compute (GB/s): GPU / CPU",
                ylabel="GB/s",
                outfile=out_dir / f"{mode_label}_N{N}_step2_compute_gbps.png",
                metrics=["GPUgbps","CPUgbps"],
                legend_labels=["GPU(GB/s)","CPU(GB/s)"],
            )
        # Summary: include FullE2E when present; no "Read"
        summary_metrics = ["End2End"]
        summary_labels  = ["End-to-End"]
        if "FullE2E" in d2.columns and d2["FullE2E"].notna().any():
            summary_metrics = ["End2End", "FullE2E"]
            summary_labels  = ["End-to-End", "FullE2E (excl. checksum)"]
        if any_notna(summary_metrics, d2):
            grouped_bar_plot(
                d2,
                title=f"{mode_label} | N={N} | Step=2 | Summary (ms): " + " / ".join(summary_labels),
                ylabel="ms",
                outfile=out_dir / f"{mode_label}_N{N}_step2_summary_ms.png",
                metrics=summary_metrics,
                legend_labels=summary_labels,
            )


def run_per_mode(df_all: pd.DataFrame, out_root: Path, modes: list):
    print(f"[info] Modes detected: {', '.join(modes)}")
    for m in modes:
        d = df_all[df_all["Mode"] == m]
        if d.empty:
            print(f"[skip] Mode={m} has no rows after filtering.")
            continue
        subdir = out_root / m
        print(f"=== Mode: {m} (rows: {len(d)}) ===")
        for N in (2048, 4096, 8192):
            make_plots_for_N(d, N, subdir, mode_label=m)


def run_combined(df_all: pd.DataFrame, out_root: Path):
    # One combined set across all modes (aggregated)
    subdir = out_root / "_ALL_"
    print("=== Combined across ALL modes ===")
    for N in (2048, 4096, 8192):
        make_plots_for_N(df_all, N, subdir, mode_label="ALL")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("manual.txt"), help="Path to manual.txt")
    ap.add_argument("--out",   type=Path, default=Path("plots"),     help="Output directory for PNGs")
    ap.add_argument("--mode",  type=str,  default=None,              help="Filter to a single Mode (exact match)")
    ap.add_argument("--all", action="store_true",     help="Also generate a combined set across all modes")
    args = ap.parse_args()

    df = parse_manual(args.input)
    modes_present = sorted(df["Mode"].unique().tolist())

    args.out.mkdir(parents=True, exist_ok=True)

    if args.mode:
        if args.mode not in modes_present:
            raise SystemExit(f"No rows match Mode='{args.mode}'. Modes present: {modes_present}")
        print(f"=== Mode: {args.mode} (rows: {len(df[df['Mode']==args.mode])}) ===")
        run_per_mode(df[df["Mode"] == args.mode], args.out, [args.mode])
    else:
        # Default: iterate per mode with subfolders; print what we're doing
        run_per_mode(df, args.out, modes_present)

    if args.all:
        run_combined(df, args.out)

    print(f"Done. Output: {args.out.resolve()}")

if __name__ == "__main__":
    main()

