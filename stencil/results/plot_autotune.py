#!/usr/bin/env python3
"""
Plot a rich set of charts from an autotune.txt results file.

Usage:
  python plot_autotune.py autotune.txt
  # optional:
  python plot_autotune.py autotune.txt --save-csv autotune_parsed.csv --outdir figs --no-show

The script will:
- Parse the text into a tidy DataFrame
- Generate multiple informative Matplotlib figures
- Save images to --outdir (default: ./figs) and (by default) display them
"""
import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROW_PATTERN = re.compile(
    r"^(?P<Mode>[a-zA-Z0-9_]+)\s+"
    r"(?P<N>\d+)\s+"
    r"(?P<Step>\d+)\s+"
    r"(?P<Iters>\d+)\s+"
    r"(?P<Ksel>\d+)\s+"
    r"(?P<ThrSel>\d+)\s+"
    r"(?P<Total>[0-9.]+)\s+"
    r"(?P<Cold>[0-9.]+|--)\s+"
    r"(?P<Warm>[0-9.]+|--)\s+"
    r"(?P<Read>[0-9.]+|--)\s+"
    r"(?P<UMback>[0-9.]+|--)",
    re.MULTILINE
)

def parse_file(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # Normalize values like ".534" -> "0.534" to be float-friendly
    norm_text = re.sub(r"(?<!\d)\.(\d)", r"0.\1", text) # Fixed regex for dot numbers

    rows = []
    for m in ROW_PATTERN.finditer(norm_text):
        d = m.groupdict()
        for key in ["N", "Step", "Iters", "Ksel", "ThrSel"]:
            d[key] = int(d[key])
        for key in ["Total", "Cold", "Warm", "Read", "UMback"]:
            val = d[key]
            d[key] = np.nan if val == "--" else float(val)
        rows.append(d)

    if not rows:
        raise ValueError("No rows matched. Check the file format or adjust the regex.")

    df = pd.DataFrame(rows).sort_values(by=["N", "Step", "Mode"]).reset_index(drop=True)
    return df

def ensure_outdir(outdir: str) -> Path:
    p = Path(outdir)
    p.mkdir(parents=True, exist_ok=True)
    return p

def savefig(outdir: Path, name: str):
    # Do not set any explicit style/colors so plots remain portable
    safe = re.sub(r"[^a-zA-Z0-9_.-]", "_", name) + ".png"
    plt.tight_layout()
    plt.savefig(outdir / safe, dpi=160)


def plot_all(df: pd.DataFrame, outdir: Path, show: bool = True):
    # --- Derived helpers for clearer plots ---
    # Ksel is a row index ('K'), which is meaningful relative to N. Add Kpct = 100*K/N.
    df = df.copy()
    # Ensure N is not zero before division
    df['Kpct'] = (100.0 * df['Ksel'] / df['N'].replace(0, np.nan)).round(1)
    
    modes = list(df["Mode"].unique())
    steps = sorted(df["Step"].unique())
    Ns = sorted(df["N"].unique())
    time_parts = ["Cold", "Warm", "Read", "UMback"]

    # 1) Total vs N for each Step (one figure per Step; lines for each Mode)
    for step in steps:
        sub = df[df["Step"] == step]
        if sub.empty:
            continue
        plt.figure()
        for mode in modes:
            sub_m = sub[sub["Mode"] == mode]
            if sub_m.empty:
                continue
            plt.plot(sub_m["N"], sub_m["Total"], marker="o", label=mode)
        plt.xlabel("N")
        plt.ylabel("Total (ms)")
        plt.title(f"Total(ms) vs N by Mode — Step {step}")
        plt.legend(loc="best")
        savefig(outdir, f"total_vs_n_step_{step}")
        if show: plt.show(); plt.close()

    # 2) Grouped bars: Total per Mode for each (N, Step)
    for N in Ns:
        for step in steps:
            sub = df[(df["N"] == N) & (df["Step"] == step)]
            if sub.empty: 
                continue
            plt.figure()
            x = np.arange(len(sub))
            plt.bar(x, sub["Total"])
            plt.xticks(x, sub["Mode"], rotation=45, ha="right")
            plt.ylabel("Total (ms)")
            plt.title(f"Total(ms) by Mode — N={N}, Step={step}")
            savefig(outdir, f"total_by_mode_N{N}_step_{step}")
            if show: plt.show(); plt.close()

    # 3) Stacked bars: Cold/Warm/Read/UMback per Mode for each (N, Step)
    for N in Ns:
        for step in steps:
            sub = df[(df["N"] == N) & (df["Step"] == step)].copy()
            if sub.empty:
                continue
            plt.figure()
            x = np.arange(len(sub))
            bottom = np.zeros(len(sub))
            for part in time_parts:
                vals = sub[part].fillna(0.0).values
                plt.bar(x, vals, bottom=bottom, label=part)
                bottom += vals
            plt.xticks(x, sub["Mode"], rotation=45, ha="right")
            plt.ylabel("Time (ms)")
            plt.title(f"Time Breakdown — N={N}, Step={step}")
            plt.legend(loc="best")
            savefig(outdir, f"time_breakdown_N{N}_step_{step}")
            if show: plt.show(); plt.close()

    # 4) Scatter: Ksel vs Total and ThrSel vs Total per Step (grouped by Mode with legend)
    for step in steps:
        sub = df[df["Step"] == step]
        if sub.empty:
            continue
        
        # Ksel vs Total
        plt.figure()
        for mode in modes:
            sub_m = sub[sub["Mode"] == mode]
            # Use scatter plots with legend, instead of annotations
            plt.scatter(sub_m["Ksel"], sub_m["Total"], label=mode)
        plt.xlabel("Ksel")
        plt.ylabel("Total (ms)")
        plt.title(f"Ksel vs Total(ms) — Step {step} (grouped by Mode)")
        plt.legend(loc="best")
        savefig(outdir, f"ksel_vs_total_step_{step}")
        if show: plt.show(); plt.close()

        # ThrSel vs Total
        plt.figure()
        for mode in modes:
            sub_m = sub[sub["Mode"] == mode]
            # Use scatter plots with legend, instead of annotations
            plt.scatter(sub_m["ThrSel"], sub_m["Total"], label=mode)
        plt.xlabel("ThrSel (CPU threads)")
        plt.ylabel("Total (ms)")
        plt.title(f"ThrSel vs Total(ms) — Step {step} (grouped by Mode)")
        plt.legend(loc="best")
        savefig(outdir, f"thrsel_vs_total_step_{step}")
        if show: plt.show(); plt.close()

    # --- NEW A) Total vs K% (one figure per Step; lines by ThrSel) ---
    for step in steps:
        sub = df[df['Step'] == step]
        if sub.empty:
            continue
        plt.figure()
        for thr in sorted(sub['ThrSel'].unique()):
            sthr = sub[sub['ThrSel'] == thr].sort_values('Kpct')
            if sthr.empty:
                continue
            plt.plot(sthr['Kpct'], sthr['Total'], marker='o', label=f'thr={thr}')
        plt.xlabel('K (percent of N)')
        plt.ylabel('Total (ms)')
        plt.title(f'Total vs K% — Step {step}')
        plt.legend(loc='best')
        # Removed annotate_best_point for cleaner plot
        savefig(outdir, f'total_vs_Kpct_step_{step}')
        if show: plt.show(); plt.close()

    # --- NEW B) Total vs CPU threads (one figure per Step; lines by K%) ---
    for step in steps:
        sub = df[df['Step'] == step]
        if sub.empty:
            continue
        # limit unique K% curves if there are many: choose sorted uniques
        kcurves = sorted(sub['Kpct'].unique())
        plt.figure()
        for kp in kcurves:
            sk = sub[sub['Kpct'] == kp].sort_values('ThrSel')
            if sk.empty:
                continue
            plt.plot(sk['ThrSel'], sk['Total'], marker='o', label=f'K≈{kp}%')
        plt.xlabel('CPU threads')
        plt.ylabel('Total (ms)')
        plt.title(f'Total vs CPU threads — Step {step}')
        plt.legend(loc='best')
        # Removed annotate_best_point for cleaner plot
        savefig(outdir, f'total_vs_threads_step_{step}')
        if show: plt.show(); plt.close()

    # --- NEW C) Heatmap: Total(ms) across (K% × threads) per Step ---
    for step in steps:
        sub = df[df['Step'] == step]
        if sub.empty:
            continue
        # Aggregate if duplicates exist: take min Total for each (ThrSel, Kpct)
        pivot = sub.pivot_table(index='ThrSel', columns='Kpct', values='Total', aggfunc='min')
        if pivot.empty:
            continue
        plt.figure()
        im = plt.imshow(pivot.values, aspect='auto', origin='lower')
        plt.colorbar(im, label='Total (ms)')
        plt.xticks(range(len(pivot.columns)), [str(c) for c in pivot.columns], rotation=45, ha='right')
        plt.yticks(range(len(pivot.index)), [str(i) for i in pivot.index])
        plt.xlabel('K (percent of N)')
        plt.ylabel('CPU threads')
        plt.title(f'Heatmap: Total(ms) vs K% × threads — Step {step}')
        savefig(outdir, f'heatmap_total_Kpct_threads_step_{step}')
        if show: plt.show(); plt.close()


    # --- NEW D) Best (K%, threads) per Mode (across all N/Steps) ---
    # Pick the minimum Total row for each Mode.
    best_mode = df.sort_values('Total').groupby('Mode', as_index=False).first()
    if not best_mode.empty:
        # Save a CSV summary of the winners
        best_csv = outdir / 'best_by_mode.csv'
        best_mode[['Mode','N','Step','Ksel','Kpct','ThrSel','Total','Cold','Warm','Read','UMback']].to_csv(best_csv, index=False)

        # Scatter: K% vs threads, one point per Mode, annotated with Mode and Total
        plt.figure()
        plt.scatter(best_mode['Kpct'], best_mode['ThrSel'])
        for _, r in best_mode.iterrows():
            # Kept this annotation as it summarizes overall bests and requires specific info
            label = f"{r['Mode']}\ntotal={r['Total']:.3f} ms"
            plt.annotate(label, (r['Kpct'], r['ThrSel']), fontsize=8, xytext=(6,6), textcoords='offset points')
        plt.xlabel('K (percent of N)')
        plt.ylabel('CPU threads')
        plt.title('Best (K%, threads) per Mode — overall')
        savefig(outdir, 'best_Kpct_threads_per_mode_overall')
        if show: plt.show(); plt.close()

    # 5) Read(ms) vs Total(ms) per Step as lines per Mode
    for step in steps:
        sub = df[df["Step"] == step].copy()
        if sub.empty:
            continue
        plt.figure()
        for mode in modes:
            sub_m = sub[sub["Mode"] == mode]
            if sub_m.empty:
                continue
            plt.plot(sub_m["Read"], sub_m["Total"], marker="o", label=mode)
        plt.xlabel("Read (ms)")
        plt.ylabel("Total (ms)")
        plt.title(f"Total vs Read — Step {step}")
        plt.legend(loc="best")
        savefig(outdir, f"total_vs_read_step_{step}")
        if show: plt.show(); plt.close()

    # 6) UMback(ms) vs N — Combined Modes and Steps (Line plot replacing original bars)
    # Merges all UMback data, N, and Steps into one plot with a legend.
    sub = df[df["UMback"].notna()].copy()
    if not sub.empty:
        plt.figure()
        # Create a combined category for the legend
        sub['Mode_Step'] = sub['Mode'] + ' (Step ' + sub['Step'].astype(str) + ')'
        
        # Determine the unique line/category combinations
        categories = sub['Mode_Step'].unique()
        
        # Plot UMback vs N for each unique combination
        for category in categories:
            cat_data = sub[sub['Mode_Step'] == category].sort_values('N')
            if not cat_data.empty:
                # Use a line and marker for better differentiation
                plt.plot(cat_data["N"], cat_data["UMback"], marker="o", linestyle='-', label=category)

        plt.xlabel("N")
        plt.ylabel("UMback (ms)")
        plt.title(f"UMback(ms) vs N — Combined Modes and Steps")
        plt.legend(loc="best", fontsize=8)
        savefig(outdir, f"umback_vs_n_combined")
        if show: plt.show(); plt.close()


def main():
    ap = argparse.ArgumentParser(description="Plot autotune results with many informative charts.")
    ap.add_argument("input", help="Path to autotune.txt (results file)")
    ap.add_argument("--save-csv", help="Optional path to save parsed CSV", default=None)
    ap.add_argument("--outdir", help="Directory to save figures", default="figs")
    ap.add_argument("--no-show", action="store_true", help="Do not display windows; just save figures")
    args = ap.parse_args()

    try:
        df = parse_file(args.input)
    except ValueError as e:
        print(f"Error parsing file: {e}")
        return

    if args.save_csv:
        df.to_csv(args.save_csv, index=False)
        print(f"Saved parsed CSV to: {args.save_csv}")

    outdir = ensure_outdir(args.outdir)
    plot_all(df, outdir=outdir, show=not args.no_show)

    print(f"Done. Figures saved under: {outdir.resolve()}")
    print(f"Rows parsed: {len(df)} | Modes: {df['Mode'].nunique()} | Steps: {df['Step'].nunique()} | Ns: {df['N'].nunique()}")

if __name__ == "__main__":
    main()
