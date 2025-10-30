#!/usr/bin/env python3
import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# --------- Parsing regexes ---------
HEADER_RE = re.compile(r'^N=(\d+)\s+iters=(\d+)(?:\s+frac=([0-9.]+))?')
MODE_RE = re.compile(r'^mode=([a-zA-Z0-9_]+)')
KERNEL_RE = re.compile(r'^Kernel avg:\s*([0-9.]+)\s*ms,\s*effective bandwidth:\s*([0-9.]+)\s*GB/s')
SECTION_TRANSPOSE = re.compile(r'^TRANSPOSE SWEEP')
ANSI_RE = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')

# --------- Parsing ---------
def parse_file(path: Path) -> pd.DataFrame:
    """Parse the metrics file and return a DataFrame for TRANSPOSE SWEEP only."""
    text = path.read_text(errors="ignore")
    text = ANSI_RE.sub('', text)  # strip ANSI color codes
    lines = [l.strip() for l in text.splitlines()]

    section = None
    current_N = None
    current_iters = None
    current_mode = None
    sweep_rows = []

    for line in lines:
        if SECTION_TRANSPOSE.search(line):
            section = "TRANSPOSE SWEEP"
            continue

        m = HEADER_RE.search(line)
        if m:
            current_N = int(m.group(1))
            current_iters = int(m.group(2))
            current_mode = None
            continue

        m = MODE_RE.search(line)
        if m:
            current_mode = m.group(1)
            continue

        m = KERNEL_RE.search(line)
        if m and section == "TRANSPOSE SWEEP" and current_mode is not None:
            sweep_rows.append({
                "N": current_N,
                "iters": current_iters,
                "mode": current_mode,
                "kernel_ms": float(m.group(1)),
                "bandwidth_GBps": float(m.group(2)),
            })
            continue

    return pd.DataFrame(sweep_rows)

# --------- CSV I/O ---------
def save_csv(sweep_df: pd.DataFrame, outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    sweep_csv = outdir / "transpose_sweep_metrics.csv"
    sweep_df.to_csv(sweep_csv, index=False)
    return sweep_csv

# --------- Helpers ---------
def _all_pairs(df, cols=("N","iters")):
    if df.empty:
        return []
    pairs = (
        df.drop_duplicates(list(cols))[list(cols)]
        .sort_values(list(cols))
        .itertuples(index=False, name=None)
    )
    return list(pairs)

def _pick_pair(sweep_df: pd.DataFrame, n: int = None, iters: int = None):
    if n is not None and iters is not None:
        return n, iters
    if sweep_df.empty:
        return None, None
    # pick the largest (N, iters) pair present
    largest_pair = (
        sweep_df.sort_values(["N", "iters"])
        .drop_duplicates(["N", "iters"], keep="last")[["N", "iters"]]
        .iloc[-1]
    )
    return int(largest_pair["N"]), int(largest_pair["iters"])

# --------- Plotting (bars only) ---------
def _bar_by_mode(sub: pd.DataFrame, value_col: str, title: str, ylabel: str, outpath: Path, show: bool):
    if sub.empty:
        return None
    sub = sub.sort_values(value_col, ascending=False)
    plt.figure()
    plt.bar(sub["mode"], sub[value_col])
    plt.title(title)
    plt.xlabel("Mode")
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    if show:
        plt.show()
    else:
        plt.close()
    return outpath

def plot_transpose_bandwidth_by_mode(sweep_df: pd.DataFrame, outdir: Path, n: int = None, iters: int = None, show=False):
    if sweep_df.empty:
        print("No TRANSPOSE SWEEP records found — skipping bandwidth plot.", file=sys.stderr)
        return None
    n, iters = _pick_pair(sweep_df, n, iters)
    subset = sweep_df[(sweep_df["N"] == n) & (sweep_df["iters"] == iters)].copy()
    if subset.empty:
        print(f"No TRANSPOSE data for N={n}, iters={iters} — skipping.", file=sys.stderr)
        return None
    outpath = outdir / f"transpose_bandwidth_by_mode_N{n}_iters{iters}.png"
    return _bar_by_mode(
        subset,
        value_col="bandwidth_GBps",
        title=f"Effective Bandwidth by Mode (TRANSPOSE, N={n}, iters={iters})",
        ylabel="Effective bandwidth (GB/s)",
        outpath=outpath,
        show=show,
    )

def plot_transpose_time_by_mode(sweep_df: pd.DataFrame, outdir: Path, n: int = None, iters: int = None, show=False):
    if sweep_df.empty:
        print("No TRANSPOSE SWEEP records found — skipping time plot.", file=sys.stderr)
        return None
    n, iters = _pick_pair(sweep_df, n, iters)
    subset = sweep_df[(sweep_df["N"] == n) & (sweep_df["iters"] == iters)].copy()
    if subset.empty:
        print(f"No TRANSPOSE data for N={n}, iters={iters} — skipping.", file=sys.stderr)
        return None
    outpath = outdir / f"transpose_time_by_mode_N{n}_iters{iters}.png"
    return _bar_by_mode(
        subset,
        value_col="kernel_ms",
        title=f"Kernel Time by Mode (TRANSPOSE, N={n}, iters={iters})",
        ylabel="Kernel time (ms)",
        outpath=outpath,
        show=show,
    )

# --------- CLI ---------
def main():
    p = argparse.ArgumentParser(description="Parse and plot TRANSPOSE SWEEP metrics (bars only).")
    p.add_argument("--input", "-i", required=True, type=Path, help="Path to results_hybrid.txt (or similar log)")
    p.add_argument("--outdir", "-o", type=Path, default=Path("plots"), help="Directory to write CSVs and PNGs")
    p.add_argument("--show", action="store_true", help="Show plots on screen after saving")
    p.add_argument("--all", action="store_true", help="Generate plots for ALL (N,iters) pairs in TRANSPOSE")

    # Transpose selection
    p.add_argument("--transpose-select", choices=["largest", "pair"], default="largest",
                   help="Which TRANSPOSE case to plot: largest N/iters found, or a specific pair")
    p.add_argument("--transpose-n", type=int, help="If --transpose-select pair, N value")
    p.add_argument("--transpose-iters", type=int, help="If --transpose-select pair, iters value")

    args = p.parse_args()

    sweep_df = parse_file(args.input)
    args.outdir.mkdir(parents=True, exist_ok=True)
    sweep_csv = save_csv(sweep_df, args.outdir)
    print(f"Wrote CSV: {sweep_csv}")

    if args.all:
        if not sweep_df.empty:
            for n, iters in _all_pairs(sweep_df, cols=("N","iters")):
                tp_bw = plot_transpose_bandwidth_by_mode(sweep_df, args.outdir, n=n, iters=iters, show=args.show)
                if tp_bw:
                    print(f"Saved TRANSPOSE bandwidth plot: {tp_bw}")
                tp_tm = plot_transpose_time_by_mode(sweep_df, args.outdir, n=n, iters=iters, show=args.show)
                if tp_tm:
                    print(f"Saved TRANSPOSE time plot: {tp_tm}")
        else:
            print("No TRANSPOSE SWEEP records found.")
        return

    # Single (largest or specified) case
    if args.transpose_select == "pair":
        if args.transpose_n is None or args.transpose_iters is None:
            print("--transpose-select pair requires --transpose-n and --transpose-iters", file=sys.stderr)
            return
        n, iters = args.transpose_n, args.transpose_iters
    else:
        n, iters = None, None  # will pick largest automatically

    tp_bw = plot_transpose_bandwidth_by_mode(sweep_df, args.outdir, n=n, iters=iters, show=args.show)
    if tp_bw:
        print(f"Saved TRANSPOSE bandwidth plot: {tp_bw}")
    tp_tm = plot_transpose_time_by_mode(sweep_df, args.outdir, n=n, iters=iters, show=args.show)
    if tp_tm:
        print(f"Saved TRANSPOSE time plot: {tp_tm}")

if __name__ == "__main__":
    main()

