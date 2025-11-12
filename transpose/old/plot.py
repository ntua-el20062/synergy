#!/usr/bin/env python3

import argparse
import re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

HEADER_RE = re.compile(r'^N=(\d+)\s+iters=(\d+)(?:\s+frac=([0-9.]+))?')
MODE_RE = re.compile(r'^mode=([a-zA-Z0-9_]+)')
ANSI_RE = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')

METRIC_RES = {
    # Prefetch / copies / pre-setup
    "UM prefetch-to-GPU": re.compile(r'^UM prefetch-to-GPU:\s*([0-9.]+)\s*ms'),
    "UM prefetch-to-CPU (post)": re.compile(r'^UM prefetch-to-CPU.*?:\s*([0-9.]+)\s*ms'),
    "H2D once": re.compile(r'^H2D once:\s*([0-9.]+)\s*ms'),
    "D2H avg per-iter": re.compile(r'^D2H avg per-iter:\s*([0-9.]+)\s*ms'),
    "Pre-setup alloc": re.compile(r'^Pre-setup alloc:\s*([0-9.]+)\s*ms'),
    "Pre-setup init": re.compile(r'^Pre-setup init:\s*([0-9.]+)\s*ms'),
    "Pre-setup total": re.compile(r'^Pre-setup total:\s*([0-9.]+)\s*ms'),
    "Pre Kernel Costs": re.compile(r'^\s*Pre Kernel Costs:\s*([0-9.]+)\s*ms'),

    # Main compute timings
    "GPU Kernel avg": re.compile(r'^GPU Kernel avg:\s*([0-9.]+)\s*ms'),
    "GPU Kernel total": re.compile(r'^GPU Kernel total:\s*([0-9.]+)\s*ms'),
    "CPU compute": re.compile(r'^CPU compute:\s*([0-9.]+)\s*ms'),
    "CPU compute total": re.compile(r'^CPU compute total:\s*([0-9.]+)\s*ms'),
    "Compute E2E (avg per-iter)": re.compile(r'^Compute E2E \(avg per-iter\):\s*([0-9.]+)\s*ms'),
    "Compute E2E (whole loop)": re.compile(r'^Compute E2E \(whole loop\):\s*([0-9.]+)\s*ms'),
    "Full    E2E": re.compile(r'^Full\s+E2E:\s*([0-9.]+)\s*ms'),
    "CPU read time": re.compile(r'CPU read time:\s*([0-9.]+)\s*ms'),

    # -------- Warm-up timings (single-iter and totals) --------
    # Single-iteration (first iter) warm-up lines — capture both "avg (first iter)" and "(first iter)"
    "WARMUP GPU Kernel (first iter)": re.compile(r'^WARMUP GPU Kernel \(first iter\):\s*([0-9.]+)\s*ms'),
    "WARMUP CPU compute (first iter)": re.compile(r'^WARMUP CPU compute (?:avg )?\(first iter\):\s*([0-9.]+)\s*ms'),
    "WARMUP D2H (first iter)": re.compile(r'^WARMUP D2H \(first iter\):\s*([0-9.]+)\s*ms'),
    "WARMUP Compute E2E (first iter)": re.compile(r'^WARMUP Compute E2E \(first iter\):\s*([0-9.]+)\s*ms'),

    # Two-iteration warm-up summaries (kept)
    "WARMUP GPU Kernel avg (2 iters)": re.compile(r'^WARMUP GPU Kernel avg \(2 iters\):\s*([0-9.]+)\s*ms'),
    "WARMUP CPU compute avg (2 iters)": re.compile(r'^WARMUP CPU compute avg \(2 iters\):\s*([0-9.]+)\s*ms'),
    "WARMUP D2H avg per-iter": re.compile(r'^WARMUP D2H avg per-iter:\s*([0-9.]+)\s*ms'),

    # Total warm-up wall-clock time
    "WARMUP total time": re.compile(r'^WARMUP total time:\s*([0-9.]+)\s*ms'),
}

BANDWIDTH_RES = {
    "GPU Kernel avg": re.compile(r'^GPU Kernel avg:\s*[0-9.]+\s*ms,\s*effective bandwidth:\s*([0-9.]+)\s*GB/s'),
    "GPU Kernel total": re.compile(r'^GPU Kernel total:\s*[0-9.]+\s*ms.*?([0-9.]+)\s*GB/s'),
    "CPU compute": re.compile(r'^CPU compute:\s*[0-9.]+\s*ms,\s*([0-9.]+)\s*GB/s'),
    "CPU compute total": re.compile(r'^CPU compute total:\s*[0-9.]+\s*ms.*?([0-9.]+)\s*GB/s'),
    "Compute E2E (avg per-iter)": re.compile(r'^Compute E2E \(avg per-iter\):.*?([0-9.]+)\s*GB/s'),
    "Full    E2E": re.compile(r'^Full\s+E2E:.*?([0-9.]+)\s*GB/s'),
    "CPU read time": re.compile(r'CPU read time:\s*[0-9.]+\s*ms.*?([0-9.]+)\s*GB/s'),
    "H2D once": re.compile(r'^H2D once:\s*[0-9.]+\s*ms.*?([0-9.]+)\s*GB/s'),
}


def parse_file(path: Path) -> pd.DataFrame:
    text = path.read_text(errors="ignore")
    text = ANSI_RE.sub('', text)
    lines = [l.strip() for l in text.splitlines()]

    current_N = None
    current_iters = None
    current_frac = None
    current_mode = None
    rows = []

    def emit(metric_name: str, time_ms: float, maybe_line: str):
        bw = None
        if metric_name in BANDWIDTH_RES:
            m_bw = BANDWIDTH_RES[metric_name].search(maybe_line)
            if m_bw:
                try:
                    bw = float(m_bw.group(1))
                except Exception:
                    bw = None
        rows.append({
            "N": current_N,
            "iters": current_iters,
            "frac": current_frac,
            "mode": current_mode,
            "metric": metric_name,
            "time_ms": float(time_ms),
            "bandwidth_GBps": bw,
        })

    for line in lines:
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
        if current_N is not None and current_iters is not None and current_mode is not None:
            for metric_name, rx in METRIC_RES.items():
                tm = rx.search(line)
                if tm:
                    emit(metric_name, float(tm.group(1)), line)
                    break

    return pd.DataFrame(rows)


def save_csv(df: pd.DataFrame, outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    p = outdir / "all_time_metrics_long.csv"
    df.to_csv(p, index=False)
    return p


def _all_pairs(df: pd.DataFrame):
    if df.empty:
        return []
    pairs = df.drop_duplicates(["N", "iters"])[["N", "iters"]].sort_values(["N", "iters"])  # type: ignore
    return [tuple(x) for x in pairs.to_records(index=False)]


def plot_metric_vs_frac(df: pd.DataFrame, outdir: Path, *, n: int, iters: int, metric: str, show=False):
    sub = df[(df["N"] == n) & (df["iters"] == iters) & (df["metric"] == metric)].copy()
    if sub.empty:
        print(f"No data for metric='{metric}' N={n} iters={iters}")
        return None

    pivot_time = sub.pivot_table(index="frac", columns="mode", values="time_ms", aggfunc="mean").sort_index()
    pivot_bw = sub.pivot_table(index="frac", columns="mode", values="bandwidth_GBps", aggfunc="mean").sort_index()

    fig, axes = plt.subplots(2, 1, figsize=(11, 10))

    if len(pivot_time.columns) == 0:
        return None
    bar_width = 0.8 / max(1, len(pivot_time.columns))
    x = list(range(len(pivot_time.index)))

    for i, col in enumerate(pivot_time.columns):
        axes[0].bar([xx + i * bar_width for xx in x], pivot_time[col], width=bar_width, label=col)
    axes[0].set_title(f"{metric} — Time vs CPU fraction (N={n}, iters={iters})")
    axes[0].set_xlabel("CPU fraction (frac)")
    axes[0].set_ylabel("Time (ms)")
    axes[0].set_xticks([xx + bar_width * (len(pivot_time.columns) - 1) / 2 for xx in x], pivot_time.index)
    axes[0].legend(ncol=2, fontsize=8)
    axes[0].grid(True, axis='y', alpha=0.3)

    if not pivot_bw.dropna(how='all').empty:
        for i, col in enumerate(pivot_bw.columns):
            axes[1].bar([xx + i * bar_width for xx in x], pivot_bw[col], width=bar_width, label=col)
        axes[1].set_title(f"{metric} — Bandwidth vs CPU fraction (N={n}, iters={iters})")
        axes[1].set_xlabel("CPU fraction (frac)")
        axes[1].set_ylabel("Bandwidth (GB/s)")
        axes[1].set_xticks([xx + bar_width * (len(pivot_time.columns) - 1) / 2 for xx in x], pivot_bw.index)
        axes[1].legend(ncol=2, fontsize=8)
        axes[1].grid(True, axis='y', alpha=0.3)

    plt.tight_layout()

    out = outdir / f"bars_{metric.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '-')}_N{n}_iters{iters}.png"
    plt.savefig(out, dpi=200)
    if show:
        plt.show()
    else:
        plt.close()
    return out


def plot_all_metrics_for_pair(df: pd.DataFrame, outdir: Path, *, n: int, iters: int, metrics: list[str], show=False):
    paths = []
    for m in metrics:
        p = plot_metric_vs_frac(df, outdir, n=n, iters=iters, metric=m, show=show)
        if p:
            print(f"Saved plot: {p}")
            paths.append(p)
    return paths


def list_available_metrics(df: pd.DataFrame) -> list[str]:
    return sorted(df["metric"].dropna().unique().tolist())


def main():
    p = argparse.ArgumentParser(description="Parse and plot ALL timing metrics as grouped bar charts (includes bandwidth where available).")
    p.add_argument("--input", "-i", required=True, type=Path, help="Path to results text file")
    p.add_argument("--outdir", "-o", type=Path, default=Path("plots"), help="Output directory")
    p.add_argument("--show", action="store_true", help="Show plots interactively")
    p.add_argument("--metrics", type=str, help="Comma-separated list of metric names to plot.")
    p.add_argument("--all", action="store_true", help="Generate plots for ALL (N,iters) pairs")
    p.add_argument("--n", type=int, help="Filter to a specific N")
    p.add_argument("--iters", type=int, help="Filter to a specific iters")
    args = p.parse_args()

    df = parse_file(args.input)
    if df.empty:
        print("No timing metrics parsed from file.")
        return

    csv_path = save_csv(df, args.outdir)
    print(f"Wrote CSV: {csv_path}")

    if args.metrics:
        metrics = [m.strip() for m in args.metrics.split(',') if m.strip()]
    else:
        metrics = list_available_metrics(df)
        print("Detected metrics:", ", ".join(metrics))

    if args.n is not None:
        df = df[df["N"] == args.n]
    if args.iters is not None:
        df = df[df["iters"] == args.iters]

    if args.all:
        pairs = _all_pairs(df)
        for n, iters in pairs:
            plot_all_metrics_for_pair(df, args.outdir, n=n, iters=iters, metrics=metrics, show=args.show)
    else:
        common_pair = df.groupby(["N", "iters"]).size().sort_values(ascending=False).index[0]
        n, iters = int(common_pair[0]), int(common_pair[1])
        plot_all_metrics_for_pair(df, args.outdir, n=n, iters=iters, metrics=metrics, show=args.show)


if __name__ == "__main__":
    main()

