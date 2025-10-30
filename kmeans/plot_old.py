#!/usr/bin/env python3
# parse_kmeans_bench.py
import argparse
import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def parse_benchmark_text(text: str) -> pd.DataFrame:
    """
    Parse the benchmark text into a tidy DataFrame with columns:
    dataset_size_MB, numObjs, numCoords, numClusters, block_size, impl, memory,
    t_alloc_ms, t_loop_total_ms, t_loop_avg_ms, t_loop_min_ms, t_loop_max_ms, t_gpu_ms
    """
    # Split by dataset sections
    sections = [s for s in text.split("dataset_size") if s.strip()]
    rows = []

    # Common helpers
    def get_float(label, s, alt=None):
        """
        Extract a float value in milliseconds after a label like 't_loop_avg' or 't_gpu'.
        If alt is provided, try that label if primary is missing.
        """
        for lab in [label] + ([alt] if alt else []):
            m = re.search(rf"{re.escape(lab)}\s*=\s*([0-9.]+)\s*ms", s)
            if m:
                return float(m.group(1))
        return None

    def get_int(label, s):
        m = re.search(rf"{re.escape(label)}\s*=\s*(\d+)", s)
        return int(m.group(1)) if m else None

    for sec in sections:
        # Header fields
        ds_mb = re.search(r"=\s*([0-9.]+)\s*MB", sec)
        bs = re.search(r"block_size\s*=\s*(\d+)", sec)
        numObjs = get_int("numObjs", sec)
        numCoords = get_int("numCoords", sec)
        numClusters = get_int("numClusters", sec)

        dataset_size_mb = float(ds_mb.group(1)) if ds_mb else None
        block_size = int(bs.group(1)) if bs else None

        # Sequential CPU block
        '''
        seq_block = re.search(
            r"\|-------------Sequential Kmeans-------------\|(.+?)\|-------------------------------------------\|",
            sec, flags=re.S | re.I
        )
        if seq_block:
            body = seq_block.group(1)
            rows.append(dict(
                dataset_size_MB=dataset_size_mb,
                numObjs=numObjs,
                numCoords=numCoords,
                numClusters=numClusters,
                block_size=block_size,
                impl="CPU Sequential",
                memory="CPU",
                t_alloc_ms=get_float("t_alloc", body),
                t_loop_total_ms=get_float("total", body),
                t_loop_avg_ms=get_float("t_loop_avg", body),
                t_loop_min_ms=get_float("t_loop_min", body),
                t_loop_max_ms=get_float("t_loop_max", body),
                t_gpu_ms=None
            ))
        '''
        # GPU (managed memory)
        managed = re.search(
            r"\|-----------Full-offload GPU KMeans \(managed memory\)------------\|(.+?)Performing validation",
            sec, flags=re.S | re.I
        )
        if managed:
            body = managed.group(1)
            rows.append(dict(
                dataset_size_MB=dataset_size_mb,
                numObjs=numObjs,
                numCoords=numCoords,
                numClusters=numClusters,
                block_size=block_size,
                impl="GPU KMeans (managed)",
                memory="Managed",
                t_alloc_ms=get_float("t_alloc", body),
                t_loop_total_ms=get_float("total", body),
                t_loop_avg_ms=get_float("t_loop_avg", body),
                t_loop_min_ms=get_float("t_loop_min", body),
                t_loop_max_ms=get_float("t_loop_max", body),
                t_gpu_ms=get_float("t_gpu", body)  # kernel time if reported
            ))

        # GPU (system-allocated)
        sysmem = re.search(
            r"\|-----------Full-offload GPU KMeans \(system allocated memory\)------------\|(.+?)Performing validation",
            sec, flags=re.S | re.I
        )
        if sysmem:
            body = sysmem.group(1)
            rows.append(dict(
                dataset_size_MB=dataset_size_mb,
                numObjs=numObjs,
                numCoords=numCoords,
                numClusters=numClusters,
                block_size=block_size,
                impl="GPU KMeans (system alloc)",
                memory="System",
                t_alloc_ms=get_float("t_alloc", body),
                t_loop_total_ms=get_float("total", body),
                t_loop_avg_ms=get_float("t_loop_avg", body),
                t_loop_min_ms=get_float("t_loop_min", body),
                t_loop_max_ms=get_float("t_loop_max", body),
                t_gpu_ms=get_float("t_gpu", body)
            ))

        # GPU (explicit transfers section labelled "Full-offload GPU Kmeans")
        explicit = re.search(
            r"\|-----------Full-offload GPU Kmeans------------\|(.+?)\|-------------------------------------------\|",
            sec, flags=re.S | re.I
        )
        if explicit:
            body = explicit.group(1)
            # In this section, the GPU time label is 't_gpu_avg'
            t_gpu_avg = get_float("t_gpu_avg", body)
            rows.append(dict(
                dataset_size_MB=dataset_size_mb,
                numObjs=numObjs,
                numCoords=numCoords,
                numClusters=numClusters,
                block_size=block_size,
                impl="GPU KMeans (explicit)",
                memory="Explicit",
                t_alloc_ms=get_float("t_alloc", body),
                t_loop_total_ms=get_float("total", body),
                t_loop_avg_ms=get_float("t_loop_avg", body),
                t_loop_min_ms=get_float("t_loop_min", body),
                t_loop_max_ms=get_float("t_loop_max", body),
                t_gpu_ms=t_gpu_avg
            ))

    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (impl, block_size): mean of metrics (handles duplicates)."""
    agg = (df.groupby(["impl", "block_size"], as_index=False)
             .agg(t_loop_avg_ms=("t_loop_avg_ms", "mean"),
                  t_gpu_ms=("t_gpu_ms", "mean")))
    return agg.sort_values(["impl", "block_size"])


def compute_speedup_vs_cpu(df_agg):
    """Pivot and compute speedup relative to CPU Sequential per block size."""
    pivot = (
        df_agg.pivot(index="block_size", columns="impl", values="t_loop_avg_ms")
        .sort_index()
    )
    if "CPU Sequential" not in pivot.columns:
        return pd.DataFrame()  # no CPU baseline found

    cpu = pivot["CPU Sequential"]

    # Optional: per-impl ms table without collision (if you want to save it too)
    ms_table = cpu.to_frame(name="CPU Sequential ms").join(
        pivot.drop(columns=["CPU Sequential"]).rename(columns=lambda c: f"{c} ms")
    )

    # Speedup table (×) = CPU_time / impl_time, exclude CPU to avoid 1× column
    speedup = cpu.to_frame(name="CPU ms").join(
        (cpu / pivot.drop(columns=["CPU Sequential"]))
        .rename(columns=lambda c: f"{c} speedup×")
    )

    # If you only need the speedup CSV, return that:
    return speedup.reset_index()


def make_plots(df_agg: pd.DataFrame, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Runtime vs block size per implementation
    plt.figure()
    for impl, grp in df_agg.sort_values("block_size").groupby("impl"):
        plt.plot(grp["block_size"], grp["t_loop_avg_ms"], marker="o", label=impl)
    plt.xlabel("Block size")
    plt.ylabel("Average per-iteration time (ms)")
    plt.title("KMeans runtime vs block size")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(outdir / "kmeans_runtime_vs_blocksize.png", bbox_inches="tight")
    plt.close()

    # 2) Speedup vs CPU (×)
    pivot = df_agg.pivot(index="block_size", columns="impl", values="t_loop_avg_ms").sort_index()
    if "CPU Sequential" in pivot.columns:
        cpu = pivot["CPU Sequential"]
        plt.figure()
        for col in pivot.columns:
            if col == "CPU Sequential":
                continue
            plt.plot(pivot.index, cpu / pivot[col], marker="o", label=col)
        plt.axhline(1.0, linestyle="--")
        plt.xlabel("Block size")
        plt.ylabel("Speedup over CPU (×)")
        plt.title("Speedup vs CPU by block size")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(outdir / "kmeans_speedup_vs_cpu.png", bbox_inches="tight")
        plt.close()

    # 3) GPU kernel time (if present)
    gpu_only = df_agg.dropna(subset=["t_gpu_ms"])
    if not gpu_only.empty:
        plt.figure()
        for impl, grp in gpu_only.sort_values("block_size").groupby("impl"):
            plt.plot(grp["block_size"], grp["t_gpu_ms"], marker="o", label=f"{impl} (t_gpu)")
        plt.xlabel("Block size")
        plt.ylabel("Reported GPU time (ms)")
        plt.title("GPU kernel time vs block size")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(outdir / "kmeans_gpu_internal_time.png", bbox_inches="tight")
        plt.close()


def main():
    ap = argparse.ArgumentParser(description="Parse and plot KMeans benchmark results.")
    ap.add_argument("input", type=Path, help="Path to results text file")
    ap.add_argument("--outdir", type=Path, default=None, help="Directory for CSVs/plots (default: alongside input)")
    args = ap.parse_args()

    outdir = args.outdir or args.input.parent

    text = args.input.read_text(encoding="utf-8", errors="ignore")
    df_raw = parse_benchmark_text(text)
    if df_raw.empty:
        raise SystemExit("No results parsed — check the input file format.")

    # Save raw
    raw_csv = outdir / "kmeans_timings_raw.csv"
    df_raw.to_csv(raw_csv, index=False)

    # Aggregate and save
    df_agg = aggregate(df_raw)
    agg_csv = outdir / "kmeans_timings_agg.csv"
    df_agg.to_csv(agg_csv, index=False)

    # Speedup table and save
    sp = compute_speedup_vs_cpu(df_agg)
    if not sp.empty:
        sp.to_csv(outdir / "kmeans_speedup_vs_cpu.csv", index=False)

    # Plots
    make_plots(df_agg, outdir)

    print(f"Wrote:\n  - {raw_csv}\n  - {agg_csv}")
    if not sp.empty:
        print(f"  - {outdir / 'kmeans_speedup_vs_cpu.csv'}")
    print(f"  - {outdir / 'kmeans_runtime_vs_blocksize.png'}")
    if (outdir / "kmeans_speedup_vs_cpu.png").exists():
        print(f"  - {outdir / 'kmeans_speedup_vs_cpu.png'}")
    if (outdir / "kmeans_gpu_internal_time.png").exists():
        print(f"  - {outdir / 'kmeans_gpu_internal_time.png'}")


if __name__ == "__main__":
    main()

