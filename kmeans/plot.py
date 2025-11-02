#!/usr/bin/env python3

import argparse
import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ---------- Parsing ----------

def parse_benchmark_text(text: str) -> pd.DataFrame:
    """
    Parse the benchmark text into a tidy DataFrame with columns:
    dataset_size_MB, numObjs, numCoords, numClusters, block_size, impl, memory,
    t_alloc_ms, t_loop_total_ms, t_loop_avg_ms, t_loop_min_ms, t_loop_max_ms, t_gpu_ms
    Handles these sections:
      - |-------------Sequential Kmeans-------------|
      - |-----------Full-offload GPU KMeans (managed memory)------------|
      - |-----------Full-offload GPU KMeans (system allocated memory)------------|
      - |-----------Full-offload GPU Kmeans------------|   (explicit; uses t_gpu_avg)
    """
    sections = [s for s in text.split("dataset_size") if s.strip()]
    rows = []

    def get_float(label, s, alt=None):
        for lab in [label] + ([alt] if alt else []):
            m = re.search(rf"{re.escape(lab)}\s*=\s*([0-9.]+)\s*ms", s)
            if m:
                return float(m.group(1))
        return None

    def get_int(label, s):
        m = re.search(rf"{re.escape(label)}\s*=\s*(\d+)", s)
        return int(m.group(1)) if m else None

    for sec in sections:
        ds_mb = re.search(r"=\s*([0-9.]+)\s*MB", sec)
        bs = re.search(r"block_size\s*=\s*(\d+)", sec)
        numObjs = get_int("numObjs", sec)
        numCoords = get_int("numCoords", sec)
        numClusters = get_int("numClusters", sec)

        dataset_size_mb = float(ds_mb.group(1)) if ds_mb else None
        block_size = int(bs.group(1)) if bs else None

        # CPU Sequential
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
                t_gpu_ms=get_float("t_gpu", body)
            ))

        # GPU (system allocated)
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

        # GPU (explicit transfers)
        explicit = re.search(
            r"\|-----------Full-offload GPU Kmeans------------\|(.+?)\|-------------------------------------------\|",
            sec, flags=re.S | re.I
        )
        if explicit:
            body = explicit.group(1)
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
                t_gpu_ms=get_float("t_gpu_avg", body)  # note: label differs here
            ))

    return pd.DataFrame(rows)


# ---------- Aggregation & Tables ----------

def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (impl, block_size): mean of key metrics (handles duplicates)."""
    agg = (df.groupby(["impl", "block_size"], as_index=False)
             .agg(t_loop_avg_ms=("t_loop_avg_ms", "mean"),
                  t_gpu_ms=("t_gpu_ms", "mean")))
    return agg.sort_values(["impl", "block_size"])


def compute_tables(df_agg: pd.DataFrame):
    """
    Build:
      - ms_table: per block_size, per implementation, avg per-iteration time (ms), including CPU.
      - speedup: speedup (×) vs CPU for non-CPU impls (empty if CPU missing).
      - gpu_kernel: GPU kernel time (ms) pivot, if available (else empty).
    """
    # ms_table (include CPU)
    ms_pivot = df_agg.pivot(index="block_size", columns="impl", values="t_loop_avg_ms").sort_index()
    ms_table = ms_pivot.rename(columns=lambda c: f"{c} ms").reset_index()

    # speedup relative to CPU (if CPU present)
    '''
    if "CPU Sequential" in ms_pivot.columns:
        cpu = ms_pivot["CPU Sequential"]
        speedup = (cpu.to_frame(name="CPU ms")
                   .join((cpu / ms_pivot.drop(columns=["CPU Sequential"]))
                         .rename(columns=lambda c: f"{c} speedup×"))
                   .reset_index())
    else:
    '''
    speedup = pd.DataFrame()

    # GPU kernel pivot (if any t_gpu_ms present)
    gpu_kernel_src = df_agg.dropna(subset=["t_gpu_ms"])
    if not gpu_kernel_src.empty:
        gpu_kernel = (gpu_kernel_src
                      .pivot(index="block_size", columns="impl", values="t_gpu_ms")
                      .sort_index()
                      .rename(columns=lambda c: f"{c} t_gpu ms")
                      .reset_index())
    else:
        gpu_kernel = pd.DataFrame()

    return ms_table, speedup, gpu_kernel


def best_block_per_impl(df_agg: pd.DataFrame) -> pd.DataFrame:
    """Return one row per impl with the block_size that minimizes avg time."""
    idx = df_agg.groupby("impl")["t_loop_avg_ms"].idxmin()
    return (df_agg.loc[idx]
            .sort_values("t_loop_avg_ms")
            .reset_index(drop=True))


# ---------- Plots ----------

def make_plots(df_agg: pd.DataFrame, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Runtime vs block size
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
    '''
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
    '''
    # 3) GPU kernel time
    gpu_only = df_agg.dropna(subset=["t_gpu_ms"])
    if not gpu_only.empty:
        plt.figure()
        for impl, grp in gpu_only.sort_values("block_size").groupby("impl"):
            plt.plot(grp["block_size"], grp["t_gpu_ms"], marker="o", label=f"{impl} (t_gpu)")
        plt.xlabel("Block size")
        plt.ylabel("Reported GPU kernel time (ms)")
        plt.title("GPU kernel time vs block size")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(outdir / "kmeans_gpu_internal_time.png", bbox_inches="tight")
        plt.close()

    # 4) Best block per impl (bar chart)
    best = best_block_per_impl(df_agg)
    plt.figure()
    plt.bar(best["impl"], best["t_loop_avg_ms"])
    for i, (impl, bs, t) in enumerate(zip(best["impl"], best["block_size"], best["t_loop_avg_ms"])):
        plt.text(i, t, f"bs={bs}\n{t:.3f} ms", ha="center", va="bottom")
    plt.ylabel("Best avg per-iteration time (ms)")
    plt.title("Best block size per implementation")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(outdir / "kmeans_best_block_bars.png", bbox_inches="tight")
    plt.close()


# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description="Parse and plot KMeans benchmark results.")
    ap.add_argument("input", type=Path, help="Path to results text file")
    ap.add_argument("--outdir", type=Path, default=None, help="Directory for CSVs/plots (default: alongside input)")
    args = ap.parse_args()

    outdir = args.outdir or args.input.parent

    text = args.input.read_text(encoding="utf-8", errors="ignore")
    df_raw = parse_benchmark_text(text)
    if df_raw.empty:
        raise SystemExit("No results parsed — check the input file format or path.")

    # Save raw tidy rows
    raw_csv = outdir / "kmeans_timings_raw.csv"
    df_raw.to_csv(raw_csv, index=False)

    # Aggregate & tables
    df_agg = aggregate(df_raw)
    agg_csv = outdir / "kmeans_timings_agg.csv"
    df_agg.to_csv(agg_csv, index=False)

    ms_table, speedup, gpu_kernel = compute_tables(df_agg)
    ms_table_path = outdir / "kmeans_ms_by_impl.csv"
    ms_table.to_csv(ms_table_path, index=False)
    if not speedup.empty:
        (outdir / "kmeans_speedup_vs_cpu.csv").write_text(
            speedup.to_csv(index=False), encoding="utf-8"
        )
    if not gpu_kernel.empty:
        (outdir / "kmeans_gpu_kernel_ms.csv").write_text(
            gpu_kernel.to_csv(index=False), encoding="utf-8"
        )

    # Best block per impl
    best_csv = outdir / "kmeans_block_best.csv"
    best_block_per_impl(df_agg).to_csv(best_csv, index=False)

    # Plots
    make_plots(df_agg, outdir)

    # Summary
    print("Wrote:")
    for p in [
        raw_csv,
        agg_csv,
        ms_table_path,
        outdir / "kmeans_speedup_vs_cpu.csv",
        outdir / "kmeans_gpu_kernel_ms.csv",
        best_csv,
        outdir / "kmeans_runtime_vs_blocksize.png",
        outdir / "kmeans_speedup_vs_cpu.png",
        outdir / "kmeans_gpu_internal_time.png",
        outdir / "kmeans_best_block_bars.png",
    ]:
        if p.exists():
            print(f"  - {p}")


if __name__ == "__main__":
    main()

