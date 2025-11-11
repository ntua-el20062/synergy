#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
from typing import List, Dict, Tuple

import pandas as pd
import matplotlib.pyplot as plt

# Header line looks like:
# dataset_size = 1024.00 MB    numObjs = 4194304    numCoords = 32    numClusters = 64, block_size = 128
CONFIG_HEADER_RE = re.compile(
    r"dataset_size\s*=\s*([\d.]+)\s*MB\s*"
    r".*?numObjs\s*=\s*(\d+)\s*"
    r".*?numCoords\s*=\s*(\d+)\s*"
    r".*?numClusters\s*=\s*(\d+)\s*,\s*block_size\s*=\s*(\d+)",
    re.IGNORECASE | re.DOTALL,
)

END2END_RE = re.compile(r"end2end\s*=\s*([\d.]+)\s*ms", re.IGNORECASE)

COMPONENT_RE = re.compile(
    r"(?:->\s*)?([A-Za-z0-9_]+)\s*[:=]\s*([\d.]+)\s*ms",
    re.IGNORECASE
)

def split_sections(text: str) -> List[Tuple[re.Match, str]]:
    """Return list of (header_match, section_text_including_header)."""
    headers = list(CONFIG_HEADER_RE.finditer(text))
    sections: List[Tuple[re.Match, str]] = []
    for i, m in enumerate(headers):
        start = m.start()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        sections.append((m, text[start:end]))
    return sections

def parse_section(section_text: str) -> Dict[str, float]:
    """Parse timings within a section."""
    data: Dict[str, float] = {}

    m = END2END_RE.search(section_text)
    if not m:
        raise ValueError("Could not find 'end2end' in a section.")
    data["end2end"] = float(m.group(1))

    for name, val in COMPONENT_RE.findall(section_text):
        key = name.strip()
        if not key.lower().startswith("t_"):
            continue
        if key.lower() in ("t_end2end", "end2end"):
            continue
        data[key] = float(val)

    return data

def parse_file(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    rows = []
    for hdr, sec in split_sections(text):
        size_mb = float(hdr.group(1))
        num_objs = int(hdr.group(2))
        num_coords = int(hdr.group(3))
        num_clusters = int(hdr.group(4))
        block_size = int(hdr.group(5))

        label = f"{size_mb:.0f}MB_{num_coords}coords_{num_clusters}centers"

        row = {
            "config": label,
            "block_size": block_size,          # kept if you still want to see it in the CSV
            "dataset_size_MB": size_mb,
            "numObjs": num_objs,
            "numCoords": num_coords,
            "numClusters": num_clusters,
        }
        row.update(parse_section(sec))
        rows.append(row)

    if not rows:
        raise ValueError("No data parsed. Check the input format.")

    df = pd.DataFrame(rows).reset_index(drop=True)

    comp_cols = [c for c in df.columns if c.startswith("t_") and c.lower() != "t_end2end"]
    if comp_cols:
        df[comp_cols] = df[comp_cols].fillna(0.0)

    df["components_sum_ms"] = df[comp_cols].sum(axis=1) if comp_cols else 0.0
    df["end2end_ms"] = df["end2end"]
    df["components_error_ms"] = df["end2end_ms"] - df["components_sum_ms"]

    ordered_cols = (
        ["config", "block_size", "dataset_size_MB", "numObjs", "numCoords", "numClusters"]
        + sorted(comp_cols)
        + ["components_sum_ms", "end2end_ms", "components_error_ms"]
    )
    return df[ordered_cols]

def plot_stacked_vs_end2end(df: pd.DataFrame, out_path: str, prefix: str):
    labels = df["config"].tolist()
    x = range(len(labels))
    width = 0.35
    offset = width / 2

    preferred = [
        "t_alloc_dealloc_cpu",
        "t_alloc_dealloc_gpu",
        "t_gpu_computation",
        "t_um_advise",
        "t_um_prefetch",
        "t_alloc_dealloc_um",
        "t_alloc_dealloc_malloc",
        "t_transfers",
        "t_other",
    ]

    comp_cols = [c for c in df.columns if c.startswith("t_") and c.lower() != "t_end2end"]
    extras = [c for c in comp_cols if c not in preferred]
    ordered = [c for c in preferred if c in comp_cols] + sorted(extras)

    fig, ax = plt.subplots(figsize=(12, 6))

    bottoms = [0.0] * len(labels)
    for col in ordered:
        ax.bar([i - offset for i in x], df[col], width, bottom=bottoms, label=col)
        bottoms = [b + v for b, v in zip(bottoms, df[col])]

    ax.bar([i + offset for i in x], df["end2end_ms"], width, label="end2end")

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_xlabel("Configuration (dataset_size_MB_numCoords_numClusters)")
    ax.set_ylabel("Total Time (ms)")
    ax.set_title(f"GPU KMeans ({prefix}): Components vs End2End")
    ax.legend(loc="upper left", ncol=2)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser(description="Plot stacked component timings vs end-to-end from KMeans logs.")
    ap.add_argument("input", help="Path to the timing log file")
    ap.add_argument("--outdir", default=".", help="Output directory (default: current directory)")
    ap.add_argument("--prefix", default="kmeans_timings", help="Output filename prefix (default: kmeans_timings)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = parse_file(args.input)

    # Save CSV
    csv_path = os.path.join(args.outdir, f"{args.prefix}_parsed.csv")
    df.to_csv(csv_path, index=False)

    # Plot (stacked components vs end2end)
    img_path = os.path.join(args.outdir, f"{args.prefix}.png")
    plot_stacked_vs_end2end(df, img_path, args.prefix)

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {img_path}")
    print("\nPreview:")
    print(df.head().to_string(index=False))

if __name__ == "__main__":
    main()

