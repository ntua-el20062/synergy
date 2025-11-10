#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
from typing import List, Dict, Tuple

import pandas as pd
import matplotlib.pyplot as plt


# --- Parsing helpers ---------------------------------------------------------

SECTION_HEAD_RE = re.compile(r"block_size\s*=\s*(\d+)", re.IGNORECASE)
END2END_RE = re.compile(r"end2end\s*=\s*([\d.]+)\s*ms", re.IGNORECASE)

# Matches lines like:
#  "t_alloc: 492.417812 ms"
#  "-> t_alloc = 0.010967 ms"
#  "t_alloc_gpu: 205.168009 ms"
#  "t_init = 233.5 ms"
COMPONENT_RE = re.compile(
    r"(?:->\s*)?([A-Za-z0-9_]+)\s*[:=]\s*([\d.]+)\s*ms",
    re.IGNORECASE
)


def split_sections(text: str) -> List[Tuple[int, str]]:
    """Return list of (block_size, section_text)."""
    sections = []
    matches = list(SECTION_HEAD_RE.finditer(text))
    for i, m in enumerate(matches):
        bs = int(m.group(1))
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append((bs, text[start:end]))
    return sections


def parse_section(section_text: str) -> Dict[str, float]:
    """Parse one section, returning dict of metrics including end2end."""
    data: Dict[str, float] = {}

    # end2end
    m = END2END_RE.search(section_text)
    if not m:
        raise ValueError("Could not find 'end2end' in a section.")
    data["end2end"] = float(m.group(1))

    # components
    for name, val in COMPONENT_RE.findall(section_text):
        key = name.strip()
        # focus on t_* keys (includes t_alloc, t_alloc_gpu, t_init, t_cpu, t_gpu, t_transfers)
        if not key.lower().startswith("t_"):
            continue
        # don't overwrite end2end if it's matched by the general pattern (it won't, but just in case)
        if key.lower() == "t_end2end" or key.lower() == "end2end":
            continue
        data[key] = float(val)

    return data


def parse_file(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    rows = []
    for block_size, sec in split_sections(text):
        row = {"block_size": block_size}
        vals = parse_section(sec)
        row.update(vals)
        rows.append(row)

    if not rows:
        raise ValueError("No data parsed. Check the input format.")

    df = pd.DataFrame(rows).sort_values("block_size").reset_index(drop=True)

    # Identify component columns (all t_* except any accidental t_end2end)
    comp_cols = [c for c in df.columns if c.startswith("t_") and c.lower() != "t_end2end"]
    # If a component is missing in a row, fill with 0 for stacking consistency
    df[comp_cols] = df[comp_cols].fillna(0.0)

    # Compute totals and error vs end2end
    df["components_sum_ms"] = df[comp_cols].sum(axis=1)
    df["end2end_ms"] = df["end2end"]
    df["components_error_ms"] = df["end2end_ms"] - df["components_sum_ms"]

    # Reorder columns nicely
    ordered_cols = ["block_size"] + sorted(comp_cols) + ["components_sum_ms", "end2end_ms", "components_error_ms"]
    return df[ordered_cols]


# --- Plotting ----------------------------------------------------------------

def plot_stacked_vs_end2end(df: pd.DataFrame, out_path: str, prefix: str):
    labels = df["block_size"].astype(str).tolist()
    x = range(len(labels))
    width = 0.35
    offset = width / 2

    # prefer a consistent stacking order if present
    preferred = ["t_alloc", "t_alloc_gpu", "t_init", "t_cpu", "t_gpu", "t_transfers"]
    comp_cols = [c for c in df.columns if c.startswith("t_") and c.lower() != "t_end2end"]
    # sort using preferred order first, then any extras alphabetically
    extras = [c for c in comp_cols if c not in preferred]
    ordered = [c for c in preferred if c in comp_cols] + sorted(extras)

    fig, ax = plt.subplots(figsize=(12, 6))

    bottoms = [0.0] * len(labels)
    # draw left stacked bar (all components)
    for col in ordered:
        ax.bar([i - offset for i in x], df[col], width, bottom=bottoms, label=col)
        bottoms = [b + v for b, v in zip(bottoms, df[col])]

    # draw right bar (end2end)
    ax.bar([i + offset for i in x], df["end2end_ms"], width, label="end2end")

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_xlabel("block_size")
    ax.set_ylabel("Time (ms) over 10 loops")
    ax.set_title(f"GPU KMeans ({prefix}): Components (stacked) vs End-to-End")
    ax.legend(loc="upper left", ncol=2)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# --- CLI ---------------------------------------------------------------------

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
    img_path = os.path.join(args.outdir, f"{args.prefix}_stacked_vs_end2end.png")
    plot_stacked_vs_end2end(df, img_path, args.prefix)

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {img_path}")
    print("\nPreview:")
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()

