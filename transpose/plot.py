#!/usr/bin/env python3
import argparse, os, re, sys
from collections import defaultdict, OrderedDict
import matplotlib.pyplot as plt

# Broad ANSI/control escape matcher (handles CSI, OSC, charset selects like ESC(B), etc.)
ANSI_RE = re.compile(
    r'(?:\x1B[@-Z\\-_]|\x1B\[[0-?]*[ -/]*[@-~]|\x1B\][^\x1B]*\x1B\\|\x1B\([0-~]|\x1B\)[0-~])'
)
NBSP_RE = re.compile(r"[\u00A0\u2000-\u200B]")
TIME_RE = re.compile(r"^\s*(t_[A-Za-z0-9_]+)\s*:\s*([0-9.]+)\s*ms\s*$")

def clean_line(s: str) -> str:
    s = ANSI_RE.sub("", s)
    s = NBSP_RE.sub(" ", s)
    return s.replace("\r", "")

def parse_header(line: str):
    toks = clean_line(line).strip().split()
    kv = {}
    for t in toks:
        if "=" in t:
            k, v = t.split("=", 1)
            kv[k.strip().strip(",").lower()] = v.strip().strip(",")
    mode = kv.get("mode")
    nstr = kv.get("n")
    fstr = kv.get("frac")
    if not (mode and nstr and fstr):
        return None
    try:
        return mode, int(nstr), float(fstr)
    except ValueError:
        return None

def parse_log(text: str):
    lines = text.splitlines()
    recs = []
    i = 0
    while i < len(lines):
        line = clean_line(lines[i])
        if "mode=" not in line:
            i += 1
            continue
        hdr = parse_header(line)
        if not hdr:
            i += 1
            continue
        mode, N, frac = hdr
        rec = {"mode": mode, "N": N, "frac": frac}
        i += 1

        accum = defaultdict(float)
        while i < len(lines):
            l = clean_line(lines[i])
            ls = l.strip()
            if "mode=" in l or ls.startswith("N=") or ls.startswith("N ="):
                break
            m = TIME_RE.match(ls)
            if m:
                k, sval = m.groups()
                v = float(sval)
                if k == "t_end_2_end":
                    rec[k] = v
                else:
                    # still aggregate subtimers in case you want them later,
                    # but they won't be used in plotting/printing now
                    accum[k] += v
            i += 1
        rec.update(accum)
        recs.append(rec)
    return recs

# ---------- NEW: end-to-end–only logic ----------

def summarize_e2e(records):
    """
    Print only end-to-end timings grouped by (mode, N, frac).
    If multiple entries for the same key exist, prints the mean.
    """
    from statistics import mean
    bucket = defaultdict(list)
    for r in records:
        if "t_end_2_end" in r:
            bucket[(r["mode"], r["N"], r["frac"])].append(r["t_end_2_end"])

    # Order: mode, N, then frac ascending
    keys = sorted(bucket.keys(), key=lambda k: (k[0], k[1], k[2]))
    print(f"Parsed {len(records)} blocks.")
    print("End-to-end (ms) by (mode, N, frac):")
    for (mode, N, frac) in keys:
        val = mean(bucket[(mode, N, frac)])
        print(f"  mode={mode:>8}  N={N:<8}  frac={frac:>5.2f}  t_end_2_end={val:.3f} ms")

def plot_e2e_per_mode_N(records, outdir, normalize_to_frac0=True):
    """
    For each (mode, N), make a separate plot of end-to-end time vs fraction.
    If normalize_to_frac0 is True, divide values by the value at frac == 0.0.
    If there's no frac=0.0 for a group, that plot is produced unnormalized with a note.
    """
    os.makedirs(outdir, exist_ok=True)

    # Group records by (mode, N), collect t_end_2_end per frac (averaged if duplicates)
    grouped = defaultdict(lambda: defaultdict(list))
    for r in records:
        if "t_end_2_end" not in r: 
            continue
        key = (r["mode"], r["N"])
        grouped[key][r["frac"]].append(r["t_end_2_end"])

    from statistics import mean

    for (mode, N), frac_dict in grouped.items():
        fracs = sorted(frac_dict.keys())
        e2e_vals = [mean(frac_dict[f]) for f in fracs]

        # Normalization
        normalized = False
        if normalize_to_frac0 and 0.0 in frac_dict:
            base = mean(frac_dict[0.0])
            if base > 0:
                e2e_plot_vals = [v / base for v in e2e_vals]
                normalized = True
            else:
                e2e_plot_vals = e2e_vals  # avoid div by zero
        else:
            e2e_plot_vals = e2e_vals

        # Plot (bar chart across fractions)
        fig_width = max(8, len(fracs) * 0.5)
        plt.figure(figsize=(fig_width, 5.5))
        x = list(range(len(fracs)))
        plt.bar(x, e2e_plot_vals, width=0.7)

        if normalized:
            plt.ylabel("Normalized E2E time (÷ value at frac=0.0)")
            plt.title(f"mode={mode}, N={N} — End-to-end (normalized to frac=0.0)")
        else:
            plt.ylabel("End-to-end time (ms)")
            title = f"mode={mode}, N={N} — End-to-end"
            if normalize_to_frac0:
                title += " (no frac=0.0 found; unnormalized)"
            plt.title(title)

        plt.xlabel("Fraction")
        plt.xticks(x, [f"{f:.2f}" for f in fracs], rotation=45, ha="right")
        plt.grid(axis="y", linestyle="--", alpha=0.4)
        plt.tight_layout()

        fname = f"e2e_mode-{mode}_N-{N}.png"
        outpath = os.path.join(outdir, fname)
        plt.savefig(outpath, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {outpath}")

def write_e2e_csv(records, path, include_normalized=True):
    """
    CSV with columns: mode, N, frac, t_end_2_end[, t_end_2_end_norm]
    Normalization is per (mode, N) baseline at frac=0.0 when present.
    If no baseline exists for a (mode, N), the normalized column is empty for that group.
    """
    from statistics import mean
    # Build (mode,N)-> baseline if frac 0.0 exists
    groups = defaultdict(lambda: defaultdict(list))
    for r in records:
        if "t_end_2_end" in r:
            groups[(r["mode"], r["N"])][r["frac"]].append(r["t_end_2_end"])

    baselines = {}
    for key, frac_map in groups.items():
        if 0.0 in frac_map:
            baselines[key] = mean(frac_map[0.0])

    # Build rows
    rows = []
    for (mode, N), frac_map in groups.items():
        for frac, vals in frac_map.items():
            e2e = mean(vals)
            if include_normalized and (mode, N) in baselines and baselines[(mode, N)] > 0:
                norm = e2e / baselines[(mode, N)]
            else:
                norm = ""
            rows.append((mode, N, frac, e2e, norm))

    # Sort for stable output
    rows.sort(key=lambda t: (t[0], t[1], t[2]))

    # Write
    with open(path, "w", encoding="utf-8") as f:
        header = "mode,N,frac,t_end_2_end_ms"
        if include_normalized:
            header += ",t_end_2_end_norm_frac0"
        f.write(header + "\n")
        for mode, N, frac, e2e, norm in rows:
            line = f"{mode},{N},{frac:.6f},{e2e:.6f}"
            if include_normalized:
                if norm == "":
                    line += ","
                else:
                    line += f",{norm:.6f}"
            f.write(line + "\n")
    print(f"Wrote {path}")

def main():
    ap = argparse.ArgumentParser(description="Plot end-to-end timings per (mode, N), normalized to frac=0.0.")
    ap.add_argument("logfile")
    ap.add_argument("--outdir", default="plots")
    ap.add_argument("--csv", help="Optional CSV export path")
    ap.add_argument("--no-normalize", action="store_true",
                    help="Disable normalization to frac=0.0")
    args = ap.parse_args()

    with open(args.logfile, "rb") as fh:
        text = fh.read().decode("utf-8", errors="ignore")

    records = parse_log(text)
    if not records:
        print("Still no records parsed. Try piping the file through: `sed -r 's/\\x1B\\[[0-9;?]*[ -\\/]*[@-~]//g' file`", file=sys.stderr)
        sys.exit(1)

    # 1) Print only end-to-end
    summarize_e2e(records)

    # 2) Plots per (mode, N), normalized to frac=0.0 by default
    plot_e2e_per_mode_N(records, args.outdir, normalize_to_frac0=(not args.no_normalize))

    # 3) Optional CSV with end-to-end (and normalized column)
    if args.csv:
        write_e2e_csv(records, args.csv, include_normalized=True)

if __name__ == "__main__":
    main()

