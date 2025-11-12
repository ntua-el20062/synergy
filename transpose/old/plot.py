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
                    accum[k] += v
            i += 1
        rec.update(accum)
        recs.append(rec)
    return recs

def plot_by_mode(records, outdir):
    import matplotlib.pyplot as plt
    import os
    from collections import defaultdict

    os.makedirs(outdir, exist_ok=True)
    by_mode = defaultdict(list)
    for r in records:
        by_mode[r["mode"]].append(r)

    print(f"Parsed {len(records)} blocks across {len(by_mode)} modes: {', '.join(sorted(by_mode))}")

    for mode, rows in by_mode.items():
        rows.sort(key=lambda r: (r["N"], r["frac"]))

        # Get all timing keys for this mode (excluding t_end_2_end)
        keys, seen = [], set()
        for r in rows:
            for k, v in r.items():
                if k.startswith("t_") and k != "t_end_2_end" and v != 0.0 and k not in seen:
                    seen.add(k)
                    keys.append(k)

        x = list(range(len(rows)))
        # Prettier, compact x-axis labels
        xlabels = [f"N={r['N']}, f={r['frac']:.1f}" for r in rows]
        bottom = [0.0] * len(rows)

        # Auto width depending on number of bars
        fig_width = max(10, len(rows) * 0.4)
        plt.figure(figsize=(fig_width, 6))
        bar_width = 0.7

        # Plot stacked bars
        for k in keys:
            heights = [r.get(k, 0.0) for r in rows]
            plt.bar(x, heights, width=bar_width, bottom=bottom, label=k.replace("_", " "))
            bottom = [b + h for b, h in zip(bottom, heights)]

        # Nice axis formatting
        plt.xticks(x, xlabels, rotation=45, ha="right", fontsize=9)
        plt.xlabel("(N, fraction)", fontsize=11)
        plt.ylabel("Time (ms)", fontsize=11)
        plt.title(f"Stacked timings for mode={mode} (excluding t_end_2_end)", fontsize=13, pad=10)
        plt.grid(axis="y", linestyle="--", alpha=0.4)

        # Add a bit of headroom so bars don’t get cut off
        ymax = max(bottom) if bottom else 0
        plt.ylim(0, ymax * 1.1)

        if keys:
            plt.legend(fontsize=8, ncol=2)

        # Avoid cropping tall bars or labels
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        outpath = os.path.join(outdir, f"stacked_{mode}.png")
        plt.savefig(outpath, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {outpath}")


def write_csv(records, path):
    cols = OrderedDict((k, True) for k in ["mode","N","frac","t_end_2_end"])
    for r in records:
        for k in r.keys():
            cols.setdefault(k, True)
    cols = list(cols.keys())
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in records:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
    print(f"Wrote {path}")

def main():
    ap = argparse.ArgumentParser(description="Plot stacked timings per mode from bench logs.")
    ap.add_argument("logfile")
    ap.add_argument("--outdir", default="plots")
    ap.add_argument("--csv", help="Optional CSV export")
    args = ap.parse_args()

    with open(args.logfile, "rb") as fh:
        text = fh.read().decode("utf-8", errors="ignore")

    records = parse_log(text)
    if not records:
        print("Still no records parsed. Try piping the file through: `sed -r 's/\\x1B\\[[0-9;?]*[ -\\/]*[@-~]//g' file`", file=sys.stderr)
        sys.exit(1)
    plot_by_mode(records, args.outdir)
    if args.csv:
        write_csv(records, args.csv)

if __name__ == "__main__":
    main()

