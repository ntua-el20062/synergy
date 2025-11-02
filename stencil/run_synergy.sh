#!/usr/bin/env bash
set -euo pipefail
SIZES=(1024 2048 4096 8192)
ITERS=(10)
STEPS=(1 2)
SEED=42
PREFETCH=1

MODES=(
  explicit
  explicit_async
  um_migrate
  gh_hbm_shared
  gh_cpu_shared
  gh_hmm_pageable
  gh_hmm_pageable_gpu_init
  um_migrate_no_prefetch
  gh_hbm_shared_no_prefetch
)

KPCTS=(0 10 25 30 50 70 80)  # % of N for K
THREADS_LIST=(72)  

SYNERGY_BIN="./stencil_bench_synergy"

hr() { printf "%s\n" "----------------------------------------------------------------------------------------------------------------"; }

# ------------------------------------------------------------------------------------
# Parser for program output.
# Supports both:
#   End-to-end (compute loop only): X.XXX ms     <-- new label
#   End-to-end: X.XXX ms                         <-- legacy label
# And also captures:
#   Full end-to-end: X.XXX ms                    <-- new (excludes checksum)
# Returns fields:
#  1: h2d_ms
#  2: um2dev_ms
#  3: d2h_ms
#  4: um2host_ms
#  5: gpu_ms
#  6: gpu_gbps
#  7: cpu_ms
#  8: cpu_gbps
#  9: end2end_ms           (compute loop only)
# 10: read_ms              (checksum read time)
# 11: full_end2end_ms      (mode total, excludes checksum)
# ------------------------------------------------------------------------------------
parse_metrics() {
  local text; text="$(cat)"

  local h2d_ms="" um2dev_ms="" d2h_ms="" um2host_ms=""
  local end2end_ms="" full_end2end_ms="" gpu_ms="" cpu_ms=""
  local gpu_gbps="" cpu_gbps="" read_ms=""

  if [[ "$text" =~ H2D\ memcpy\ \+\ D\ memset:\ ([0-9.+-eE]+)\ ms ]]; then
    h2d_ms="${BASH_REMATCH[1]}"
  fi
  if [[ "$text" =~ UM\ prefetch-to-Device\ \(2x\ buffers\):\ ([0-9.+-eE]+)\ ms ]]; then
    um2dev_ms="${BASH_REMATCH[1]}"
  fi
  if [[ "$text" =~ D2H\ memcpy:\ ([0-9.+-eE]+)\ ms ]]; then
    d2h_ms="${BASH_REMATCH[1]}"
  fi
  if [[ "$text" =~ UM-\>Host\ prefetch\ \(dOut\):\ ([0-9.+-eE]+)\ ms ]]; then
    um2host_ms="${BASH_REMATCH[1]}"
  fi

  # New label first, then legacy as fallback
  if [[ "$text" =~ End-to-end\ \(compute\ loop\ only\):\ ([0-9.+-eE]+)\ ms ]]; then
    end2end_ms="${BASH_REMATCH[1]}"
  elif [[ "$text" =~ End-to-end:\ ([0-9.+-eE]+)\ ms ]]; then
    end2end_ms="${BASH_REMATCH[1]}"
  fi

  # New full end-to-end (excludes checksum)
  if [[ "$text" =~ Full\ end-to-end:\ ([0-9.+-eE]+)\ ms ]]; then
    full_end2end_ms="${BASH_REMATCH[1]}"
  fi

  if [[ "$text" =~ GPU\ compute:\ ([0-9.+-eE]+)\ ms ]]; then
    gpu_ms="${BASH_REMATCH[1]}"
  fi
  if [[ "$text" =~ CPU\ compute:\ ([0-9.+-eE]+)\ ms ]]; then
    cpu_ms="${BASH_REMATCH[1]}"
  fi

  if [[ "$text" =~ GPU\ stencil\ BW:\ ([0-9.+-eE]+)\ GB/s ]]; then
    gpu_gbps="${BASH_REMATCH[1]}"
  fi
  if [[ "$text" =~ CPU\ stencil\ BW:\ ([0-9.+-eE]+)\ GB/s ]]; then
    cpu_gbps="${BASH_REMATCH[1]}"
  fi

  if [[ "$text" =~ CPU\ checksum.*read\ time:\ ([0-9.+-eE]+)\ ms ]]; then
    read_ms="${BASH_REMATCH[1]}"
  fi

  printf "%s|%s|%s|%s|%s|%s|%s|%s|%s|%s" \
    "$h2d_ms" "$um2dev_ms" "$d2h_ms" "$um2host_ms" \
    "$gpu_ms" "$gpu_gbps" "$cpu_ms" "$cpu_gbps" "$end2end_ms" "$full_end2end_ms"
}

print_header_stencil() {
  printf "%s\n" "STENCIL RESULTS"
  hr
  printf "%-22s %-8s %-5s %-7s %-10s %-12s %-10s %-12s %-10s %-10s %-10s %-13s %-13s \n" \
    "Mode" "N" "Step" "Iters" \
    "H2D(ms)" "UM->Dev(ms)" "D2H(ms)" "UM->Host(ms)" \
    "GPU(ms)" "GPU(GB/s)" "CPU(ms)" "CPU(GB/s)" "End2End(ms)" "FullE2E(ms)" 
  hr
}

print_row_stencil() {
  local mode="$1" N="$2" steps="$3" iters="$4"
  local h2d_ms="$5" um2dev_ms="$6" d2h_ms="$7" um2host_ms="$8"
  local gpu_ms="$9" gpu_gbps="${10}" cpu_ms="${11}" cpu_gbps="${12}" end2end_ms="${13}" read_ms="${14}" full_end2end_ms="${16}"

  for v in h2d_ms um2dev_ms d2h_ms um2host_ms gpu_ms gpu_gbps cpu_ms cpu_gbps end2end_ms full_end2end_ms; do
    [[ -z "${!v:-}" ]] && printf -v "$v" -- "--"
  done

  printf "%-22s %-8s %-5s %-7s %-10s %-12s %-10s %-12s %-10s %-10s %-10s %-13s \n" \
    "$mode" "$N" "$steps" "$iters" \
    "$h2d_ms" "$um2dev_ms" "$d2h_ms" "$um2host_ms" \
    "$gpu_ms" "$gpu_gbps" "$cpu_ms" "$cpu_gbps" "$end2end_ms" "$full_end2end_ms" 
}

print_header_synergy_manual() {
  printf "%s\n" "STENCIL SYNERGY (Manual split) RESULTS"
  hr
  printf "%-22s %-8s %-5s %-7s %-6s %-8s %-10s %-12s %-10s %-12s %-10s %-10s %-10s %-13s %-13s %-13s\n" \
    "Mode" "N" "Step" "Iters" "K%" "Threads" \
    "H2D(ms)" "UM->Dev(ms)" "D2H(ms)" "UM->Host(ms)" \
    "GPU(ms)" "GPU(GB/s)" "CPU(ms)" "CPU(GB/s)" "End2End(ms)" "FullE2E(ms)" 
  hr
}

print_row_synergy_manual() {
  local mode="$1" N="$2" steps="$3" iters="$4" kpct="$5" thr="$6"
  local h2d_ms="$7" um2dev_ms="$8" d2h_ms="$9" um2host_ms="${10}"
  local gpu_ms="${11}" gpu_gbps="${12}" cpu_ms="${13}" cpu_gbps="${14}" end2end_ms="${15}" full_end2end_ms="${16}"

  for v in h2d_ms um2dev_ms d2h_ms um2host_ms gpu_ms gpu_gbps cpu_ms cpu_gbps end2end_ms full_end2end_ms; do
    [[ -z "${!v:-}" ]] && printf -v "$v" -- "--"
  done

  printf "%-22s %-8s %-5s %-7s %-6s %-8s %-10s %-12s %-10s %-12s %-10s %-10s %-10s %-12s %-13s %-13s\n" \
    "$mode" "$N" "$steps" "$iters" "$kpct" "$thr" \
    "$h2d_ms" "$um2dev_ms" "$d2h_ms" "$um2host_ms" \
    "$gpu_ms" "$gpu_gbps" "$cpu_ms" "$cpu_gbps" "$end2end_ms" "$full_end2end_ms" 
}



run_one_synergy_manual() {
  local mode="$1" N="$2" iters="$3" steps="$4" kpct="$5" thr="$6"
  [[ ! -x "$SYNERGY_BIN" ]] && return 0
  local out tmp
  out="$("$SYNERGY_BIN" -n "$N" -i "$iters" -t "$steps" -m "$mode" -p "$PREFETCH" -r "$SEED" \
       -- --kPct "$kpct" --threads "$thr" 2>&1 || true)"
  tmp="$(parse_metrics <<<"$out")"

  IFS='|' read -r \
    h2d_ms um2dev_ms d2h_ms um2host_ms \
    gpu_ms gpu_gbps cpu_ms cpu_gbps end2end_ms full_end2end_ms \
    <<<"$tmp"

  print_row_synergy_manual "$mode" "$N" "$steps" "$iters" "$kpct" "$thr" \
    "$h2d_ms" "$um2dev_ms" "$d2h_ms" "$um2host_ms" \
    "$gpu_ms" "$gpu_gbps" "$cpu_ms" "$cpu_gbps" "$end2end_ms" "$full_end2end_ms"
}


main() {
    any=1
    # manual
    print_header_synergy_manual
    for N in "${SIZES[@]}"; do
      for it in "${ITERS[@]}"; do
        for step in "${STEPS[@]}"; do
          for mode in "${MODES[@]}"; do
            for kpct in "${KPCTS[@]}"; do
              for thr in "${THREADS_LIST[@]}"; do
                if [[ "$mode" == "explicit_async" ]]; then
                  continue
                fi
                run_one_synergy_manual "$mode" "$N" "$it" "$step" "$kpct" "$thr"
              done
            done
          done
          hr
        done
      done
    done
    echo

}

main "$@"

