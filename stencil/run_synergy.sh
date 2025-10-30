#!/usr/bin/env bash
set -euo pipefail

SIZES=(2048 4096 8192)
ITERS=(5)
STEPS=(1 2)
SEED=12345
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

#“synergy” manual sweep knobs
KPCTS=(10 15 20 25 30)      # % of N for K
THREADS_LIST=(1 2 4 8 16 32 64)   # CPU threads 

STENCIL_BIN="./stencil_bench"
SYNERGY_BIN="./stencil_bench_synergy"

hr() { printf "%s\n" "----------------------------------------------------------------------------------------------------------------"; }


# Common metrics parser for both binaries
parse_metrics() {
  local text; text="$(cat)"
  # NOTE: kernel_ms_total is kept here but intentionally left unpopulated 
  # to force print_row to calculate Total = Cold + Warm.
  local kernel_ms_total="" eff_gbps="" read_ms="" d2h_ms="" um_back_ms="" cold_ms="" warm_ms="" 

  # Capture the Step 1 (cold) time (B.BBB) - Uses robust tag
  if [[ "$text" =~ \[Step\ 1\ cold\]:\ ([0-9.+-eE]+)\ ms ]]; then
    cold_ms="${BASH_REMATCH[1]}"
  fi
  # Capture the Steps 2..X (warm) time (C.CCC) - Uses robust tag
  if [[ "$text" =~ \[Steps\ 2..[0-9]+\ warm\ avg\]:\ ([0-9.+-eE]+)\ ms ]]; then
    warm_ms="${BASH_REMATCH[1]}"
  fi

  # NOTE: The ambiguous 'Kernel avg' line is NOT parsed here to avoid reading Read Time.

  if [[ "$text" =~ effective\ bandwidth:\ ([0-9.+-eE]+)\ GB/s ]]; then
    eff_gbps="${BASH_REMATCH[1]}"
  fi
  if [[ "$text" =~ CPU\ checksum.*read\ time:\ ([0-9.+-eE]+)\ ms ]]; then
    read_ms="${BASH_REMATCH[1]}"
  elif [[ "$text" =~ CPU\ checksum.*read:\ ([0-9.+-eE]+)\ ms ]]; then
    read_ms="${BASH_REMATCH[1]}"
  fi
  if [[ "$text" =~ D2H\ memcpy:\ ([0-9.+-eE]+)\ ms ]]; then
    d2h_ms="${BASH_REMATCH[1]}"
  fi
  if [[ "$text" =~ UM\ prefetch-to-CPU:\ ([0-9.+-eE]+)\ ms ]]; then
    um_back_ms="${BASH_REMATCH[1]}"
  fi

  # Pass ALL metrics. kernel_ms_total is empty but occupies the first time position.
  printf "%s|%s|%s|%s|%s|%s|%s" "${kernel_ms_total}" "${eff_gbps}" "${read_ms}" "${d2h_ms}" "${um_back_ms}" "${cold_ms}" "${warm_ms}"
}

# Parse the autotune choice line from synergy binary:
parse_autotune_choice() {
  local text; text="$(cat)"
  local K="" thr=""
  if [[ "$text" =~ Autotune\ picked:\ K=([0-9]+)[^,]*,\ threads=([0-9]+) ]]; then
    K="${BASH_REMATCH[1]}"
    thr="${BASH_REMATCH[2]}"
  fi
  printf "%s|%s" "$K" "$thr"
}

#printing
print_header_stencil() {
  printf "%s\n" "STENCIL RESULTS"
  hr
  # Added Cold(ms) and Warm(ms)
  printf "%-22s %-8s %-5s %-7s %-12s %-12s %-12s %-12s %-10s\n" "Mode" "N" "Step" "Iters" "Total(ms)" "Cold(ms)" "Warm(ms)" "Read(ms)" "UMback(ms)"
  hr
}


print_row_stencil() {
  # kernel_ms_ignored is the empty string from parse_metrics, cold_ms/warm_ms are the actual values
  local mode="$1" N="$2" steps="$3" iters="$4" kernel_ms_ignored="$5" read_ms="$6" um_back_ms="$7" cold_ms_parsed="$8" warm_ms_parsed="$9"

  local cold_ms="$cold_ms_parsed"
  local warm_ms="$warm_ms_parsed"
  local total_kernel_ms="--"

  [[ -z "$um_back_ms" ]] && um_back_ms="--"

  if [[ "$steps" -gt "1" ]]; then
      # Calculate Total Time = Cold + (Warm * (Steps - 1)) using bc
      local total=$(echo "scale=3; $cold_ms + $warm_ms * ($steps - 1)" | bc)
      # Ensure leading zero for values < 1
      total_kernel_ms=$(printf "%.3f" "$total")

  elif [[ "$steps" == "1" ]]; then
      # If steps=1, Total is just the Cold time.
      total_kernel_ms=$(printf "%.3f" "$cold_ms")
      warm_ms="--"
  fi

  # Final sanitization for printing
  [[ -z "$cold_ms" ]] && cold_ms="--"
  [[ -z "$warm_ms" ]] && warm_ms="--"

  # Print the CALCULATED Total Kernel Time
  printf "%-22s %-8s %-5s %-7s %-12s %-12s %-12s %-12s %-10s\n" \
    "$mode" "$N" "$steps" "$iters" "$total_kernel_ms" "$cold_ms" "$warm_ms" "$read_ms" "$um_back_ms"
}



print_header_synergy_manual() {
  printf "%s\n" "STENCIL SYNERGY (Manual split) RESULTS"
  hr
  # Added Cold(ms) and Warm(ms)
  printf "%-22s %-8s %-5s %-7s %-6s %-8s %-12s %-12s %-12s %-12s %-10s\n" \
         "Mode" "N" "Step" "Iters" "K%%" "Threads" "Total(ms)" "Cold(ms)" "Warm(ms)" "Read(ms)" "UMback(ms)"
  hr
}

print_row_synergy_manual() {
  local mode="$1" N="$2" steps="$3" iters="$4" kpct="$5" thr="$6" kernel_ms_ignored="$7" read_ms="$8" um_back_ms="$9" cold_ms_parsed="${10}" warm_ms_parsed="${11}"
  
  local cold_ms="$cold_ms_parsed"
  local warm_ms="$warm_ms_parsed"
  local total_kernel_ms="--"

  [[ -z "$um_back_ms" ]] && um_back_ms="--"
  
  if [[ "$steps" -gt "1" ]]; then
      total_kernel_ms=$(echo "scale=3; $cold_ms + $warm_ms * ($steps - 1)" | bc)
  elif [[ "$steps" == "1" ]]; then
      total_kernel_ms="$cold_ms"
      warm_ms="--"
  fi

  [[ -z "$cold_ms" ]] && cold_ms="--"
  [[ -z "$warm_ms" ]] && warm_ms="--"
  
  printf "%-22s %-8s %-5s %-7s %-6s %-8s %-12s %-12s %-12s %-12s %-10s\n" \
         "$mode" "$N" "$steps" "$iters" "$kpct" "$thr" "$total_kernel_ms" "$cold_ms" "$warm_ms" "$read_ms" "$um_back_ms"
}

print_header_synergy_auto() {
  printf "%s\n" "STENCIL SYNERGY (Autotune) RESULTS"
  hr
  printf "%-22s %-8s %-5s %-7s %-6s %-8s %-12s %-12s %-12s %-12s %-10s\n" \
         "Mode" "N" "Step" "Iters" "Ksel" "ThrSel" "Total(ms)" "Cold(ms)" "Warm(ms)" "Read(ms)" "UMback(ms)"
  hr
}

print_row_synergy_auto() {
  local mode="$1" N="$2" steps="$3" iters="$4" Ksel="$5" thrSel="$6" kernel_ms_ignored="$7" read_ms="$8" um_back_ms="$9" cold_ms_parsed="${10}" warm_ms_parsed="${11}"
  
  local cold_ms="$cold_ms_parsed"
  local warm_ms="$warm_ms_parsed"
  local total_kernel_ms="--"

  [[ -z "$um_back_ms" ]] && um_back_ms="--"
  [[ -z "$Ksel" ]] && Ksel="auto"
  [[ -z "$thrSel" ]] && thrSel="auto"
  
  if [[ "$steps" -gt "1" ]]; then
      total_kernel_ms=$(echo "scale=3; $cold_ms + $warm_ms * ($steps - 1)" | bc)
  elif [[ "$steps" == "1" ]]; then
      total_kernel_ms="$cold_ms"
      warm_ms="--"
  fi
  
  [[ -z "$cold_ms" ]] && cold_ms="--"
  [[ -z "$warm_ms" ]] && warm_ms="--"
  
  printf "%-22s %-8s %-5s %-7s %-6s %-8s %-12s %-12s %-12s %-12s %-10s\n" \
         "$mode" "$N" "$steps" "$iters" "$Ksel" "$thrSel" "$total_kernel_ms" "$cold_ms" "$warm_ms" "$read_ms" "$um_back_ms"
}


run_one_stencil() {
  local mode="$1" N="$2" iters="$3" steps="$4"
  [[ ! -x "$STENCIL_BIN" ]] && return 0
  local out tmp
  out="$("$STENCIL_BIN" -n "$N" -i "$iters" -t "$steps" -m "$mode" -p "$PREFETCH" -r "$SEED" 2>&1 || true)"
  tmp="$(parse_metrics <<<"$out")"
  # Read: kernel_ms_total (empty), eff_gbps, read_ms, d2h_ms, um_back_ms, cold_ms, warm_ms
  IFS='|' read -r kernel_ms_ignored eff_gbps read_ms d2h_ms um_back_ms cold_ms warm_ms <<<"$tmp"
  print_row_stencil "$mode" "$N" "$steps" "$iters" "$kernel_ms_ignored" "$read_ms" "$um_back_ms" "$cold_ms" "$warm_ms"
}

run_one_synergy_manual() {
  local mode="$1" N="$2" iters="$3" steps="$4" kpct="$5" thr="$6"
  [[ ! -x "$SYNERGY_BIN" ]] && return 0
  local out tmp
  out="$("$SYNERGY_BIN" -n "$N" -i "$iters" -t "$steps" -m "$mode" -p "$PREFETCH" -r "$SEED" \
       --  --kPct "$kpct" --threads "$thr" 2>&1 || true)"
  tmp="$(parse_metrics <<<"$out")"
  # Read: kernel_ms_total (empty), eff_gbps, read_ms, d2h_ms, um_back_ms, cold_ms, warm_ms
  IFS='|' read -r kernel_ms_ignored eff_gbps read_ms d2h_ms um_back_ms cold_ms warm_ms <<<"$tmp"
  print_row_synergy_manual "$mode" "$N" "$steps" "$iters" "$kpct" "$thr" "$kernel_ms_ignored" "$read_ms" "$um_back_ms" "$cold_ms" "$warm_ms"
}

run_one_synergy_auto() {
  local mode="$1" N="$2" iters="$3" steps="$4"
  [[ ! -x "$SYNERGY_BIN" ]] && return 0
  local out tmp choice
  out="$("$SYNERGY_BIN" -n "$N" -i "$iters" -t "$steps" -m "$mode" -p "$PREFETCH" -r "$SEED" \
        -- --auto 2>&1 || true)"
  tmp="$(parse_metrics <<<"$out")"
  # Read: kernel_ms_total (empty), eff_gbps, read_ms, d2h_ms, um_back_ms, cold_ms, warm_ms
  IFS='|' read -r kernel_ms_ignored eff_gbps read_ms d2h_ms um_back_ms cold_ms warm_ms <<<"$tmp"
  choice="$(parse_autotune_choice <<<"$out")"
  IFS='|' read -r Ksel thrSel <<<"$choice"
  print_row_synergy_auto "$mode" "$N" "$steps" "$iters" "$Ksel" "$thrSel" "$kernel_ms_ignored" "$read_ms" "$um_back_ms" "$cold_ms" "$warm_ms"
}


main() {
  local any=0
  if [[ -x "$STENCIL_BIN" ]]; then
    any=1
    print_header_stencil
    for N in "${SIZES[@]}"; do
      for it in "${ITERS[@]}"; do
        for step in "${STEPS[@]}"; do
          for mode in "${MODES[@]}"; do
            run_one_stencil "$mode" "$N" "$it" "$step"
          done
          hr
        done
      done
    done
    echo
  fi

  if [[ -x "$SYNERGY_BIN" ]]; then
    any=1
    #manual 
    print_header_synergy_manual
    for N in "${SIZES[@]}"; do
      for it in "${ITERS[@]}"; do
        for step in "${STEPS[@]}"; do
          for mode in "${MODES[@]}"; do
            for kpct in "${KPCTS[@]}"; do
              for thr in "${THREADS_LIST[@]}"; do
                if [[ "$mode" == "explicit" || "$mode" == "explicit_async" ]]; then
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

    #autotune
    print_header_synergy_auto
    for N in "${SIZES[@]}"; do
      for it in "${ITERS[@]}"; do
        for step in "${STEPS[@]}"; do
          for mode in "${MODES[@]}"; do
            if [[ "$mode" == "explicit" || "$mode" == "explicit_async" ]]; then
              continue
            fi
            run_one_synergy_auto "$mode" "$N" "$it" "$step"
          done
          hr
        done
      done
    done
    echo
  fi
}

main "$@"
