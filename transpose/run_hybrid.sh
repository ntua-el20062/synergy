#!/usr/bin/env bash
# run.sh — sweep transpose configurations (classic + hybrid)
# Usage:
#   chmod +x run.sh
#   ./run.sh
# Env overrides:
#   BIN_TRANSPOSE=./transpose_bench BIN_HYBRID=./transpose_hybrid N="4096 8192" ITERS="10" PREFETCHES="1" FRACS="0.5 0.8" ./run.sh

set -euo pipefail

# -------- Binaries --------
BIN_TRANSPOSE=${BIN_TRANSPOSE:-./transpose_bench}
BIN_HYBRID=${BIN_HYBRID:-./transpose_bench_hybrid}

# -------- Sweeps (override with env) --------
SIZES=(${N:-2048 4096 8192})
ITERS=(${ITERS:-5 10})
PREFETCHES=(${PREFETCHES:-1})     # used by um_migrate/gh_hbm_shared
SEED=${SEED:-12345}

# Hybrid-only sweep (GPU fraction)
FRACS=(${FRACS:-0.1 0.2 0.3 0.4 0.5 0.7 0.8})

# Full mode set (include your custom ones; errors are tolerated per-mode)
MODES=(
  explicit
  explicit_async            
  um_migrate
  um_migrate_no_prefetch
  gh_hbm_shared
  gh_hbm_shared_no_prefetch
  gh_cpu_shared
  gh_hmm_pageable
  gh_hmm_pageable_cuda_init 
)

# -------- Pretty printing --------
BOLD=$(tput bold 2>/dev/null || true)
DIM=$(tput dim 2>/dev/null || true)
RED=$(tput setaf 1 2>/dev/null || true)
GRN=$(tput setaf 2 2>/dev/null || true)
YEL=$(tput setaf 3 2>/dev/null || true)
BLU=$(tput setaf 4 2>/dev/null || true)
CYN=$(tput setaf 6 2>/dev/null || true)
RST=$(tput sgr0 2>/dev/null || true)

hr(){ printf "%s\n" "--------------------------------------------------------------------------------"; }
hdr(){ printf "%s\n" "${BOLD}${BLU}$*${RST}"; }
sub(){ printf "%s\n" "${CYN}$*${RST}"; }

# -------- Helpers --------
has_bin() { [[ -x "$1" ]]; }

run_transpose_case() {
  local bin="$1" mode="$2" N="$3" iters="$4" prefetch="$5" seed="$6"
  echo "${DIM}mode=${mode} N=${N} iters=${iters} prefetch=${prefetch}${RST}"
  set +e
  "$bin" -n "$N" -i "$iters" -m "$mode" -p "$prefetch" -r "$seed"
  local rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "${YEL}[skip]${RST} ${mode} failed or unsupported (rc=${rc})"
  fi
  echo
}

run_hybrid_case() {
  local bin="$1" mode="$2" N="$3" iters="$4" prefetch="$5" frac="$6" seed="$7"
  echo "${DIM}mode=${mode} N=${N} iters=${iters} prefetch=${prefetch} frac=${frac}${RST}"
  set +e
  "$bin" -n "$N" -i "$iters" -m "$mode" -p "$prefetch" -f "$frac" -r "$seed"
  local rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "${YEL}[skip]${RST} ${mode} failed or unsupported (rc=${rc})"
  fi
  echo
}

# ================= MAIN =================
main() {
  if has_bin "$BIN_TRANSPOSE"; then
    hdr "TRANSPOSE SWEEP (${BIN_TRANSPOSE})"
    for N in "${SIZES[@]}"; do
      for it in "${ITERS[@]}"; do
        sub "N=${N}  iters=${it}"
        hr
        for mode in "${MODES[@]}"; do
          # prefetch only meaningful for UM/GH_HBM_SHARED (but safe to pass always)
          for p in "${PREFETCHES[@]}"; do
            run_transpose_case "$BIN_TRANSPOSE" "$mode" "$N" "$it" "$p" "$SEED"
          done
        done
        hr; echo
      done
    done
  else
    echo "${YEL}[warn]${RST} ${BIN_TRANSPOSE} not found or not executable — skipping classic transpose."
  fi

  if has_bin "$BIN_HYBRID"; then
    hdr "HYBRID SWEEP (${BIN_HYBRID})"
    for N in "${SIZES[@]}"; do
      for it in "${ITERS[@]}"; do
        for frac in "${FRACS[@]}"; do
          sub "N=${N}  iters=${it}  frac=${frac}"
          hr
          for mode in "${MODES[@]}"; do
            for p in "${PREFETCHES[@]}"; do
     		if [[ "$mode" == "explicit" || "$mode" == "explicit_async" ]]; then
    			continue
		fi

              run_hybrid_case "$BIN_HYBRID" "$mode" "$N" "$it" "$p" "$frac" "$SEED"

      done
          done
         hr; echo
       done
     done
    done
  else
    echo "${YEL}[warn]${RST} ${BIN_HYBRID} not found or not executable — skipping hybrid transpose."
  fi

  echo "${GRN}${BOLD}Done.${RST}"
}

main "$@"

