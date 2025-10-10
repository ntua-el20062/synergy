#!/usr/bin/env bash
set -euo pipefail

BIN=${BIN:-./transpose_bench}
N=${N:-4096}
ITER=${ITER:-10}
PREFETCH=${PREFETCH:-1}   # only used by um_migrate / gh_hbm_shared
SEED=${SEED:-12345}

modes=(explicit um_migrate gh_hbm_shared gh_cpu_shared gh_hmm_pageable)

for m in "${modes[@]}"; do
  echo "mode: $m"
  # keep going even if a mode errors out (e.g., gh_cpu_shared on some setups)
  "$BIN" -n "$N" -i "$ITER" -m "$m" -p "$PREFETCH" -r "$SEED" || true
  echo "---"
done

