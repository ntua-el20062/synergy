#!/usr/bin/env bash
set -euo pipefail

BIN=${BIN:-./transpose_bench}
N=${N:-4096}
ITER=${ITER:-10}
PREFETCH=${PREFETCH:-1}   #only used by um_migrate / gh_hbm_shared
SEED=${SEED:-12345}

modes=(explicit um_migrate um_migrate_no_prefetch gh_hbm_shared gh_hbm_shared_no_prefetch gh_cpu_shared gh_hmm_pageable gh_hmm_pageable_cuda_init)

for m in "${modes[@]}"; do
  echo "mode: $m"
  "$BIN" -n "$N" -i "$ITER" -m "$m" -p "$PREFETCH" -r "$SEED" || true
  echo "---"
done


