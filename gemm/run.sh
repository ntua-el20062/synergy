#!/usr/bin/env bash
set -euo pipefail

# Paths to executables (adjust if needed)
BASELINE=./cublas_gemm_example
MANAGED=./cublas_gemm_example_managed
MALLOC=./cublas_gemm_example_malloc

# Arrays of sizes to test
MS=(5000 8000 14000 25000 55000)
NS=(5000 10000 25000 55000)
KS=(5000 9000 15000 25000 55000) 

for m in "${MS[@]}"; do
  for n in "${NS[@]}"; do
    for k in "${KS[@]}"; do
      echo "Config: m=${m}, n=${n}, k=${k}"
      echo "----------------------------------------------"
      $MANAGED "$m" "$n" "$k"

      #$MALLOC "$m" "$n" "$k"

      $BASELINE "$m" "$n" "$k"

      echo
    done
  done
done

