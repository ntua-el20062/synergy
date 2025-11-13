#!/usr/bin/env bash
set -euo pipefail

BASELINE=./cublas_gemm_example
MANAGED=./cublas_gemm_example_managed
MALLOC=./cublas_gemm_example_malloc

MS=(10000 15000 25000 45000 55000)
NS=(10000 15000 25000 45000 55000)
KS=(10000 15000 25000 45000 55000) 

for m in "${MS[@]}"; do
  for n in "${NS[@]}"; do
    for k in "${KS[@]}"; do
      echo "Config: m=${m}, n=${n}, k=${k}"
      echo "----------------------------------------------"
      $MANAGED "$m" "$n" "$k"

      $MALLOC "$m" "$n" "$k"
 
      $BASELINE "$m" "$n" "$k"

      echo
    done
  done
done

