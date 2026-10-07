#!/bin/sh
# v2 main runs: A (schema gets trace), C (no trace, no schema), C2 (trace to controller). 5 seeds.
export OMP_NUM_THREADS=1
for s in 0 1 2 3 4; do
  echo "python3 train.py --name v2_A_s$s --seed $s --schema 16 --flash-to-ctrl 0 --v2 1 --trace-to schema > results/v2_A_s$s.log 2>&1"
  echo "python3 train.py --name v2_C_s$s --seed $s --n-h 73 --flash-to-ctrl 0 --v2 1 > results/v2_C_s$s.log 2>&1"
  echo "python3 train.py --name v2_C2_s$s --seed $s --n-h 72 --flash-to-ctrl 0 --v2 1 --trace-to ctrl > results/v2_C2_s$s.log 2>&1"
done | xargs -P 2 -I{} sh -c "{}"
