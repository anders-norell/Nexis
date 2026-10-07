#!/bin/sh
# Sprint 2: condition A (schema) and C (parameter-matched, no schema), 5 seeds each.
export OMP_NUM_THREADS=1
for s in 0 1 2 3 4; do
  echo "python3 train.py --name s2_A_s$s --seed $s --schema 16 --flash-to-ctrl 0 > results/s2_A_s$s.log 2>&1"
  echo "python3 train.py --name s2_C_s$s --seed $s --schema 0 --n-h 74 --flash-to-ctrl 0 > results/s2_C_s$s.log 2>&1"
done | xargs -P 2 -I{} sh -c "{}"
