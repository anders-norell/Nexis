#!/bin/sh
# Sprint 2: train the schema-free baseline (efference copy on, schema channel and 1b head off),
# three seeds in parallel, one thread each, 1M episodes per seed (entropy annealed over 700k).
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ARGS="--minutes 105 --max-episodes 1000000 --ent-episodes 700000 --schema off --sl 0"
for s in 0 1 2; do
  python3 train_rl.py --seed $s --out results/s2_base_s$s $ARGS > results/s2_base_s$s.stdout 2>&1 &
done
wait
