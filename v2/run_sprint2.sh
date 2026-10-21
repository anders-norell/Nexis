#!/bin/sh
# Sprint 2: train the schema-free baseline (efference copy on, schema channel and 1b head off),
# three seeds. Seeds 0 and 1 run in parallel on one core each, then seed 2.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ARGS="--minutes 100 --max-episodes 1500000 --ent-episodes 1000000 --schema off --sl 0"
python3 train_rl.py --seed 0 --out results/s2_base_s0 $ARGS > results/s2_base_s0.stdout 2>&1 &
python3 train_rl.py --seed 1 --out results/s2_base_s1 $ARGS > results/s2_base_s1.stdout 2>&1 &
wait
python3 train_rl.py --seed 2 --out results/s2_base_s2 $ARGS > results/s2_base_s2.stdout 2>&1
