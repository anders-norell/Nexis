#!/bin/sh
# Sprint 4 (MVP): Baslinje (off), Fast schema 1a (fixed), Blandat schema (shuffled), training seeds
# 10, 11, 12: nine runs of 1 000 000 episodes, two at a time, one thread each. Standard working
# point (reveal 1, p_capture 0.25, windows 1 and 2). Entropy annealed over 700 000 episodes as in
# sprint 2's 1M-episode runs. Best checkpoint chosen on validation episodes (seed 500 000 + seed);
# the test bank (seeds 2 000 000 + i) is never used in training.
# Resumable: rerunning continues unfinished runs from their checkpoints and skips finished ones.
#   sh run_sprint4.sh                 the nine runs
#   sh run_sprint4.sh off:1010 ...    named extra runs (pre-registered rerun rule: seed + 1000)
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ARGS="--minutes 600 --max-episodes 1000000 --ent-episodes 700000 --sl 0"
if [ $# -gt 0 ]; then JOBS="$*"; else
  JOBS="off:10 fixed:10 shuffled:10 off:11 fixed:11 shuffled:11 off:12 fixed:12 shuffled:12"
fi
for j in $JOBS; do echo "$j"; done | xargs -P 2 -I{} sh -c '
  c=${1%%:*}; s=${1##*:}; n=results/s4_${c}_s${s}
  if [ -f $n.final.npz ]; then echo "skip $n"; exit 0; fi
  echo "start $n $(date +%H:%M:%S)"
  python3 train_rl.py --seed $s --schema $c --out $n '"$ARGS"' >> $n.stdout 2>&1
  echo "end   $n $(date +%H:%M:%S)"
' _ {} >> results/s4_runner.log 2>&1
