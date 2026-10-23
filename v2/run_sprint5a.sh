#!/bin/sh
# Sprint 5a pilot. Two at a time, one thread each, 400 000 episodes (entropy annealed over 300 000, as in
# sprint 3's pilot), standard working point. Resumable: finished runs are skipped.
#   sh run_sprint5a.sh A        Baslinje + the two reference variants (cost 0.001 and 0.0001), seeds 20, 21
#   sh run_sprint5a.sh B T50 T25 T12   the three levels, regulated to the given bits per step
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ARGS="--minutes 240 --max-episodes 400000 --ent-episodes 300000"
phase=$1
if [ "$phase" = "A" ]; then
  JOBS="off:20:0:0 off:21:0:0 ref1e-3:20:0.001:0 ref1e-3:21:0.001:0 ref1e-4:20:0.0001:0 ref1e-4:21:0.0001:0"
else
  JOBS="lvl50:20:0.001:$2 lvl50:21:0.001:$2 lvl25:20:0.001:$3 lvl25:21:0.001:$3 lvl12.5:20:0.001:$4 lvl12.5:21:0.001:$4"
fi
for j in $JOBS; do echo "$j"; done | xargs -P 2 -I{} sh -c '
  name=$(echo "$1" | cut -d: -f1); s=$(echo "$1" | cut -d: -f2)
  cost=$(echo "$1" | cut -d: -f3); target=$(echo "$1" | cut -d: -f4)
  n=results/p5_${name}_s${s}
  if [ -f $n.final.npz ]; then echo "skip $n"; exit 0; fi
  if [ "$name" = "off" ]; then schema="--schema off"; else schema="--schema channel --rate-cost $cost --rate-target $target"; fi
  echo "start $n $(date +%H:%M:%S)"
  python3 train_rl.py --seed $s $schema --out $n '"$ARGS"' >> $n.stdout 2>&1
  echo "end   $n $(date +%H:%M:%S)"
' _ {} >> results/p5_runner.log 2>&1
