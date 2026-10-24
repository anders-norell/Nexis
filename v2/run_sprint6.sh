#!/bin/sh
# Sprint 6 (1c) pilot, seeds 40 and 41. Two runs at a time, one thread each, 400 000 episodes, entropy annealed
# over 300 000 (as in the 5a pilot). Resumable: finished runs are skipped. Levels in bits per step are 1b's
# raster 44.2 / 22.1 / 11.05 (5a, halving rule).
#   sh run_sprint6.sh A           Baslinje + 1c with alarm at the three levels                   (8 runs)
#   sh run_sprint6.sh B <L> e|u   the other alarm condition + matched capacity baseline at the chosen level (4 runs);
#                                 the capacity baseline matches the condition that decided the level (e: with alarm)
#   sh run_sprint6.sh U           no level gives "with" a benefit: 1c without alarm at all levels  (6 runs)
#   sh run_sprint6.sh C           no benefit anywhere (after the collapse check): fixed cost 0.0001 (4 runs)
#   sh run_sprint6.sh D           still none: capture risk doubled (p_capture 0.5), cost 0.0001   (6 runs)
# Names: results/c6_<cond>_s<seed>; off = Baslinje (train_rl.py), e = with alarm, u = without, cap / capu = capacity
# baseline with / without alarm.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ARGS="--minutes 240 --max-episodes 400000 --ent-episodes 300000"
S="40 41"
phase=$1
JOBS=""
add() { for s in $S; do JOBS="$JOBS $1:$s"; done; }
case "$phase" in
  A) add off; add e44.2; add e22.1; add e11.05 ;;
  B) if [ "$3" = "u" ]; then add e$2; add capu$2; else add u$2; add cap$2; fi ;;
  U) add u44.2; add u22.1; add u11.05 ;;
  C) add e1e-4; add u1e-4 ;;
  D) add dc-off; add dc-e1e-4; add dc-u1e-4 ;;
  *) echo "phase A|B <level>|U|C|D"; exit 1 ;;
esac
for j in $JOBS; do echo "$j"; done | xargs -P 2 -I{} sh -c '
  name=$(echo "$1" | cut -d: -f1); s=$(echo "$1" | cut -d: -f2)
  n=results/c6_${name}_s${s}
  if [ -f $n.final.npz ]; then echo "skip $n"; exit 0; fi
  pc=0.25; base=$name
  case "$name" in dc-*) pc=0.5; base=${name#dc-} ;; esac
  case "$base" in
    off)   cmd="train_rl.py --schema off" ;;
    e1e-4) cmd="train_1c.py --alarm --rate-cost 0.0001" ;;
    u1e-4) cmd="train_1c.py --rate-cost 0.0001" ;;
    e*)    cmd="train_1c.py --alarm --rate-target ${base#e}" ;;
    u*)    cmd="train_1c.py --rate-target ${base#u}" ;;
    capu*) cmd="train_1c.py --shuffle --rate-target ${base#capu}" ;;
    cap*)  cmd="train_1c.py --alarm --shuffle --rate-target ${base#cap}" ;;
  esac
  echo "start $n $(date +%H:%M:%S)"
  python3 $cmd --seed $s --p-capture $pc --out $n '"$ARGS"' >> $n.stdout 2>&1
  echo "end   $n $(date +%H:%M:%S)"
' _ {} >> results/c6_runner.log 2>&1
