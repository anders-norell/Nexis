#!/bin/sh
# Copy logs of finished runs (sprint 3: p3_*, sprint 4: s4_*) and the runner logs into results/logs/,
# which is tracked. Live logs of runs in progress are git-ignored so the repo stays clean while
# training runs. Best and final checkpoints of finished runs are force-added.
cd "$(dirname "$0")/results"
for f in p3_*.final.npz s4_*.final.npz p5_*.final.npz c6_*.final.npz; do
  [ -f "$f" ] || continue
  n=${f%.final.npz}
  cp -p "$n.log.jsonl" "$n.stdout" logs/ 2>/dev/null
done
cp -p p3_grid.stdout p3_confirm.stdout s4_runner.log p5_runner.log c6_runner.log logs/ 2>/dev/null
true
cd ..
for f in results/p3_*.final.npz results/s4_*.final.npz results/p5_*.final.npz results/c6_*.final.npz; do
  [ -f "$f" ] && git add -f "$f" "${f%.final.npz}.best.npz" 2>/dev/null
done
true
