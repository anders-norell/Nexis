#!/bin/sh
# Copy logs of finished sprint 3 runs (and the runner logs) into results/logs/, which is tracked.
# Live logs of runs in progress are git-ignored so the repo stays clean while training runs.
cd "$(dirname "$0")/results"
for f in p3_*.final.npz; do
  n=${f%.final.npz}
  cp -p "$n.log.jsonl" "$n.stdout" logs/ 2>/dev/null
done
cp -p p3_grid.stdout p3_confirm.stdout logs/ 2>/dev/null
true
# best checkpoints are ignored while a run trains (they appear mid-run); add them once finished
cd ..
for f in results/p3_*.final.npz; do git add -f "$f" "${f%.final.npz}.best.npz" 2>/dev/null; done
true
