# Nexis — a proof of concept of Attention Schema Theory

A small agent that tracks a target with its attention and carries an internal model (a "schema") of that attention, built to test predictions of Michael Graziano's Attention Schema Theory (AST).

**Product owner:** Anders Norell. **Built by:** Claude (Anthropic), October 2026.
**Status:** paused after v2. Code written by an AI and not yet independently reviewed — do not cite results before review.

## Results so far

| Stage | Outcome |
| --- | --- |
| Sprint 1: task and baseline agent | Passed. Steered attention: 66–68% exact-pixel accuracy vs 39% uniform, 1.6% chance. |
| Sprint 2 (v1): schema added | H0, H1 failed pre-registered thresholds. Schema helps by 2–3 points (p < 0.001). |
| v2: attention drift + coarse attention trace | H0, H0b, H1 failed. Agents without a schema do equally well. |
| Exploratory (v2, not pre-registered) | When attention drifted, schema reports were biased toward the *intended* location (31% vs ~7% chance). |

Full design, hypotheses, thresholds and results: [`docs/spec.md`](docs/spec.md).
Swedish project overview: [`docs/PROJECT_OVERVIEW.md`](docs/PROJECT_OVERVIEW.md).

## Code

| File | What it is |
| --- | --- |
| `autodiff.py` | Small reverse-mode autodiff on NumPy (PyTorch was unavailable in the build environment) |
| `test_autodiff.py` | Finite-difference check of every op |
| `env.py` | 8×8 field, drifting target, periphery + fovea views, distractor flashes, v2 drift |
| `model.py` | GRU controller, belief-driven bounded attention, optional attention schema |
| `train.py` | Training and evaluation; writes `results/<name>.json` |
| `diagnose.py` | Model loading and attention-capture profile |
| `eval_sprint2.py`, `eval_v2.py` | Hypothesis tests: paired bootstrap CIs and permutation tests |
| `run_sprint2.sh`, `run_v2.sh` | All training runs for each stage |
| `plot_sprint1.py` | Sprint 1 figure → `report/sprint1.png` |

`results/` holds every run's log (JSON), trained parameters (`.npz`) and the verdict files.

## Reproduce

Needs only Python 3, NumPy and Matplotlib.

```
export OMP_NUM_THREADS=1
python3 test_autodiff.py
./run_v2.sh          # 15 runs, about 30 minutes on 2 cores
python3 eval_v2.py
```

## Rules this project follows

- Thresholds are fixed before a run and never lowered afterwards.
- Every result is reported, including failures.
- A pass would show the mechanism behaves as AST predicts — nothing about experience.
