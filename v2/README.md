# Nexis v2 — attention-schema predictions in small RL agents

Report: [`../docs/report_v2.md`](../docs/report_v2.md) (draft for external review, not peer reviewed).
Illustrated episode explaining the task and conditions (constructed, not a recorded run): https://claude.ai/artifact/YaJypo4vjvo7cFUKTaswsz#en

## Start here (reviewers)

```
python3 reproduce.py tables      # seconds: main tables from the committed result files
python3 reproduce.py evaluate    # ~35 min: re-run every evaluation on the committed checkpoints, compare number by number
python3 reproduce.py train       # ~10-15 h on 2 cores: retrain every run from scratch, then evaluate
```

Requirements: Python 3, NumPy (Matplotlib only for plots). No GPU, no PyTorch: backpropagation is hand-written and
tested against finite differences (`test_agent_rl.py`, `test_schema_channel.py`, `test_1c.py`).

Checks run on 26 Oct 2026: `evaluate` reproduced all six result files exactly
([`results/reproduce_evaluate_report.md`](results/reproduce_evaluate_report.md)); retraining one run from scratch gave
bit-identical weights ([`results/reproduce_determinism.md`](results/reproduce_determinism.md)).

## Map

| Report section | Code | Result file |
| --- | --- | --- |
| 2.1 Task | `env.py` | – |
| 2.2 Agent | `agent_rl.py`, `train_rl.py` | – |
| 4.1 Competence | `eval_gate2.py` | `results/sprint2_gate.json` |
| 4.2 Main experiment (F, H1, H2, H3) | `bank.py`, `eval_mvp.py`, `supp_h2a.py` | `results/sprint4_mvp.json`, `sprint4_supp_h2a.json` |
| 4.3 H5 in 1b | `schema_channel.py`, `h5.py`, `h5_eval.py`, `h5_5a.py` | `results/sprint5a_h5.json` (lock: `h5_lock.json`) |
| 4.4 Emergent schema 1c | `schema_1c.py`, `train_1c.py`, `probe_1c.py`, `eval_main_1c.py` | `results/sprint6_main.json` (lock: `h5_1c_lock.json`, pre-registration `sprint6_main_prereg.json`) |
| 4.5 Step 0 | `train_alarm_direct.py`, `eval_step0.py` | `results/sprint6_step0.json` (pre-registration `sprint6_step0_prereg.json`) |
| 5 Measurement calibration | `test_probe_1c.py` | – |

Checkpoints of every run are in `results/*.best.npz` / `*.final.npz`; training logs in `results/logs/`. Independent
reviews at each gate: `results/*review*.md`. Run scripts per sprint: `run_sprint*.sh`. The day-by-day decision log
(in Swedish) is the Status section of `../CLAUDE.md`.

Locked evaluators (`h5_5a.py measure`, `eval_main_1c.py run`) check SHA-256 hashes of the code they depend on and
refuse to run if any of it has changed.
