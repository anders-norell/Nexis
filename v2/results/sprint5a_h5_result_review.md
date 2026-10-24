# Gate 5a: independent review of the H5 result

Reviewer: separate agent, 24 Oct 2026. Files reviewed: `results/h5_lock.json`, `results/sprint5a_h5.json`,
`results/sprint5a_h5.stdout`, `results/sprint5a_h5_feasibility.json`, `results/test_h5_eval.txt`, `h5_5a.py`,
`h5_eval.py` (build/evaluate), git history. Nothing re-measured; no code edited.

## Verdict

**The results are valid.** The measurement was run once on the locked code, parameters, checkpoints and bank. The
verdicts follow the locked rule: **22.1 bits not passed, 11.05 bits not passed**; the reported levels (5.53, 44.2,
no-cost) also do not pass. In every one of the ten runs the schema's share on the intended window is *lower* than the
null model's, by 17 to 71 SE. It is not a near miss: the sign is the opposite of the AST prediction.

## What was checked

1. **Lock.** I recomputed SHA-256 for all 20 files in the lock (9 source files, the feasibility file, 10 `*.final.npz`):
   all match. `measure()` compares sha256, params and criterion with the lock and refuses to run on a mismatch. The lock
   (`988b8ae`, 15:57:02) comes before the first result commit (`63e81ee`, 15:57:26). The stdout grows one line per
   commit (1 to 11) across a single contiguous series of commits, and `sprint5a_h5.json` does not exist before the
   lock: one run, no reruns. Params as locked: QS 41, K_NULL 400, N_BOOT 1000, boot seeds 1000/1001, DROP_SHARED,
   EQUALIZE_DIFFUSE and CLASS_DIFFUSE on, plateau 0.1/0.01, seeds 20–21, bank 2 000 000 + i (i < 4000) = the MVP bank,
   checkpoint `final`. `test_h5_eval.py` re-run now: ALL PASSED, output identical to the stored file apart from the
   descriptive line.
2. **Verdicts.**
   - Main rule: for every run, `passed == (diff > 2·SE)`, and no run passes. A judged level needs both seeds →
     22.1 and 11.05 are "ej klarad".
   - Argmax sensitivity: was run for all ten runs (argmax was possible everywhere). It also fails everywhere
     (diffs −0.14 to −0.47). Same verdict → no "osäker".
   - Condition 1 (fallback bias up to −0.10 to −0.12 toward failing in the stress case): the judged diffs are
     −0.28 to −0.41, so even after removing the full stress bias they stay clearly negative. The fail verdict at 22 and
     11 bits does not depend on that bias. At 5.53 bits (−0.13/−0.17, reported only) the stress bias is about the same
     size as the gap, so that level should not be read as clearly below the null. It is still not a pass.
   - Condition 2 (no-fallback subset, ~45 % of events): no sign change; the largest shift versus the main result is
     +0.011 (0.85 SE of the difference, 1.1 of the subset's SE). **Not sensitive to the fallback estimate.**
   - Condition 3 (no-cost variant): it does **not** pass (−0.44/−0.35). The condition is not triggered: there is no
     pass to attribute to architecture.
3. **Internal consistency.** All ten runs: n_events, n_episodes, n_h5_events_all, hit and matched q (mass and argmax)
   are equal to the blind feasibility values. The null share equals the feasibility "bar" at the matched q. Mass q
   0.62–0.71, inside the feasibility range. `point_clipped`, `clip_flag` false everywhere; `boot_clipped_frac` 0
   (main, argmax and no-fallback). Bootstrap q intervals are narrow (width ≤ 0.10).

Minor, no effect on validity: the two seeds differ noticeably at 44.2 bits and no-cost (schema share 0.13 vs 0.22,
0.15 vs 0.23). This is within-run noise of the 1b modules, not a measurement problem.

## Interpretation

The rise in the schema's share on intended as bits fall (0.13–0.23 at 44–178 bits → 0.21–0.25 at 11 → 0.31–0.34 at 5.5)
goes together with the rest mass rising (0.39–0.47 → 0.64 → 0.74–0.76). It moves toward the ≈ 0.40 that an untrained,
diffuse module gives (rest mass 0.85). That fits a schema whose concentrated error mass increasingly gives way to
diffuse mass spread roughly in proportion to the cell counts of I and O. It does not fit a schema that falls back on
intent: at every level the share stays *below* the diffuse value, and a crude two-component reading (diffuse part at
≈ 0.40, rest concentrated) puts the concentrated error mass almost entirely (≈ 90 % or more) on O. In other words, when
the schema is wrong about a capture and puts mass outside the actual window, that mass goes mainly to the *other
salient windows*, not to the intended one. That is what a module tuned toward detecting salience would do. The null
share also falls (0.59 → 0.48) because by design it inherits the schema's diffuse share, so the narrowing gap at low
bits is partly mechanical and should not be read as a trend toward H5. Limits: the two-component split is a rough
reading and not a fitted model; the area ratio for these exact events was not computed (the 0.40 comes from a different
agent's event set); and the result concerns this 1b architecture (channel_e, alarm signal, 400k episodes) only. It says
nothing about 1c or about whether compression under other training regimes would pull toward intent.
