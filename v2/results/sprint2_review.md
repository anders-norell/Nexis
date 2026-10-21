# Sprint 2 — independent gate review (22 Oct 2026)

Reviewer: separate agent that had not seen the work being produced. Read-only review of code and
results against CLAUDE.md and the spec.

**Verdict: PASS WITH NOTES.** No information leakage, no error in the RL math, gate numbers
reproduce on fresh episodes.

## Findings

1. **NOTE — no leakage.** `features()` reads only obs (periphery, cue, fovea, efference) plus
   `env.B` and static config. Remap uses the efference argmax only; never `info["actual"]`,
   `env.target`, `env.w`, `env.salient`. Trained weights run against a stub env carrying only `B`
   give 89.95 % on 2000 new episodes (seed 1) — same as the gate figure. Schema channel is zero
   and the 1b head has no effect with `--schema off --sl 0`. With the efference copy zeroed the
   hit rate falls to 63.3 %: memory runs through the efference copy as designed.
2. **NOTE — RL math correct; tests pass (15/15).** Full A2C surrogate checked against finite
   differences (worst rel. err. 1.3e-5). GAE terminal bootstrap, entropy and value gradients,
   active-step advantage normalisation, masking, clipping and Adam checked by hand. Weak spots:
   the "no gradient when off" tests pass trivially; no test of the remap under capture (covered
   by the stub-env run).
3. **NOTE — evaluation valid.** Seeds disjoint (train 10000+s, validation 500000+s, test
   900000+s); random agent on the same episodes; stochastic policy as locked; checkpoint
   selection negligible (2000 new episodes, best/final: 89.4/89.25, 89.95/90.05, 87.75/88.9 %;
   random 28.45 %).
4. **NOTE — beating the "inferred capture" reference (86 %) looks benign.** Forced capture every
   step: RL 5.6 % (intended memory 5.6 %, perfect schema 4.5 %). Capture forbidden: 99.8 %.
   p_capture 0.6: RL 50.7 / intended 39.3 / inferred 50.0 / perfect 60.3 %. p_capture 1.0: 23.1 /
   18.8 / 22.9 / 26.9 %. The RL agent tracks the inferring agent, well below a perfect schema.
   The gain is all at window 1 (85.6 vs 79.2 %; window 2: 94.2 vs 94.1 %). Plausible mechanism:
   if every target-coloured cell is marked covered, a mark must be false, so re-search; a
   hand-coded version of that rule lifts the inferring agent to 83.2 % at window 1.
5. **SHOULD FIX — 71.5 % reference not traceable in the repo.** → Fixed: `v2/references_sprint2.py`
   regenerates all references, including the memoryless agent with quadrant prior (best of
   weights 1.5/2/3/5: weight 3, 71.4 %; the spec's 71.5 % was a rounding slip). Threshold timing
   verified: locked before training started.
6. **SHOULD FIX — CLAUDE.md stale.** → Fixed in the same commit. One training run per seed, no
   discarded attempts; training stopped at the 1M-episode cap after ~49 min.
7. **NOTE — exploited regularity.** Salient objects are always distractors; the agent sends only
   0.8 % of target-colour commands to salient cells (base rate 13 %). ~1 point of hit rate. For
   H5: captures are always pulls away from the intended place; the H5 null model must use the
   same behaviour.
8. **NOTE — biological realism reasonable.** Retinotopic per-cell GRU with corollary-discharge
   remapping. The remap is noise-free; consider efference noise later.

## Implications for sprint 3

Headroom is small at standard settings (baseline ~89–90 % vs 93.5 % perfect-schema reference) —
v1's failure mode again. H2 and H5 are intervention tests, so this need not sink the MVP, but a
schema agent may learn to ignore a channel that adds little reward. Pilot should first:

1. Train the fixed-schema (1a) agent at standard settings; measure gain over this baseline; run
   lesion and shuffled-schema tests to see whether the policy reads the channel.
2. If the gain is under ~3 points, move the operating point before locking thresholds: window 1
   only (RL 85.6 % vs perfect schema 91 %), higher capture rate (gap ~10 points at p_capture
   0.6), or a fovea that reveals less about where it landed. Spec change; needs product-owner
   approval.
3. Make the H5 null model account for the agent avoiding salient cells.
