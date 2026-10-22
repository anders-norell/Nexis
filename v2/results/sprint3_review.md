# Sprint 3 pilot: independent review

**Verdict: approve with remarks.** The pilot's central results (working point, 1a gain, lesion, partial lesions, flat H6 curve) are correct, match the JSON and are free of leaks. Two things must be fixed before thresholds are locked: the proposed H1 measure (finding 1) and the "paired episodes" assumption the H2 design relies on (finding 2). The architectural explanation for the 1b result (claim 5) is plausible but overstated (finding 3).

## What I verified

- Tests: `test_env.py` 55/55 ok, `test_h5.py` 0 failed, `test_agent_rl.py` 21 ok and **1 FAIL**: "module gradients match finite differences", worst rel err 2.55e-4 (see finding 6).
- `reveal=1` bit-identical to sprint 2: I stepped the current `env.py` and the sprint-2 `env.py` (commit 2393902) side by side for 3 seeds × 25 steps with the same actions. All observations, rewards and actual windows were identical, and the RNG states matched afterwards (`env.py:263` draws nothing when reveal = 1).
- Masked glimpse (`env.py:209-214`) depends only on `w`, and `show` is drawn independently of position. It carries no location information.
- Module mode: no leak. The module sees only `obs` (`train_rl.py:54`, `features(..., None)`). The true window `prev_actual` is used only as a training target in `mrecs`. `schema_input` returns None for 'module' (`agent_rl.py:214`). Timing matches 1a (prediction of window t-1 fed at step t). I checked `module_grads` by reading it and by finite differences (see finding 6). It is correct.
- Lesion: `test(mode='off')` sets `schema_mode='off'`, so the schema stays None and the module (if any) runs but is not fed in. `zero_channel` monkeypatches `train_rl.schema_input`, which `rollout` looks up at call time, so it works as claimed (`pilot_eval.py:40-58`).
- Numbers: claims 2-4 match `sprint3_pilot.json` / `sprint3_slope_capture.json` (grid gains 6.8-7.7 / 10.1-12.1; confirm base 86.4-87.5, 1a 94.2-95.0, shuffled 86.4-88.3, lesion 47.6-57.9, covered-zeroed 77.6-83.0, window-zeroed 94.2-94.7; slopes 0.477-0.517 / 0.369-0.390 / 0.472).
- Working-point rule: applied correctly. The standard point is the first candidate in the rule's order, and it passes both criteria on seed 0 (gain 7.2 pp, lesion cost 44 pp).
- Seed collisions between training (10000+s+7919·it), validation (500000+s), test (900000+s) and slope (7x0000+…) streams: none for it < 10^4.
- H5 blindness kept: `pilot_eval.py:164-167` stores only counts. No head mass or argmax appears anywhere in `results/`. I deliberately did not inspect the head at capture events myself.

## Findings

1. **Major: the proposed H1 measure mostly restates F.** `slope_capture.py` averages steps over *all* episodes. Failed episodes run to the budget ceil(1.2(k+1)), which itself grows with k, so the slope is roughly 1.2 × failure rate plus the true search slope. I re-ran seeds 1 and 3 with a found-only slope:

   | Condition | All episodes | Found only |
   |---|---|---|
   | 1a s1 | 0.369 | 0.354 |
   | 1a s3 | 0.383 | 0.361 |
   | Baseline s1 | 0.477 | 0.384 |
   | Shuffled s1 | 0.472 | **0.362** |

   Hit rates are 98 % vs 90 %. Among found episodes, shuffled is as flat as 1a. The gap of about 0.1 steps per distractor is almost entirely failures hitting the budget, which F already measures. Either use found-only slope, or set a budget large enough that nearly every episode finds the target, or drop H1 as a separate criterion. Switching to "with capture" after seeing the no-capture slope is acceptable for a pilot (fresh seeds 10-12 in MVP), provided it is documented as a post-hoc design choice. But the measure itself must change first.

2. **Major: test episodes are not paired across conditions.** The `pilot_eval.py` docstring and the MVP plan say "same episodes in every condition". Only the first batch of 500 is shared. The env RNG advances by step count (capture, noise and reveal draws), so later resets diverge. I checked baseline vs 1a, seed 1: targets are identical in batch 0 and differ in batches 1-3. The reported SEs (`np.hypot`, `pilot_eval.py:186`) assume independence, so they are valid. The problem is H2: "intervention vs the same episodes without intervention" needs a deep copy of env, hidden state, `visited` and both RNGs at step 3, or a fixed per-episode seed bank.

3. **Major: the claim-5 explanation is overstated.** `features()` remaps the glimpse onto the *intended* cells (`agent_rl.py:74-88`). Each of those cells also carries its own periphery (occupied, colour). So detecting "my intended cell was not looked at" is *local*: the cell compares the glimpse with its own periphery. Only *localising the capturer* needs the global mean `g` (`agent_rl.py:104`). For the covered map, un-marking the intended cells is most of the value. Two alternative causes are not ruled out:
   - (a) The module/1b feeds `exp(slogp)`, a distribution summing to 1, as the "current window" and builds its union with `max`. For w = 2 each cell is about 0.25 instead of 1 as in 1a (`train_rl.py:59-63`, `agent_rl.py:218-219`). This is a scale mismatch in the covered map.
   - (b) The shared-head 1b is one step staler than 1a (its prediction of t-1 enters at t+1).

   Also, "module correct in 99.9-100 % of steps without capture" is not in any results file. Rephrase as "architecture *may* make H5 easy". Note too that a per-cell match bit (option A) adds no new information, only a hand-built feature.

4. **Minor: 1b/module results lack provenance.** They appear only in the spec, not in `sprint3_pilot.json`. My 2 000-episode re-evaluation broadly agrees:

   | Run | Re-evaluated (intact / lesion) | Spec (intact / lesion) |
   |---|---|---|
   | Module s1 | 88.3 / 86.1 | 88.4 / 86.6 |
   | Module s2 | 89.0 / 87.4 | 88.8 / 86.5 |
   | sl 0.3 | 86.8 / 85.0 | 85.2 / 84.5 |
   | sl 0.1 | 89.0 / 88.0 | 89.0 / 86.8 |

   SE is about 0.7. Write the evaluation to JSON. The with-capture slope exists for shuffled seed 1 only, and for no seed-0 run.

5. **Minor: H2 is underspecified.** Before locking, define:
   - (i) whether the lie persists (edit `visited`, which is a running max, so a one-shot edit is restored at step 4) or is one-shot;
   - (ii) "aims there" for w = 2 (centre cell vs window overlap);
   - (iii) what happens in episodes with fewer than 3 uncovered target-colour cells, and whether the target cell may be among them;
   - (iv) conditioning on episodes still running at steps 4-6;
   - (v) shuffled as a negative control for H2 and H3. It ignores the channel (lesion has no effect), so it shows specificity.

   Note that H2(a) is nearly implied by H3 given a 44 pp lesion cost. Report effect sizes, not only signs.

6. **Minor: the failing gradient test is a tolerance artefact, not a bug.** The failure is on `Uu[2,2]`: gradient about 9.8e-7, rel err 2.5e-4 at eps 1e-6 and 2.9e-5 at eps 1e-5. Use a mixed absolute/relative tolerance, or a larger eps. The suite currently exits non-zero.

7. **Minor: multiple criteria.** F, H1, H2a, H2b and H3 × 3 seeds, with "> 2 SE and same direction in all seeds", is conservative per criterion. Pre-state how the overall gate is read (all required? H2 primary?).

8. **Minor: interpretation limits.**
   - H6 "unsupported" holds for this architecture only. A blind fovea making no difference is consistent with finding 3 (glimpse barely used), not a property of the task.
   - The lesioned 1a (50 %) falls far below the baseline because zeros are out of distribution. H3 shows dependence, not model-likeness.
   - Best-checkpoint selection (by validation max, 205k-358k episodes) is the same in all conditions. That is fine.
