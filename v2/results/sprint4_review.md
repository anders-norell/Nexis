# Sprint 4 (MVP) – independent review, gate 4

**Verdict: approve with remarks.** The code implements the locked seed bank and H2 as written, the verdict (F passed, H1 not passed, H2 passed, H3a passed, H3b passed) follows correctly from the locked criteria, and I reproduced the key numbers through my own code. No leaks found, no runs discarded or restarted. My remarks are about interpretation, not about whether the verdict holds. The biggest one: the Blandat control for H2 (a) is close to empty by construction. That half of the H2 (a) criterion is nearly automatic and has to be reported that way.

## What I verified

- **Tests.** `test_bank.py` 13/13, `test_env.py` 0 failed, `test_h5.py` 0 failed. I skipped `test_agent_rl.py` (optional, unchanged since sprint 3).
- **Order of events.** Code commit `4b550d4` was made at 22:54:00 and training started at 22:54:10 (`logs/s4_runner.log`). `bank.py`, `eval_mvp.py`, `env.py`, `train_rl.py` and `agent_rl.py` are unchanged from `4b550d4` to `HEAD`. `train_rl.py` and `agent_rl.py` are also unchanged since `5b999ed`.
- **env.py hooks.** The diff against `5b999ed` only moves `rng.normal(0, sal_noise, salient.shape)` and `rng.random(B)` into `_normal_noise()` and `_uniform()`. These are the same calls in the same order with the same shapes, so the output is bit-identical. My own single-episode re-implementation (plain `SearchEnv`, `B=1`, `env.rng` = the episode's env stream) gave the same `found` and `steps` as `BankEnv` on 1 000 bank episodes. That confirms the per-episode streams and batch independence.
- **Seed bank.** Seeds are 2 000 000 + i with `SeedSequence(seed).spawn(3)` for env, agent and h2 (`bank.py:59-61`). Agent sampling uses Gumbel noise from the agent stream for each episode (`bank.py:229-231`). The H1 bank seeds (3 040 000 to 3 161 999) and the validation seed (500 000 + s, plain `default_rng`) do not overlap with the test bank, and training never touches the bank.
- **H2 line by line** (`bank.py:146-265`):
  - Timing: the cells are computed at `t == 3` from `vis` after three steps, and the lie first enters the schema used for the 4th command.
  - (a): `A = vis>0` (all cells covered in steps 1-3). After the lie, each such cell shows only `post`, i.e. coverage at t ≥ 3. So the lie persists and new coverage is re-registered.
  - (b): the candidates are target colour, not the target, not covered, and valid. Up to 3 are drawn from the h2 stream. `elig_b` requires at least one candidate.
  - Eligibility is "running after 3 steps".
  - "Aims there" uses `info["intended"]` and counts only at t = 3, 4, 5.
  - The cells and the h2 draws are identical with and without the lie.
  - Mode `off` passes `schema=None`, which gives zeros, so a lesioned agent gets no schema.
  - `test_schema_matches_training_rule` checks the bank schema against `agent_rl.schema_input`, and I read both: same rule, same t=0 behaviour, donor maps frozen when the donor episode ends.
  - The only target-dependent quantity is the exclusion of the target from the (b) cells. No target position reaches the agent in the unmanipulated runs.
- **Independent H2 check on a trained agent.** My re-implementation (`scratchpad/review/indep.py`, own schema construction and own aim counting) on `s4_fixed_s10`, episodes 0-999, matched `run_bank` exactly in every one of these: hits (95.5 %), steps, eligibility (471 / 459), per-episode `aim_a` and `aim_b`, both with and without the lie.
  - The first three commands are identical with and without the lie in all 1 000 episodes, for (a) and for (b).
  - Effects on this subset: (a) +0.255 ± 0.024, (b) −0.854 ± 0.045. These agree with the full-bank values, +0.219 ± 0.011 and −0.806 ± 0.023.
- **Full-bank reruns agree with the JSON.** I reran the full bank through `run_bank` for `s4_fixed_s10`, `s4_fixed_s12`, `s4_shuffled_s10` and `s4_shuffled_s12`. All H2 (a)/(b) means, SEs, n and hit rates are identical to `sprint4_mvp.json`.
- **judge()** (`eval_mvp.py:126-180`):
  - F: 1a minus Baslinje and 1a minus Blandat, each > 2 SE.
  - H1: 1a slope minus the other < −2 SE ("flatter").
  - H2: (a) sign +, (b) sign −. The 1a-vs-Blandat test uses the same sign, so for (b) it requires "more negative".
  - H3a: Baslinje minus 1a, hits < −2 SE and revisits > +2 SE.
  - H3b: lesion minus intact, same directions.
  - `all()` over seeds 10, 11 and 12 is required everywhere.
  - Runs used: the nine originals. `below_k2` is empty; the lowest run is 87.7 %, so no reruns were due.
- **Training.**
  - All nine `.stdout` files end with `done 3907 1000192` and contain no "resumed" line. The runner log shows nine start/end pairs, two at a time, one each.
  - `m_args` in every best/final file: `max_episodes 1e6`, `ent_episodes 700000`, `sl 0`, `reveal 1`, `p_capture 0.25`, `windows 1,2`, `batch 256`, `lr 1e-3`, with the right seed and schema.
  - There are no other s4 files, so no hidden attempts.
- **p3 agents.**
  - `eval_mvp.py` only builds `s4_*` names, and `sprint4_mvp.json` contains only `s4_*` runs.
  - `sprint3_pilot.json` has no `aim_a`, `aim_b` or `elig_*` fields. Its `aim_repeat_share` is the sprint-3 metric from `metrics.py`.
  - `test_bank.py` uses untrained agents only.
  - Nothing in the repo calls `run_bank` on `p3_*`. I cannot prove a negative outside the repo, but I found no trace of such a run.

## Findings

**Major (interpretation; the verdict is unchanged)**

1. **The Blandat control for H2 (a) barely receives a lie.** For Blandat, the (a) cells are the agent's own covered cells, and they are switched off in the *donor's* map. That map only shows them as covered if the donor happened to cover them too.
   - On `s4_shuffled_s12`, the 4th-command map differs from the no-lie map by 0.76 cells on average, and in 59 % of eligible episodes not at all. For 1a the figure is 6.1 cells, and never zero.
   - So "1a's (a) effect > Blandat's" (z = 10.7 to 19.1) mostly reflects that Blandat got almost no manipulated input. It does not show that Blandat ignores an equal-sized lie.
   - The (b) control is fair: 2.57 changed cells for Blandat against 2.79 for 1a.
   - The choice was declared before training, so it stands for the verdict.
   - **Fix (report):** state this in the gate text. Base the "schema is read" argument for H2 mainly on (b). Optionally report an (a)-control with an equal-sized input change, e.g. un-marking the cells the donor map shows as covered. That would be descriptive only.

**Minor**

2. **The size of the H2 (a) effect depends on the checkpoint.** The best checkpoints were picked at very different points in training: `fixed_s11` at 307 k episodes, `fixed_s12` at 512 k, `fixed_s10` at 973 k.
   - On the *final* checkpoints, (a) is +0.299 (s11) and +0.187 (s12), against +0.147 and +0.070 on the best checkpoints. (b) barely moves: −0.81 against −0.83.
   - The best-checkpoint choice is close to noise. For example, s11's validation hit rates are 0.948-0.959 from 256 k episodes onward, while the validation SE is about 0.005.
   - **Fix:** report that the threefold seed spread in (a) partly reflects checkpoint selection. F also holds on the final checkpoints: off 88.1 / 89.1 / 88.4 %, shuffled 88.5 / 88.5 / 88.5 %, 1a 95.3 / 95.3 / 95.4 %.
3. **Part of the (b) effect is mechanical.** The (b) lie never covers the target, so it acts as a pruning hint. 1a's hit rate rises from 95.1 % to 98.2-98.3 %. Episodes therefore end sooner: 1.93 against 2.28 counted commands (s10). This lowers `aim_b` by itself.
   - The effect survives a per-command rate: 0.364 to 0.078 aims per command, −0.286 ± 0.008.
   - It also survives in episodes that give all three commands in both runs: −0.94 ± 0.04, n = 635.
   - For (a) the bias runs the other way and is negligible: 2.29 against 2.26 commands, and a per-command difference of +0.074 ± 0.004.
   - **Fix:** report the per-command rate next to the locked measure.
4. **Blandat donors are intervened too** (declared choice). This lets a donor's changed trajectory leak into the recipient's map. I measured the effect and it is negligible.
   - Intervening only even episodes, so that their donors stay clean, gave per-episode aims identical to the full run, for (a) and for (b) (`s4_shuffled_s10`).
   - Only 24 of 500 donor episodes changed trajectory.
   - No fix needed. Mention it in one line.
5. **Process: the bank was evaluated on 8 runs before run 9 finished** (`eb503a5`, "no verdict until run nine"), so the 1a H2 numbers were visible while `shuffled_s12` was still training. No code or argument changed afterwards, and nothing could have influenced the run. Still, mention it.
6. **`eval_mvp.py` reuses cached runs.** `main` skips any run already present in `sprint4_mvp.json` (`eval_mvp.py:188`). If a checkpoint were replaced, the results would be stale without any warning. That did not happen here: the checkpoints predate the JSON, and my reruns match it. **Fix:** store a checkpoint hash with each run, or add a `--force` option.

## Interpretation notes

- **The SE choice changes no verdict.**
  - F, H3a and H3b pass with z ≥ 9.5.
  - H1 fails for every SE choice: 1a minus Blandat goes the wrong way in s11 and s12 (+0.0057, +0.0005), and the largest |z| is 1.56.
  - The H2 comparisons with Blandat are between different agents, so an unpaired SE is appropriate. Pairing over episodes could only shrink the SE.
- **What H2 (b) shows.** It is large because uncovered target-colour cells are where the 1a agent normally looks: about 1 of 2.3 commands. Once the map marks them as covered, it almost never aims there. This is genuine control through the map and not an input-distribution artefact:
  - It is the same for window 1 and window 2 (−0.89 against −0.76, s12), although isolated marked cells only occur in training for window 1.
  - Blandat receives a lie of the same size and responds 4-12× less.
  - However, the lie is in a record of *where attention has been*, and here the lie helps the agent. So (b) shows "the agent obeys its coverage map". It does not show that a model of attention misleads the agent into worse performance.
- **The asymmetry between (a) and (b) is informative.** Saying "you have not looked here" is followed weakly: +0.07 to +0.22, and hits fall by 2-3 points. Saying "you have looked here" is followed almost completely. The agent apparently keeps its own memory of what it saw (GRU plus remapped fovea) and lets it override a false "unvisited". Report (a) as a partial effect.
- **Blandat reacts to (b).** The effect is significant in all three seeds but changes sign: −0.07, −0.21, +0.07. At the same time, a lesion leaves its hit rate unchanged (88.7 % against 88.7 %). Its policy is therefore sensitive to the channel without making use of it. This is exactly why the comparison with Blandat is needed. It also means that "the channel is read" alone is not evidence of a model.
- **H3b** (lesion at 45-60 % against Baslinje at 88 %) shows that 1a has built its search around the map. Since the revisit measure is what the map records, H3's revisit half partly holds by construction.
- **H1's failure is clean.** All slopes lie between 0.358 and 0.379 with SE about 0.009. The schema does not change how search time grows with the number of distractors in found episodes; it raises the hit rate.
- **About 1M episodes.** The best checkpoints come from 307 k to 973 k episodes, so "1M episodes, to the baseline's advantage" refers to training, not to the evaluated checkpoint.
