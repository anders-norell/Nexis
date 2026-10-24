# Gate 5a: independent review of the H5 evaluation code

Reviewed: `v2/h5_eval.py`, `v2/h5_5a.py`, the `v2/h5_bank.py` changes (commit ad3e092) and `v2/test_h5_eval.py`. I checked them against the product owner's decisions of 24 Oct 2026 and findings 1–4 of `sprint5a_review.md`.

**Verdict: reject for locking in its current form.** The code implements the 24 Oct decisions faithfully, and I found no bugs in the parts I checked. But the decided measure, as implemented, is **biased toward fail by construction** (finding 1). The bias is about as large as any effect we could plausibly see. One change, which needs the product owner's approval, fixes it, and I have tested that change on synthetic data. Once it is in, I would approve with the remaining remarks.

**Blindness:** I kept it. I ran no trained agent. I did not compute the schema's I or O split, H5's outcome or the behaviour groups, and I did not create `h5_lock.json` or run `measure`. All my checks use the random agent or untrained agents, with synthetic schemas built from the null model. The blind feasibility run had already finished when I read `results/sprint5a_h5_feasibility.json`. I used only its blind-safe fields: renormalised mass on A, argmax in A, and rest mass.

## What I verified

- **Tests:** `test_h5_eval.py` passes 22/22, and my rerun gives output byte-identical to `results/test_h5_eval.txt`. `test_h5_5a.py` passes 0 failed.
- **Regions:**
  - I is the intended window. A is the actual window minus I. O is the union of the other candidate windows, minus I and A.
  - Candidates are salient objects outside the intended window, which is the same rule as `env.step`'s eligibility.
  - Cells are counted the same way for schema and null. The null's window posterior is spread evenly over each window's w² cells, so overlapping windows add up.
  - `has_other` is O.any(). This is slightly stricter than "another salient candidate": a candidate whose window lies entirely inside I ∪ A does not count. It is reasonable but should be stated in the lock text.
- **Prior:** `h5_eval.prior` reproduces the environment's capture rule, cap included. At q = 0 the null equals `observer_restricted` on live environment states, with a worst error of 7.5e-15 (the test's own check).
- **Glimpse replacement:** the replacement source is drawn from the capture prior with true salience, and the test at q = 1 confirms the mean answer equals the prior. Note that the prior **includes the intended window ("no capture") with probability 1 − Σp ≥ 0.4**. This is the only reading that avoids a near-trivial pass: if replacement were conditioned on a capture, the null's mass on I would be ≈ 0. The lock text should say this explicitly.
- **Salience replacement:** each eligible candidate's value is independently replaced, with probability q, by a draw from U(s_min, 1). The same random numbers are used across q.
- **Likelihood:** the glimpse is compared with each candidate's content on the restricted channels (occupied and colour), as in the module's matching signal. `event_detail` uses no randomness, so blind and unblinded runs see the same events.
- **Matching:** q is matched on the mean renormalised mass on A, with linear interpolation and first crossing on a 41-point grid. The null's mass curve is monotone non-increasing on my synthetic events.
- **Reserve order:** chosen per level: mass in every seed, else argmax in every seed, else not assessable. This matches the decision.
- **Statistics:** the share is a ratio of sums for both schema and null. The bootstrap resamples episodes and re-matches q in each resample. The pass rule is diff > 2 SE per seed, and a level counts as "klarad" only if both seeds pass. There is no overall yes, as decided.
- **Cluster bootstrap check (mine):** I duplicated every event 5× within its own episode, and the SE was unchanged (0.0009 → 0.0009). When I gave each copy its own episode, the SE shrank as expected (0.0004 = 0.0009/√5).
- **Blindness in the code:**
  - In blind mode only `sA`, `s_cand`, `s_rest` and `s_argA` are stored. None of them splits I from O.
  - `evaluate` refuses blind data, and `measure` refuses without the lock file.
  - With `detail=False`, `h5_bank` behaves exactly as before.
- **Levels and checkpoints match the decisions:** 22.1 and 11.05 bits are judged; 5.53, 44.2 and the no-cost variant are reported. Final checkpoints, seeds 20–21, MVP bank 2 000 000 + i, 4 000 episodes.

## Findings

### 1. Blocking: diffuse schema mass on candidate cells biases H5 toward fail

Renormalising over candidate cells removes the schema's mass *outside* the candidates. It does not remove the diffuse part that lands *on* candidate cells. The module's map is a softmax over all valid cells, computed from a noisy channel, and it is diffuse. The blind feasibility run shows how diffuse:

| Run | Rest mass | Argmax in A | Renormalised mass on A |
| --- | --- | --- | --- |
| No-cost variant | 0.41–0.46 | 0.51 | 0.38–0.40 |
| 5.5 bits | 0.74–0.75 | 0.12–0.13 | 0.34 |

A uniform map gives a renormalised mass on A of 0.29 (random-agent events). At low bits, the mass on A therefore mostly measures window area, not knowledge.

That diffuse mass hurts the schema twice:
- It lowers the schema's mass on A. That pushes the matched q up, which raises the null's share on I, because the null loses mass on A preferentially toward the intended window, through the prior.
- It dilutes the schema's own share on I toward the area ratio, which is I/(I+O) ≈ 0.42.

**Synthetic test (random-agent events, 1 738 events).** I built a schema whose window-level beliefs are *exactly* the null's at q0, spread evenly like the null, then mixed in a fraction of uniform mass over the valid cells. The correct answer is a difference of 0:

| q0 | Diffuse share | Rest mass | Matched q | Code: schema share − null share |
| --- | --- | --- | --- | --- |
| 0.3 | 0 | 0.00 | 0.300 | −0.000 |
| 0.3 | 0.45 | 0.39 | 0.360 | **−0.047** |
| 0.3 | 0.65 | 0.56 | 0.414 | **−0.083** |
| 0.3 | 0.85 | 0.73 | 0.524 | **−0.143** |
| 0.6 | 0.45 | 0.39 | 0.627 | **−0.032** |
| 0.6 | 0.85 | 0.73 | 0.697 | **−0.113** |

The pilot runs' rest masses (0.38–0.75) fall right in this range. The built-in bias is −0.03 to −0.14 in share, against a bootstrap SE of roughly 0.01 for real schemas (0.007 for the untrained schema). A schema that falls back on intent moderately more than the null would be judged "ej klarad", and the 11-bit level is hit hardest. The tests do not catch this, because their synthetic schemas are built in region space with `s_rest = 0` and never pass through `schema_regions` (finding 5).

**Fix, tested:** give the null the same diffuse share as the schema before both are renormalised.
- Per event: null cell map = (1 − λ_e) · mixing-null map + λ_e · uniform over the valid cells. λ_e is chosen so that the null's rest mass equals the schema's rest mass, i.e. λ_e = rest_e / (1 − candidate area share_e).
- Rest mass is blind-safe, so this can go into `feasibility`.
- Uniform over the valid cells is the natural baseline, since the decoder's softmax runs over valid cells and its noise is the same in every cell.
- Then match q on renormalised mass on A, as decided.

I implemented this with a single λ for all events. In every row above it recovered q0 to within 0.001 and gave a difference of 0.000. A per-event λ_e is better and equally simple.

This changes the null model, so the product owner has to approve it. The schema-side alternative is to subtract the schema's mean mass per non-candidate cell from each candidate cell before renormalising. That is equivalent under uniform diffuse mass, and either can be reported as a sensitivity check of the other. Re-run feasibility after the change; the matched q values will drop.

### 2. Moderate: the null's even spreading within windows puts other candidates' mass on I cells

When windows overlap (w = 2), the null spreads the mass of the actual and other candidate windows evenly over their cells, some of which lie in I. In my sample the actual window overlaps I in 4.7 % of events, and some other candidate window does in 6.3 %.

A schema with the same window beliefs that puts each salient candidate's mass on the object's own cell (which always lies outside I) instead of across the window shows a bias of **−0.007** (synthetic z ≈ −3 to −4). It is small but has the same sign as finding 1.

*Fix:* treat cells shared between I and any other candidate window as ambiguous and leave them out of I, A and O for both schema and null. Otherwise, report a sensitivity run with them left out.

### 3. Minor (moot today): the argmax reserve has two flaws

The reserve is not used: feasibility chose mass matching at every level. Fix it anyway if it stays in the locked text.
- **Different supports:** `s_argA` takes the schema's argmax over the whole map, so an argmax in the rest counts as "not A". The null's argmax only ever falls on candidate cells. A lower schema value raises the matched q and the bar, so the bias is toward fail. *Fix:* take the schema's argmax over candidate cells only.
- **Plateau:** the null's argmax curve is flat at about 0.73 for q ≈ 0.1–0.45 (random-agent events), while the null's share on I rises from 0.23 to 0.49 over the same range. If the schema sits on the plateau, "first crossing" picks the smallest q, which gives the lowest bar, so the bias is toward pass. *Fix:* report the q interval of all crossings, and judge at its upper end, or mark the level not assessable.

### 4. Minor: the lock is not tied to the code

`measure()` only checks that `h5_lock.json` exists. *Fix:*
- Have the lock record SHA-256 hashes of `h5_eval.py`, `h5_5a.py` and `h5_bank.py`, and of `sprint5a_h5_feasibility.json`.
- Record in it QS, K_NULL, N_BOOT, the bootstrap seeds, the judged and reported levels and the criterion text.
- Make `measure()` verify all of these before unblinding.

### 5. Minor: test gaps

- The synthetic-schema tests bypass `schema_regions` and have no diffuse mass, which is why finding 1 went unnoticed. *Add:*
  - a cell-map synthetic test through `schema_regions`, with uniform noise added, which must give a difference ≈ 0 once finding 1 is fixed;
  - an object-peaked variant for finding 2.
- No test covers the null's argmax tie handling: an even split over tied cells, including overlap cells.
- The tests "regions are disjoint" and "frac rows sum to 1" are true by construction. They are harmless, but they verify little.
- No test covers the cluster bootstrap. My duplication check above could be added.

### 6. Notes (no change needed)

- Null Monte Carlo noise (K = 400 draws, with the same random numbers across q) is not in the bootstrap SE. It averages out over about 1 900 events and is negligible.
- **How high the bar is:** at q ≈ 0.7–0.8 the null's share on I is about 0.62–0.65 (random-agent events). To pass, the schema has to put roughly two-thirds or more of its error mass on candidates onto the intended window. That follows from the decided prior-predictive replacement and is intended, but the product owner should see the number before locking.
- With the fix in finding 1, the matched q values per level will change. The feasibility file should be regenerated and re-read (blind) before the lock.

## Re-review of fixes (commits 73c3c2b and 9eccd59)

**Verdict: approve with remarks.** One condition must be met before the lock: R1, which is cheap and blind-safe.

- Finding 1 is fixed under the model it assumes: diffuse mass spread evenly over the valid cells.
- Findings 2–5 are fixed, or reduced to a small residual.
- R1: the result is still sensitive to whether the schema's diffuse mass really is uniform. The blind feasibility numbers suggest it may not be. This must be checked blind, with a pre-declared response, before locking.

Blindness was kept. My checks used random-agent events and synthetic schemas only. From the trained runs I read only the blind-safe fields of the second feasibility run (`results/sprint5a_h5_feasibility_v2.json`, the same run as `_v2.stdout`). I started no runs on trained agents, created no lock and ran no `measure`.

### What I verified

- **Tests:** `test_h5_eval.py` passes 30/30, and my output is identical to the committed `results/test_h5_eval.txt`. `test_h5_5a.py` also passes. The new tests go through `schema_regions` on cell maps and include diffuse and object-peaked schemas, null argmax ties, the plateau rule and the cluster bootstrap. They test what they claim.
- **Diffuse fix (`build`):**
  - λ = (rest_schema − r_null) / ((1 − cand_share) − r_null) is correct. It sets the null's mass outside the regions, (1 − λ)·r_null + λ·(1 − cand_share), equal to the schema's rest mass. r_null is the null's own mass on the dropped shared cells, and λ is clipped to [0, 1].
  - Uniform mass is spread over the n × n valid cells, which is the same support as the decoder's softmax.
  - λ uses only the schema's rest mass, so it is blind-safe.
  - Both maps are renormalised over I + A + O.
- **Independent synthetic check:** a schema with the null's beliefs plus 65 % uniform noise gives diff +0.0009 (q0 = 0.4) and +0.0003 (q0 = 0.8), with q recovered to within 0.002. Before the fix the same kind of schema gave −0.06 to −0.14.
- **Shared cells (DROP_SHARED):** I is the intended window minus cells shared with the actual or another candidate window. A is the actual window minus the intended window. O is the other windows minus I and A. The test that windows are fully covered was updated to allow for the dropped cells.
- **Argmax reserve:**
  - The schema's argmax and the null's argmax are both taken over candidate cells only.
  - The null splits ties evenly; the new tie test confirms it.
  - Plateau rule: if the curve stays within 0.01 of the target over a q-span wider than 0.10, the reserve counts as impossible. Applied in both feasibility and evaluate.
- **Lock (`h5_5a.py lock` / `measure`):**
  - The lock stores SHA-256 hashes of h5_eval, h5_5a, h5_bank, h5 and the feasibility JSON, plus the parameters and the criterion text.
  - `measure` refuses on any mismatch, and `lock` refuses to overwrite an existing lock.
- **9eccd59:** feasibility now also reports the null's bar, i.e. its share on I at the matched q, for both matching methods. It is computed from the null alone; λ depends only on the schema's rest mass. Blind-safe.

### Remaining findings

**R1. Major, condition before lock: the fix assumes the schema's diffuse mass is uniform, and the answer is sensitive to that.**

In a synthetic test I put the diffuse mass on occupied cells (cells with objects) instead of spreading it evenly. Everything else was identical to the null (u = 0.65), but the result was diff **−0.14** (q0 = 0.4) and **−0.11** (q0 = 0.8), biased toward fail. The direction depends on where the diffuse mass sits. On cells near salient objects (O, A) it biases toward fail. On target-colour cells near the aimed cell (I) it would bias toward pass.

The blind feasibility v2 data hint that the diffuse mass is not uniform:
- **Mass-matched and argmax-matched q diverge as rest mass grows.** For a schema that was null plus uniform noise, the two would agree; they did in all my synthetic runs.

  | Variant | Mass-matched q | Argmax-matched q | Gap |
  | --- | --- | --- | --- |
  | No-cost | 0.68 / 0.71 | 0.72 / 0.72 | 0.01–0.04 |
  | 22 bits | 0.65 / 0.65 | 0.73 / 0.73 | ≈ 0.08 |
  | 11 bits | 0.64 / 0.63 | 0.74 / 0.74 | ≈ 0.10 |
  | 5.5 bits | 0.64 / 0.67 | 0.75 / 0.75 | ≈ 0.10 |

- **The mass-matched q is flat or falling across levels (0.63–0.71)**, although the argmax rate in A over candidates falls from 0.51 to 0.37.

*Fix (blind-safe, before lock):*
- (a) Compute the schema's mean mass per **non-candidate** cell for each trained run, by cell class: empty, occupied non-target colour, target colour, and possibly visited or unvisited. This touches neither I nor O.
- (b) Pre-declare a rule. If the class densities differ by more than an agreed factor (for example 1.5), the null's diffuse component uses class-matched densities: the schema's per-class rest density, applied to each candidate cell by its class.
- (c) Pre-declare that H5 is also reported with **argmax matching** as a sensitivity analysis. The bar it gives is now in the feasibility output (9eccd59).
- (d) If the mass-matched and argmax-matched results disagree, call the level "inconclusive" rather than "klarad" or "ej klarad".

This changes the reporting rule, so the product owner must approve it.

**R2. Minor: the object-peaked residual is acceptable.**

| Case | Diff | SE |
| --- | --- | --- |
| The test's own case | −0.0041 | 0.0019 |
| My check, q0 = 0.4 | −0.0030 | 0.0013 |
| My check, q0 = 0.8 | −0.0023 | 0.0012 |
| No diffuse mass | −0.0018 to +0.0006 | — |

Dropping every cell shared between any two candidate windows does not remove it (−0.005 / −0.003). Against a realistic SE for a trained schema of about 0.005–0.01, this is 0.2–0.8 SE, toward fail.

It does not matter for the verdict unless a judged level ends near the 2 SE line. *Fix:* state it in the lock text as a known conservative bias of at most about 0.005. Note that the test passes only through its +0.005 tolerance: the diff is 2.2 synthetic SE.

**R3. Minor: q is weakly identified at low bits.**

After the diffuse share is equalised, the null's mass curve becomes narrow:

| Level | Null mass curve spans |
| --- | --- |
| 5.5 bits | 0.27–0.49 |
| 11 bits | 0.25–0.63 |
| No-cost | 0.22–0.79 |

On a narrow curve the matched q is sensitive to small changes. The bootstrap re-matches q, so the SE will include this, but resamples may fall outside the curve and be clipped. *Fix:* report `boot_clipped_frac` and a bootstrap interval for q for every run. Pre-declare that a judged level with clipping above 5 % is reported as weakly identified.

**R4. Minor: the lock's hashes leave inputs out.**

The hashes do not cover `env.py`, `bank.py`, `pilot_5a.py` (`load`), `agent_rl.py` or `schema_channel.py`, or the ten checkpoint files (`*.final.npz`). Any of these changes the events. *Fix:* add them to `FILES`.

## Final check (commit 174ac46)

**Verdict: not lockable as implemented. One more small change is needed (F1).** Once F1 is fixed and passes its test, the thresholds can be locked without another review round, as long as the lock text states the points listed below.

I verified the following:
- `test_h5_eval.py` passes 32/32, and my output is identical to the committed file.
- **R1 implementation:**
  - The class densities are measured on non-candidate valid cells, with shared cells excluded.
  - D is the class density times each region's class counts, and λ is the total diffuse mass.
  - The null map is (1 − λ)·null + D, renormalised over I + A + O.
  - Blind-safe: only cells outside I ∪ A ∪ O are used.
  - A schema equal to the null plus uniform noise comes out at +0.0008 (q0 = 0.4) and +0.0003 (q0 = 0.8). My earlier occupied-cells case comes out at −0.011 / −0.007, down from −0.14 / −0.11.
- **R2:** to be stated in the lock text. With the class-matched diffuse part, my object-peaked case now gives +0.003 / −0.000.
- **R3:** `q_interval` and `clip_flag` (clipping above 5 %) are in place.
- **R4:** the lock now hashes all the code inputs and the ten final checkpoints.
- **`measure`:**
  - The argmax sensitivity run is used only where feasibility found argmax identified for that run.
  - A judged level becomes "osäker" only when both seeds have a sensitivity result and the two verdicts differ.
  - The sensitivity run uses its own bootstrap seeds (2000 + i).

### F1. Major: the cell classes ignore whether a cell has been visited, which biases toward pass

A learned schema may well put its spare mass where the spotlight usually goes next, which is unvisited target-colour cells.

I tested a synthetic schema: the null's beliefs, plus 65 % diffuse mass on target-colour cells not yet covered by the spotlight. The result is **+0.090 (z = 7.8) at q0 = 0.4 and +0.101 (z = 15.5, q clipped at 1.0) at q0 = 0.8**. That is a false pass.

The cause is in how the null's diffuse part is estimated:
- The density of the target-colour class is estimated over *all* target-colour rest cells, and the visited ones carry no mass, which dilutes the estimate.
- The intended window was not covered at the capture step, so its target-colour cell gets more diffuse mass in the schema than the null is given.

Such a schema puts mass on the intended window only because that window is "a plausible next place", not because the schema falls back on intent. H5 must not credit it.

*Fix:*
- Split each class by visited status, giving 6 classes: {empty, other colour, target colour} × {covered before the capture step, not covered}.
- Use the spotlight's actual coverage up to step t − 1 (`env.visits` **before** the step, not after). Then the intended and actual windows are classified the same way as at the time the agent aimed.
- Fall back to the colour-only density when a class has no rest cells.
- Add my case as a test, with the requirement |diff| < 0.01 at q0 = 0.4 and 0.8.

### What the lock text must state

1. **Scope:**
   - H5 on 1b is a descriptive test run once on the pilot's E runs: seeds 20–21, final checkpoints, MVP bank 2 000 000 + i, i < 4 000.
   - Levels 22.1 and 11.05 bits are judged. 5.53 bits, 44.2 bits (the reference) and the no-cost variant are reported only.
   - There is no overall verdict across levels, and 5b is not run.
2. **Events:**
   - Captures where the fovea excludes the intended window by occupied and colour, the capture does not find the target and the episode keeps running.
   - The map is read at step t + 1.
   - Only events where some other candidate window has cells outside I ∪ A count.
3. **Regions:**
   - I is the intended window minus cells shared with any other candidate window.
   - A is the actual window minus the intended window.
   - O is the other candidate windows minus I and A.
   - Shared cells count for neither the schema nor the null.
4. **Null model:**
   - The restricted ideal observer, weakened by mixing at rate q.
   - The glimpse is replaced by the content of a window drawn from the **capture prior, including "no capture" (the intended window) with probability 1 − Σp**.
   - Each salience value is replaced by a draw from U(s_min, 1).
   - K = 400 draws, using the same random numbers across q.
   - The window posterior is spread evenly over each window's cells.
   - It gets a diffuse part equal to the schema's own density per cell class, measured on the schema's non-candidate cells. The classes are colour × visited (after F1).
   - Both maps are renormalised over I + A + O.
5. **Matching:**
   - Primary: q matched on the mean renormalised mass on A.
   - Reserve: argmax over candidate cells, using the plateau rule (within 0.01 of the target over a q-span wider than 0.10 means q is not identified).
   - Otherwise the level is "ej bedömbar".
   - The same method is used for both seeds of a level.
6. **Criterion:**
   - Measure: the share of candidate error mass on I, as a ratio of sums.
   - SE from a bootstrap over episodes (1 000 resamples) with q re-matched in each resample.
   - Pass: schema share > null share by more than 2 SE, per seed. A judged level is "klarad" if both seeds pass.
   - Argmax matching is reported as a sensitivity analysis. If it gives a different verdict, the level is "osäker".
   - Runs with more than 5 % of resamples clipped, or a wide q interval, are reported as weakly identified.
7. **Known residual biases (synthetic tests):**
   - About −0.004 toward fail from object-peaked maps.
   - About −0.007 to −0.011 toward fail from diffuse mass on occupied cells.
   - Any diffuse structure the cell classes do not capture can bias either way. That is why the sensitivity rule exists.
8. **The bar:** the null's share on I at the matched q, for each run and both methods, from the blind feasibility file. It should be quoted as numbers, so the product owner sees how high the bar is.
9. **Interpretation (condition c):**
   - A "klarad" result is what compression predicts. It is consistent with AST but not specific to it.
   - The schema barely helps the agent (0–1 pp), so H5 here concerns a model of attention that is hardly used for control.
10. **Hashes:** SHA-256 hashes of the code, the feasibility JSON and the checkpoints, all verified by `measure`. The feasibility file must be regenerated after F1, before the lock.
