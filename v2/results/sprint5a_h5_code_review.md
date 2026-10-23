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
