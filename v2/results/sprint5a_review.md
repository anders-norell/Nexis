# Sprint 5a – independent review (gate 5a)

**Verdict: approve with remarks.** I accept the code, the pilot and the condition b bookkeeping. **The H5 criterion proposed in step 4 should not be locked as written.** Findings 1–4 have to be fixed first. As specified, the null model cannot be matched to the schema at any level, and the comparison is biased toward failure by construction.

Blindness: I kept it. I computed no schema mass on the intended window, no behaviour outcome and no group counts, and I did not run H2 (b) on trained agents. The null-model numbers below come from the null model alone. I ran it on capture events from the trained **Baseline** (p5_off, no module) or from a random agent.

## What I verified

- **Tests:** `test_schema_channel` 14/14, `test_h5_5a` 12/12, `test_bank` 14/14, `test_env` 55/55 and `test_h5` 9/9 all pass.
- **Independent finite-difference check:** I wrote my own loss on an untrained alarm (E) module with real environment features, float64 and beta 0.3. Worst relative error was 3.9e-5. The bits formula KL(N(mu,s²)‖N(0,1)) is correct. The cost applies only to rows with a target (t > 0). The clip gradient is handled.
- **Bottleneck:** the decoder sees only z (`logit = w_d·z + b_d`). The policy gets only `[map, union]` from the module, and the policy has no matching or alarm input (`matching=False`). Neither the module nor the policy gets the actual window or the target position: `observe()` holds periphery, cue, fovea and efference only. The matching signal's channel indices agree with `h5.restricted`. The alarm is any mismatch, broadcast, and it fires exactly at detectable captures (tested).
- **Training:** I read the args of all 14 E and Baseline checkpoints. Seeds were 20/21, 400,128 episodes (1,563 iterations), ent-episodes 300k, p_capture 0.25, reveal 1, windows 1,2. There were no resumes. The pilot bank (SeedSequence(4 000 000 + i).spawn) is disjoint from training (`default_rng(10 020/10 021 + 7919·it)`) and from evaluation (500 020/500 021).
- **Regulation:** last-quarter bits were 44.0/44.1, 22.1/22.1, 11.1/11.1 and 5.5/5.5 against targets 44.2/22.1/11.05/5.53. Measured on the bank: 44.9/43.6 … 5.5/6.0, matching the spec.
- **Pilot numbers:** I recomputed them with my own code on the full 4,000-episode bank, using allowed quantities only, and they match exactly. `p5_eref1e-4_s20`: hit 87.42 %, 182.88 bits, accuracy over all steps 76.22 %, 2,586 events, accuracy at H5 events 57.19 %, mass on actual 0.256. `p5_ehalf11.05_s21`: 87.18 %, 11.44 bits, 64.89 %, 2,553 events, 26.52 %, 0.147.
- **Precondition:** the first module had 19.8–20.6 % (fail). E had 57.2 % and 57.4 % (pass).
- **Condition b:** ratios 2.36 (first module) and 2.37 (E) are both above 1.5. Accuracy over all steps: 0.0001 variant mean 76.31 %. At 44.2 bits: 75.61 % (−0.70 pp; per seed −0.19 and −1.22). At 22.1 bits: −3.0 pp. All match the JSON.
- **Step 1:** `locked_check` on the best checkpoints reproduces `sprint4_mvp.json` H2 (a)/(b) exactly: same n and means, e.g. +0.2192/−0.8058 for seed 10. The salient-object exclusion is documented in the docstring and the spec.
- **Blindness in the code:** `blind=False` and `group_counts` are not used outside h5_bank and its test. The pilot JSON stores no events.

## Findings (by severity)

**1. Blocking: q cannot be matched as proposed.** I measured the null's floor on Baseline capture events (531 events). Its mass on the actual window is 0.92 at q = 0, 0.45 at q = 0.8 and **0.33 at q = 1**. The schema's mass on the actual window is 0.26 (177 bits), 0.24–0.29 (44), 0.23–0.24 (22), 0.14 (11) and 0.09 (5.5). That is below the floor at every level, including the no-cost variant.

Matching on mass also departs from the spec's own definition. The spec defines träffsäkerhet ("Tekniska val i bygget") as argmax in the actual window, not mass.

Matching on argmax does not fully work either. The null's argmax accuracy is about 0.93 until q ≈ 0.8, then 0.59 at q = 0.9 and 0.24 at q = 1. It can match 57 % and 44 %, only just reaches 25 % (q ≈ 1, a null that ignores all evidence), and cannot reach 14 %.

*Fix:* the product owner chooses the matching measure. Report feasibility per level before locking. Pre-declare what happens when matching is impossible: the level is either "not assessable" or uses q = 1.

**2. Blocking: the schema and the null have different supports, which biases H5 toward failure.** The null spreads its mass over candidate windows only (intended, actual, other salient). The schema is a diffuse softmax over 36–81 cells: its argmax is in the actual window 57 % of the time, yet only 0.26 of its mass is there. All its mass on non-candidate cells goes into the denominator of "mass on intended / mass outside actual". The null has no such mass.

There is a second problem. When the capturing object is the only salient candidate (18–20 % of events, both agents), the null's ratio is **identically 1** for any q > 0, so no schema can beat it on those events.

The design fails a schema that falls back on intended only if that schema is also diffuse. *Fix:* put both on the same support. Either renormalise the schema's mass over the candidate windows, or use argmax categories for both. Use only events with at least one other salient candidate, and report non-candidate ("rest") mass descriptively. Cells must be counted the same way for both: `ma = actual & ~mi`, and windows can overlap.

**3. Major: the null's glimpse-replacement distribution decides the bar and needs a product-owner decision.** `observer_mixed` replaces the glimpse with a **uniformly** chosen candidate's content. Condition a says "a random value from its own distribution". Its rationale also says that errors then land according to the prior, which favours the intended place. Uniform replacement does not do that.

I compared the two on Baseline events, using the proposed ratio over windows:
- Uniform: 0.54 (q = 0.5) to 0.58 (q = 0.9).
- Prior-predictive (glimpse drawn from the capture prior): 0.72 to 0.77.

Prior-predictive replacement also lowers the floor of the null's mass on actual to about 0.25 at q = 0.9. The choice moves the bar by about 0.2, so it must be settled before lock. The synthetic test (p_capture 0.6, 2–3 salient) never contains single-salient events, which is why it missed the "identically 1" case in finding 2.

**4. Major: the H5 evaluation code does not exist yet.** `observer_mixed` is not called from `run_bank_module`, and the events do not keep the environment state. Nothing implements the q search, the ratio or the bootstrap. Before lock, it should be written, tested on untrained agents and synthetic cases, and pushed.

The bootstrap must resample **episodes**, since events cluster within episodes. It should either redo the q matching inside each resample or state that q is fixed. The overall rule across the three levels (any, all, or per level only) must also be pre-specified; at present the spec only says "per level, reported".

**5. Moderate: behaviour measure.** The yes/no leak is real but weak. It bounds how often the argmax lands in the intended window, but only within behaviour-eligible events, whose total was never reported. It was nonetheless known before the H5 proposal was written, and the spec says so. Option (i), a 20k bank, is acceptable if its seeds are defined and disjoint from 2 000 000+ and 4 000 000+.

E improves hit rate by only 0–1 pp, so the agent barely reads the schema. A null behaviour result would be uninformative, and the same applies to H2 (b) on 1b. I recommend (ii), descriptive, unless H2 (b) on 1b first shows that the map is read.

**6. Moderate: reading of the reference.** The text supports the mean reading. "Referensvärdet är medelvärdet över de två pilotfröna" is the only aggregation rule, and the factor 1.5 test in `pilot_5a.py` uses means. But the decision is within noise: 0.70 pp against a 1 pp margin, a per-run SE of about 0.4 pp, and a seed spread at 44 bits of 0.85 pp.

Under the mean reading the lowest level is 5.5 bits. There the module is wrong in 52–53 % of all steps, and the null cannot be matched on any measure (finding 1). The product owner should decide. If the mean reading is kept, pre-declare 5.5 bits as reported, not judged, unless finding 1 is fixed.

**7. Minor: wording in step 1.** "Ingen av E1, E3 och E4 förklarar asymmetrin" overstates the case for E1. The capture-only cells sit at a floor: 0.00 with SE 0 in two seeds. E1 was **not tested**, which is not the same as no support.

The obvious E1/E2 account predicts the asymmetry: aimed (a)-cells were actually seen as non-target objects, while (b)-cells have no contrary memory. The headline should say "E3 and E4 not supported; E1 not decidable". The salient-object exclusion is justified, a design zero, and transparently reported.

**8. Minor: an unmeasured claim.** The first module "säger nästan alltid 'där jag siktade'" (spec and CLAUDE.md) is a statement about where errors go, and it was never measured (blind). It should be rephrased as a hypothesis.

**9. Minor: the 5b plan.**
- Lock the bit targets as numbers taken from the pilot.
- Re-check the precondition (≥ 50 %) in seeds 30–32, with a stated consequence if it fails.
- Pre-declare the checkpoint. "Best" is chosen on the policy's hit rate, which is unrelated to the module; "final" is better for a measurement made on the module.
- The module loss still falls 3–7 % between the second and fourth quarters, so "plateaus at 200k" is slightly generous. 400k is acceptable if all levels use it.

## Interpretation notes

- At every H5 event the alarm is on by definition, because H5 events are exactly the detectable mismatches. Any mass on the intended window therefore means the module throws away a fully reliable "not where you aimed" signal. That is what compression is expected to drop first, which reinforces condition c: a yes on H5 is a consequence of compression, not evidence specific to AST.
- E's design pushes the module toward "the most salient object" at captures, so errors could plausibly go to other salient windows. The candidate-restricted comparison in finding 2 is what separates "falls back on intent" from "guesses salience".
- Bits are a loose KL upper bound. A shift common to all cells costs bits but carries no information, and accuracy is flat from 177 to 44 bits, while naming one cell needs at most about 6.3 bits. "50 % of reference" is therefore a relative dose in loose units, not a capacity in any absolute sense.
- At 5.5 bits the module fails generally, and H5 errors there mostly reflect that general failure.
- The schema helps the agent by 0–1 pp. Whatever H5 shows concerns a model of attention that the agent barely uses for control. It is not evidence about a control-relevant self-model, which is for 1c to test.
