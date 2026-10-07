# Nexis PoC — Attention Schema Specification

Oct 7, 2026 · Product owner: Anders · Draft by Claude

Live version (editable, with architecture diagram): https://claude.ai/code/artifact/1a1627cc-72dd-4b2e-b93f-c73593336bf1
This file is a snapshot of that document, taken 7 Oct 2026 after the v2 results.

## Purpose and research question

This PoC tests whether an agent that reports on its own attention through an attention schema makes the systematic reporting errors that Attention Schema Theory (AST) predicts. That is the part of AST that concerns awareness, and it has not been tested in artificial agents.

**Already established (we replicate, not re-discover):**

- An attention schema improves endogenous control of attention; disabling it after training breaks performance ([Wilterson & Graziano 2021](https://dataspace.princeton.edu/handle/88435/dsp01kp78gk430)).
- With transformer attention, a schema improves judging other agents' attention and joint tasks, and the gain is not explained by extra parameters ([Farrell, Ziman & Graziano](https://arxiv.org/abs/2411.00983v3)).
- A schema can be built into the transformer's dot-product attention as a controller ([ASAC, 2025](https://arxiv.org/html/2509.16058v1)).

**What this PoC adds:** AST claims the schema is a simplified, partly inaccurate model of attention, and that a system's claims of being aware come from the schema, not from attention itself. We therefore ask: when an agent reports where it is attending using only its schema, where does that report diverge from its actual attention, and are the divergences the ones AST predicts?

**Why it matters for MACTA:** it turns the Awareness criterion from a definition into something measurable — a system whose self-report tracks a model of its own processing, with a known and testable error profile.

## Hypotheses

Four hypotheses, each stated so that a result can reject it. H1 is replication; H2 and H3 are the new contribution.

| ID | Hypothesis | Rejected if |
| --- | --- | --- |
| H0 | The schema is not just extra capacity: gains vanish in a parameter-matched agent without a schema | The parameter-matched control performs as well as the full agent |
| H1 | Ablating the schema after training degrades attention control | Task score after ablation is within 5% of the full agent |
| H2 | Reports generated from the schema track actual attention well in ordinary trials, but diverge systematically when attention is captured bottom-up by a distractor | Report accuracy is equal in captured and ordinary trials, or the errors are random rather than biased toward the schema's prediction |
| H3 | A report head reading attention directly (not the schema) does not show the H2 divergence | The direct report head shows the same divergence pattern as the schema-based head |

H2 is the artificial analogue of attention without awareness in humans: attention moves, but the system's model of its attention, and therefore its report, does not follow. H3 shows the divergence comes from the schema, not from reporting as such.

## System architecture

The agent has four parts — attention, controller, schema and report heads — and only the schema-based report is under test. (The architecture diagram is in the live document.)

- **Task.** A target moves across an 8×8 field for 12 steps; the agent must name its location at the end. Attention gates which of 16 patches the controller sees, so good attention control is needed to solve it.
- **Attention.** Softmax over the 16 patches. Logits = top-down query from the controller + bottom-up salience. Distractors raise salience for 2 steps without the agent choosing it.
- **Schema.** A small recurrent network with a narrow bottleneck. It sees the same input as the controller plus a copy of the query, and is trained to predict the next attention map. Its state is fed back to the controller, so it is used for control, as AST requires.
- **Report heads.** Small classifiers that answer "which patch am I attending to?". Head A reads only the schema state; head D reads the actual attention map. Both are trained on ordinary trials only.

## Experimental conditions and controls

Four agent conditions, each trained with 5 random seeds, plus one sweep over schema capacity.

| Condition | Schema | Report head reads | Tests |
| --- | --- | --- | --- |
| A — Full | Yes | Schema state | H1, H2 (reference) |
| B — Ablated | Trained, then replaced by its mean output | Schema state | H1 |
| C — No schema | None; controller widened to match A's parameter count ±1% | — | H0 |
| D — Direct report | Yes | Actual attention map | H3 |

**Trial types at test time:**

- *Ordinary:* one target, no distractor.
- *Captured:* a salient distractor flashes and adds a bottom-up bias to attention at its location for 2 steps, outside the agent's control.

**Capacity sweep:** schema bottleneck width 4, 8, 16 and 32 units. This guards against a trivial result: if only a starved schema fails to model capture, H2 says something about capacity, not about schemas.

The report heads are trained on ordinary trials only, then frozen. Captured trials are never seen during report training.

## Verification criteria

Each hypothesis passes only if its threshold is met with 95% confidence across 5 seeds. The thresholds below are proposals for the product owner to confirm before any training run.

| Hypothesis | Metric | Pass threshold |
| --- | --- | --- |
| H0 | Task accuracy, A vs C | A exceeds C by ≥ 10 percentage points |
| H1 | Task accuracy, A vs B | B falls ≥ 20 points below A |
| H2 | Report accuracy, ordinary vs captured trials (condition A) | Drop ≥ 15 points in captured trials |
| H2 | Error direction in captured trials | ≥ 70% of report errors point to the top-down intended location (chance ≈ 1 in 16) |
| H3 | Report accuracy drop, condition D | Drop < 5 points in captured trials |

**Definitions:**

- *Task accuracy:* share of episodes where the agent localises the target at the final step.
- *Report accuracy:* share of steps where the reported cell equals the cell with the highest actual attention weight.
- *Intended location:* the cell the controller's top-down query points to, before the bottom-up bias is added.

**Statistics:** mean and 95% bootstrap confidence interval per condition; paired permutation test between conditions, p < 0.01. All results are reported, including failed hypotheses.

**Stop rule:** if H0 or H1 fails, the setup does not replicate known results. Work stops and is debugged before H2 and H3 are run.

## Environment and tooling

Everything runs on CPU in Claude's cloud workspace; a full experiment should take a few hours, not days.

- **Language and libraries:** Python 3, NumPy, Matplotlib. PyTorch could not be installed (package downloads blocked), so training uses a small purpose-built autodiff engine in NumPy, checked against finite differences (worst relative error 1.8e-7).
- **Compute budget:** one training run ≤ 10 minutes; 4 conditions × 5 seeds + 4 sweep widths × 5 seeds = 40 runs.
- **Reproducibility:** fixed seeds, every configuration in a YAML file, results logged as JSON per run, plots generated from the logs only.
- **Code layout:**
    - `env.py` — task, target and distractor dynamics
    - `model.py` — attention, schema, controller, report heads
    - `train.py` / `eval.py` — training and the H0–H3 tests
    - `configs/`, `results/`, `report/`
- **Delivery:** the code and results as files to the product owner, plus a short results report. Publishing on GitLab or GitHub is the product owner's decision.

**Independent check:** the code is written by an AI. Before any public claim, a human with ML experience should review it, or at minimum rerun it from scratch.

## Iteration plan

Architecture is fixed first; then four short sprints, each ending at a gate the product owner signs off.

1. **Sprint 0 — Spec sign-off.** This document reviewed; thresholds and open decisions settled.
    - Gate: product owner approves the spec.
2. **Sprint 1 — Environment and baseline.** Task and distractors working; agent without schema learns the task.
    - Gate: baseline accuracy clearly above chance, learning curves shown.
3. **Sprint 2 — Schema and replication.** Schema and controller added; conditions A, B, C trained.
    - Gate: H0 and H1 pass. If not, stop and debug (stop rule).
4. **Sprint 3 — Self-report.** Report heads trained on ordinary trials; captured trials run; conditions A and D compared.
    - Gate: H2 and H3 evaluated, whatever the outcome.
5. **Sprint 4 — Robustness and write-up.** Capacity sweep, plots, a short results report in plain language.
    - Gate: product owner decides on publication and outreach.

## Limitations and open decisions

A pass shows that the mechanism behaves as AST predicts; it says nothing about whether anything is experienced. AST itself makes no stronger claim.

**Limitations:**

- *Toy scale.* A 16-cell field and a small network; results may not carry over to large models.
- *Divergence by construction.* If the schema simply cannot see the distractor, H2 is trivial. The schema therefore gets the same visual input as the controller, and the capacity sweep tests whether the effect depends on bottleneck size.
- *One task.* A single catch-and-localise task. A second task would be needed before generalising.
- *Author bias.* An AI designs and builds a test of a theory of awareness. Independent review of code and analysis is part of the plan, not an option.

**Link to MACTA:** a pass gives Awareness an operational test — a self-report that tracks a model of the system's own processing, with a predictable error profile. A fail is just as informative: it would show that this route to Awareness does not hold in the simplest case.

**Open decisions for the product owner:**

- [ ] Confirm or change the pass thresholds in the verification table
- [ ] Language of the final report: English or Swedish
- [ ] Whether results and code are published, and where
- [ ] Whether to contact Graziano's lab once results exist

## Sprint 1 results (7 Oct 2026)

Gate passed: the agent without a schema learns the task, and steering attention almost doubles its accuracy over uniform attention.

| Agent | Final pixel accuracy | Patch accuracy | Seeds |
| --- | --- | --- | --- |
| Steered attention | 66–68% | 77–79% | 3 |
| Uniform attention (control) | 39% | 69% | 1 |
| Chance | 1.6% | 6.3% | — |

A flash puts 95% of attention weight on the flash patch, dropping attention on the target from 43% to 3%. One step after the flash, attention is back on the target at 43%.

**Changes from the spec, found while building:**

- *Task:* the agent reports the exact pixel (64 options), not the patch. Reporting only the patch let the agent ignore attention and park it on half the field.
- *Attention:* driven by the agent's own belief about where the target is, plus a free query. Without this the network got stuck early.
- *Bounded control:* top-down attention logits are capped at ±4 and a flash adds 10. Uncapped, the agent learned to override every flash, so capture was not outside its control.
- *Periphery noise:* lowered from 0.35 to 0.20, so the target can be found at all.

**Risk for Sprint 2:** the controller sees the flash as an input, so it can infer capture without a schema. That may shrink the schema's benefit (H0, H1). Option: hide the flash input from the controller, so only the attention system feels it.

## Sprint 2 results (7 Oct 2026)

H0 and H1 fail their thresholds. The schema helps, and the effect is real (p < 0.001), but it is about a tenth of the size the spec required. By the stop rule, H2 and H3 are not run yet.

| Hypothesis | Measured difference | 95% CI | Threshold | Verdict |
| --- | --- | --- | --- | --- |
| H0: schema (A) vs no schema, same size (C) | +2.6 points | 1.5 to 3.8 | ≥ 10 | Fail (direction confirmed) |
| H1: schema on (A) vs ablated (B) | +1.8 points | 1.1 to 2.4 | ≥ 20 | Fail (direction confirmed) |

Final pixel accuracy over 5 seeds, 4000 identical test episodes each: A 69–71%, B 67–69%, C 66–69%. The flash was hidden from the controller, so only the schema could signal capture. C has 74 hidden units to match A's parameter count within 0.12%.

**Why the effect is small:**

- The schema works: at flash onset it puts 96% of its prediction on the flash patch, and it predicts the attention peak correctly 92% of the time.
- But the task rarely needs it. The controller always knows where it *meant* to attend, so the schema only adds news of a capture. Capture lasts 2 of 12 steps in half the trials, and the agent recovers quickly either way. The gain is largest when the flash ends just before the report (A 66% vs C 62%).

**Consequence for H2:** this schema sees the flash directly and models capture almost perfectly. A report built on it would not show the error AST predicts. The current design would very likely fail H2 for a trivial reason.

**Not an option:** lowering the thresholds after seeing the data. That would turn the test into a description.

**Proposed v2 (needs product-owner decision):** make the agent unable to know its own attention without a schema, as in Wilterson & Graziano. Attention gets an internal drift the controller does not control. The schema receives only a coarse, noisy trace of actual attention (4 quadrants), never the flash itself. A second control, C2, gives that same trace straight to the controller, to separate the effect of the schema from the effect of the extra information. New thresholds are fixed before any v2 run.

## v2 design and thresholds (fixed 7 Oct 2026, before any v2 run)

Approved by the product owner. Same task, agents and statistics as v1, with three changes:

- **Attention drift:** each patch's attention logit gets a slow random drift (AR(1), coefficient 0.8, stationary SD 2.5). The controller does not see it, so where attention actually lands differs from where the controller aimed.
- **Attention trace:** a coarse, noisy read-out of actual attention — its mass in each of the 4 quadrants, plus noise (SD 0.1).
- **Schema input:** the top-down query and the trace. The schema no longer sees the flash.

| Condition | Gets the trace | Schema |
| --- | --- | --- |
| A | Schema | Yes, width 16 |
| B | Schema | A with schema state replaced by its mean |
| C | No | No; widened to A's parameter count |
| C2 | Controller, directly | No; widened to A's parameter count |

| Hypothesis | Pass if |
| --- | --- |
| H0 | A beats C by ≥ 5 points (lower 95% CI bound), p < 0.01 |
| H0b | A beats C2, lower 95% CI bound > 0, p < 0.01 |
| H1 | B falls ≥ 5 points below A (lower 95% CI bound), p < 0.01 |

H0b is the new and harder test: does a dedicated model of attention help beyond simply having the raw signal? One pilot run of C only, to check that the drift really misaligns attention, is allowed before the main runs; no hypothesis is evaluated on it.

## v2 results (7 Oct 2026)

All three v2 hypotheses fail. In this task, an agent that models its own attention does not perform better than one that does not — not even when it gets the raw attention signal directly.

The pilot confirmed the drift works: actual attention peaked somewhere other than intended in 29% of steps (0% with drift off).

| Hypothesis | Measured difference | 95% CI | Threshold | Verdict |
| --- | --- | --- | --- | --- |
| H0: A vs C | +0.2 points | −1.4 to 1.6 | ≥ 5 | Fail |
| H0b: A vs C2 | +0.7 points | −1.4 to 2.8 | > 0 | Fail |
| H1: A vs B | +4.3 points | 3.6 to 5.1 | ≥ 5 | Fail (narrowly) |

Final pixel accuracy, 5 seeds: A 58–62%, B 54–57%, C 59–61%, C2 58–61%. C has 73 hidden units (A's parameter count ±0.04%); C2 has 72 (+1.0%).

**Reading:** the controller relies on its schema (ablating it costs 4.3 points), but an agent without one learns an equally good strategy. C2 shows that even the raw trace does not help, so the limit is the task, not the schema. Most likely the controller can infer where attention went from what the glimpse shows.

**Exploratory observation (not pre-registered, not evidence):** when attention drifted, the schema named the actual peak in 38% of those steps and the *intended* peak in 31%. Any single other patch would get about 7% by chance. That is the qualitative bias AST predicts — the model of attention is pulled toward where attention was aimed. It may partly follow from what the schema is given as input, which is why it needs its own pre-registered test.

**Status:** two pre-registered attempts at H0/H1 have failed. Further redesigns aimed at making H0 pass would risk searching until something works. Decision for the product owner below.
