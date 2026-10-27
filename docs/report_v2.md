<!-- Snapshot of the living document https://claude.ai/artifact/WgWqgRBXEqaChu3NMB76JF, 27 Oct 2026 (rev 60). If they differ, the living document applies. -->

# A learned attention-schema module adds no benefit over hand-coded mismatch features in small RL search agents

27 Oct 2026 · Anders Norell

> **Status:** working draft for informal external review, not peer reviewed. Project lead: Anders Norell (independent; not a trained researcher). Code, experiments, analysis and this text were produced with Claude (Anthropic) under his direction, with a separate Claude instance acting as independent reviewer at each gate. This draft was reviewed once by another Claude instance (19 comments, each checked against code and data before being addressed); no human with ML training has reviewed it yet. Revised 27 Oct 2026 after external comments: title and claims narrowed, task design limitation and the status of the H5 prediction added, comparison with Piefke et al. corrected. All thresholds were locked before results were seen unless marked post hoc. Code and raw results: github.com/anders-norell/Nexis (directory `v2/`). Plain-language glossary: https://claude.ai/artifact/2syuVpzf2GcAfaP9DgSgjC (English tab).

## Abstract

Attention Schema Theory (AST) holds that a system's knowledge of its own attention comes from a simplified model of attention, not from attention itself. We tested two consequences in small reinforcement-learning agents doing visual search with a hard spotlight that salient distractors sometimes capture.

With a correct, externally computed schema, agents found the target 6–7.5 percentage points (pp) more often than a no-schema baseline, and they followed a false schema: a false "already visited" mark cut visits to the marked cells by 0.81–0.84 of 3 commands (H2, pre-registered, passed). This shows the schema is used as an authority. It does not show that the schema is a simplification, since it was never wrong in training.

A learned schema module (supervised, with an information bottleneck) placed its errors after capture on other salient objects, not on the intended location (H5, pre-registered, two seeds, failed at both judged capacity levels; these modules located the spotlight at only 26–45 % of capture events). A module trained only through reward, given comparator inputs, gained about 2 pp over the baseline but failed two pre-registered prerequisites in one of three seeds, so H5 was not judged for it. A pre-registered follow-up fed the same comparator inputs (a per-cell mismatch map for the aimed window plus a one-bit alarm, both hand-coded functions of the observation) directly to the policy, with no module. It matched the module (difference +0.2 pp, 95 % CI about −0.8 to +1.2 pp). In this task, then, a learned module gave no detectable advantage over hand-coded mismatch features. This does not show that the agent lacks an internal representation of its attention: the recurrent policy itself may carry one, and we did not test that. The task also put little pressure on representing attention, because the agent chooses absolute grid locations and never needs to know where its spotlight currently is. We make no claim about consciousness or about AST in general. We also report five measurement artefacts that made "does the schema point at the intention" true or false by construction, and the calibration procedure that removed them.

## 1 Introduction

We asked whether an artificial agent's information about its own attention behaves like a simplified model of attention, as AST predicts, rather than like a readout of attention itself.

AST (Graziano) proposes that the brain controls attention with an attention schema, a simplified internal model of what attention is doing, and that reports of awareness are reports of that model. Prior computational work shows that a schema can help control. Wilterson and Graziano (2021) gave an RL agent a descriptive model of its attention and found better control of a visuospatial spotlight. Piefke et al. (2024) gave a PPO agent an extra resource decoupled from its attention window; the agent learned to use it as an emergent schema, which improved performance and made the window's location more decodable by separately trained decoders, most at intermediate noise. Liu et al. (2023) found that an attention-schema module improved team reward in cooperative multi-agent RL. Farrell, Ziman and Graziano (2024) found that a schema made networks better at judging, and easier to judge by, other networks, and improved cooperation; the improvement was specific to tasks involving other agents' attention. Hard spotlight attention trained with RL goes back to Mnih et al. (2014); the theory is summarised in Graziano and Webb (2015).

That a schema helps is therefore not new. We targeted two predictions that follow from the schema being a *model* rather than a copy:

1. **False schema (H2).** If the schema reports something false, the agent moves its real attention accordingly. This tests whether the schema is causally upstream of control.
2. **Intended versus actual (H5).** When a salient distractor captures the spotlight, a simplified schema that leans on the agent's own command should err towards the *intended* location more than an impartial observer with the same information would. We treated this as the more AST-specific prediction: errors that reveal the model rather than the process. It does not follow from AST alone, however; it assumes that the schema leans on the command more than on sensory evidence (section 6).

The work ran in six sprints, each closed by a gate where the project lead approved the result against a written specification before the next began. This report covers sprints 1–6 and one pre-registered follow-up ("step 0").

## 2 Environment and agents

The agent searches a grid for one target with a hard spotlight; a salient distractor can capture the spotlight so that it lands somewhere other than where the agent aimed. An illustrated episode, constructed to explain the task and the conditions rather than recorded from a run, is available at [Through the agent's eyes](https://claude.ai/artifact/YaJypo4vjvo7cFUKTaswsz#en).

### 2.1 Task

- **Grid and objects.** Grid n×n with n ∈ {6, 7, 8, 9}; one target and k ∈ [4, 16] distractors; 4 colours × 4 shapes. Weak cue: 75 % of distractors share the target's colour. The target lies in a "hot" quadrant with probability 0.4.
- **Observation.** A periphery map of the whole grid (cell valid, occupied, colour, and a noisy salience channel, noise SD 0.15) but not shape; a fovea showing the contents of the cells actually under the spotlight; and an efference copy of the previous command. The spotlight is w×w, w ∈ {1, 2}, fixed within an episode.
- **Capture.** Each episode has 0–3 salient distractors (salience ~ U(0.3, 1)). Each step the spotlight is captured by a salient object outside the intended window with probability p_capture × salience (total capped at 0.6). p_capture = 0.25 by default, 0.5 in the sprint 6 main run.
- **Episode.** Ends when the target is under the spotlight (reward +1) or the step budget ⌈U(0.6, 1.2)·(k+1)⌉ (minimum 3) runs out; each step costs −0.02.

### 2.2 Agent

A retinotopic priority map: one GRU unit per cell (hidden state 16, readout layer 32, weights shared across cells; 4 610 parameters in total for the baseline), cells exchanging information only through a global mean. The policy is a softmax over cells; a critic estimates value. The agent places fovea contents on the cells it *intended* to look at, using the efference copy (cf. Sperry 1950; von Holst and Mittelstaedt 1950), so a capture misplaces the glimpse in its internal map. Training: A2C (Mnih et al. 2016) with GAE (Schulman et al. 2016; γ 0.99, λ 0.95), Adam, learning rate 10⁻³, batch 256, value-loss weight 0.5, gradient-norm clipping at 1.0, entropy weight 0.01 → 0.001 over 700 000 episodes in sprint 4 and 300 000 elsewhere, 1 000 000 episodes per run (400 000 in pilots). PyTorch could not be installed, so backpropagation is hand-written in NumPy and every gradient is tested against finite differences.

### 2.3 Schema conditions

| Condition | What the schema channel carries | How it is formed |
| --- | --- | --- |
| Baseline | nothing | – |
| 1a fixed | per cell: the actual window last step and a "covered so far" map | computed from the true spotlight, always correct |
| Shuffled | 1a from another episode | control for extra input |
| 1b learned | one noisy number per cell from a schema module that sees all inputs plus a matching signal | a decoder that sees only the channel is trained to predict the actual window; information cost in bits per step |
| 1b-E | as 1b plus the alarm | as 1b |
| 1c emergent | same module architecture as 1b, no decoder, no supervised loss; without alarm it sees exactly the policy's inputs (no matching signal), with alarm it also gets the matching signal and the alarm, as 1b-E | only through the policy gradient flowing into the channel; bit cost as a penalty on the module's own loss |
| Capacity baseline | 1c module fed another episode's inputs from the start | control for extra parameters |
| Alarm direct (step 0) | no module; matching signal and alarm fed straight to the policy | – |

The **matching signal** marks, for each cell in the aimed window, whether the fovea's content disagrees with the periphery there, which signals a likely capture. The **alarm** is 1 if the matching signal shows a mismatch anywhere and 0 otherwise, broadcast to every cell: "something did not match where I aimed". Both are hand-coded, deterministic functions of inputs the policy already receives. It is a reafference comparator in von Holst and Mittelstaedt's sense, related to forward models in motor control (Wolpert et al. 1995). The channel is Gaussian; bits per step are the KL divergence from N(0, 1), summed over cells.

## 3 Pre-registration and statistics

Every judged result was compared against criteria written down and locked before the result was computed.

- **Gates.** Each sprint ended with a written plan, a run, an independent review by a separate Claude instance that had not built the code, and approval by the project lead. Pilots were evaluated blind to the direction of schema errors.
- **Locking.** Thresholds for sprint 4 were locked after the sprint 3 pilot. For H5 (sprint 5a) and the 1c main run (gate 6a), the evaluation scripts store SHA-256 hashes of all code they depend on and refuse to run if any file has changed.
- **Criterion.** A difference counts if it exceeds 2 standard errors in every seed, in the predicted direction. SE is computed over test episodes within a seed: unpaired for hit rates, paired for H2 (same episode with and without the intervention), bootstrap over episodes for H5.
- **Seeds and banks.** Three training seeds per condition in judged runs (10–12 for sprint 4, 30–32 for sprint 6 and step 0), except H5 in sprint 5a, which was judged on the two pilot seeds (appendix item 7). Pilots used two seeds. Test episodes come from a fixed bank: episode i has seed 2 000 000 + i, 4 000 episodes, with separate random streams for environment and agent choices. Pilot and probe banks use different offsets (4 000 000 and 7 000 000).
- **Checkpoints.** Sprint 4 used the checkpoint with the best validation hit rate; sprint 5a, sprint 6 and step 0 used the final checkpoint.
- **Power.** With 4 000 test episodes the SE of a hit-rate difference is about 0.9 pp near 80 % hits (sprint 6) and 0.6–0.7 pp near 90 % (sprint 4). At SE 0.9 pp a true 2 pp effect passes one seed with probability about 0.59 and all three about 0.20. The per-seed SE covers test-episode noise only, not variation between training seeds (Agarwal et al. 2021); we therefore also give between-seed means ± SE where it matters.

Deviations from the plan and post-hoc analyses are listed in the appendix.

## 4 Results

This section reports data only; interpretation follows in section 6. Values are given per seed; pp = percentage points.

### 4.1 Competence and pilot (sprints 2–3)

The baseline reached 89.0 / 89.9 / 88.5 % hits (seeds 0–2) against about 28 % for a random agent (locked gate: > 73 %, passed). In the pilot (seeds 0–3, 400 000 episodes) 1a added 7.2–8.2 pp. Zeroing the "covered so far" part of 1a's channel at test dropped 1a to about 80 %; zeroing the "window last step" part did not matter. Hiding fovea contents did not change the schema's benefit (H6: the schema helps more when the fovea reveals less; not supported). 1b without the alarm gave no gain.

### 4.2 Main experiment (sprint 4, seeds 10–12, p_capture 0.25)

| Measure | Seed 10 | Seed 11 | Seed 12 | Criterion |
| --- | --- | --- | --- | --- |
| Hit rate, Baseline | 87.7 % | 88.2 % | 88.2 % | – |
| Hit rate, 1a | 95.1 % | 94.4 % | 95.1 % | – |
| Hit rate, Shuffled | 88.7 % | 88.4 % | 88.5 % | – |
| F: 1a − Baseline | +7.5 pp (z 12.0) | +6.2 (9.9) | +6.9 (11.3) | passed |
| F: 1a − Shuffled | +6.4 pp (z 10.6) | +6.0 (9.6) | +6.6 (10.9) | passed |
| H1: search slope, 1a − Baseline (steps per distractor) | −0.021 (z −1.6) | −0.001 (−0.1) | −0.003 (−0.2) | not passed |
| H2 (a) false "unvisited": change in commands aimed at marked cells, of 3 | +0.22 (SE 0.011) | +0.15 (0.009) | +0.07 (0.007) | passed |
| H2 (b) false "visited": same | −0.81 (SE 0.023) | −0.83 (0.022) | −0.84 (0.022) | passed |
| H2 (b), Shuffled control | −0.07 | −0.21 | +0.07 | – |
| H3a: revisit rate, Baseline − 1a | +5.7 pp (z 10.2) | +5.4 (9.6) | +5.5 (9.5) | passed |
| H3b: acute lesion of 1a at test, hit change | −50.2 pp | −34.6 | −46.7 | passed |

In H2 the false mark was inserted before the fourth command and kept for the rest of the episode, on at most 3 target-coloured cells, never the target. 1a's H2 effect exceeded the Shuffled control's by more than 10 SE in both parts. The reviewer found that the Shuffled control for H2 (a) was nearly empty: it changed 0.76 cells on average against 6.1 for 1a, because the donor's map seldom differed on the agent's own covered cells. A supplementary analysis with a real lie in the Shuffled condition gave ±0.02–0.05, with 1a still more than 10 SE larger. Part of H2 (b) is mechanical: the false mark never covers the target, so it raises 1a's hit rate (95.1 → 98.2–98.3 %) and episodes end sooner. The effect survives as a per-command rate (−0.29 ± 0.008, seed 10) and in episodes with all three commands in both runs (−0.94 ± 0.04). H2 (a) depends on checkpoint choice: final checkpoints give +0.30 / +0.19 (seeds 11 / 12) against +0.15 / +0.07 at the best-validation checkpoints; F also holds on final checkpoints. Between seeds, F against Baseline is +6.9 ± 0.4 pp (mean ± SE).

### 4.3 H5 in a learned schema (sprint 5a, seeds 20–21, p_capture 0.25, 400 000 episodes)

The pre-registered prerequisite was that the module locates the spotlight correctly at ≥ 50 % of H5 events. 1b without the alarm reached 20–21 % (stopped; decision recorded). 1b-E without cost reached 57.2 / 57.4 %, against 94 % for an ideal observer with the same inputs. At the two judged capacity levels localisation was lower, about 44 % (22 bits) and 26 % (11 bits), below the 50 % bar that motivated the prerequisite.

An H5 event is a capture where the fovea rules out the intended location, the target was not found, and the schema is read one step later. The measure is the share of the schema's error mass that falls on the intended window, compared with a weakened ideal observer matched to the schema's accuracy (q = 0.62–0.71). AST predicts schema share > null share.

| Channel (bits per step) | Hit rate | Schema share on intended | Null-model share | Difference (z) | Judged |
| --- | --- | --- | --- | --- | --- |
| No cost (172–178) | 88.8 / 88.4 % | 0.16 / 0.24 | 0.59 / 0.59 | −0.44 (−64) / −0.35 (−45) | reported |
| 44 | 88.0 / 88.5 % | 0.13 / 0.22 | 0.59 / 0.61 | −0.46 (−71) / −0.38 (−49) | reported |
| 22 | 87.6 / 87.9 % | 0.16 / 0.17 | 0.56 / 0.57 | −0.40 (−52) / −0.41 (−51) | not passed |
| 11 | 87.8 / 87.3 % | 0.25 / 0.21 | 0.53 / 0.52 | −0.28 (−32) / −0.31 (−39) | not passed |
| 5.5 | 87.2 / 86.8 % | 0.31 / 0.34 | 0.48 / 0.48 | −0.17 (−22) / −0.13 (−17) | reported |

About 1 900 H5 events per run. Both pre-registered sensitivity analyses (argmax matching; excluding fallback events) gave the same verdict. As capacity fell, the schema's share on the intended window rose, but so did its mass outside all candidate windows (0.39 → 0.76). The Baseline hit rate in this setting was 86.7 %.

### 4.4 Emergent schema (sprint 6)

**Pilot (seeds 40–41, 400 000 episodes, p_capture 0.25 and, for one cost setting, 0.5).** No condition met prerequisite (a), a gain over Baseline > 2 SE in both seeds. Best with alarm: +1.2 / +1.9 pp (z 1.6 / 2.6). Without alarm: −1.6 to +0.7 pp. The pilot stopped by its rule. Bits after capture did not differ systematically from bits without capture (pilot −2.6 to +9.2 bits per step; main run −4.2 / +5.3 / +0.4).

**Post-hoc extension (seeds 42–43, 1 000 000 episodes, p_capture 0.5).** 1c with alarm: +2.3 / +2.8 pp over Baseline (z 2.6 / 3.1). Used only to decide whether to run a main experiment.

**Main run (pre-registered, seeds 30–32, p_capture 0.5, cost 0.0001 per bit).**

| Condition | Seed 30 | Seed 31 | Seed 32 | Bits per step |
| --- | --- | --- | --- | --- |
| Baseline | 79.0 % | 78.6 % | 78.9 % | – |
| 1c with alarm | 80.9 % | 81.3 % | 80.5 % | 20–28 |
| 1c without alarm | 78.2 % | 79.0 % | 77.9 % | 2–4 |
| Capacity baseline | 78.5 % | 78.3 % | 79.0 % | 0.5–2.4 |
| 1b-E (supervised) | 81.3 % | 81.2 % | 81.5 % | 139–149 |

Prerequisites (a) and (b) both failed in seed 32: +1.6 pp over Baseline (z 1.75) and +1.4 pp over the capacity baseline (z 1.56); (c), probe above chance, passed (z 18 / 41 / 29). H5 was therefore not judged. Post hoc: pooled over seeds the gain over Baseline is +2.0 pp (z ≈ 3.9 from test noise); between seeds +2.0 ± 0.3 pp (mean ± SE). Shuffling 1c's channel at test cost 17.8 / 5.7 / 14.7 pp, so the policy did read it. Descriptively, the locked H5 probe put *less* mass on the intended window than on either control window (−17 to −95 SE); the correlation between mass on intended and on actual was −0.2. The sign follows from the probe: it is trained only on capture steps, where the intended window is always a wrong answer, and synthetic channels that simply mark the intended location read −56 / −62 SE. The untrained shadow module gave −0.003 to −0.032 (z −3.6 to −15).

### 4.5 Step 0: the comparator inputs without a module (pre-registered, seeds 30–32, p_capture 0.5)

Rule, locked before training: if the policy with the matching signal and alarm fed directly is within 2 SE of 1c with alarm, or above it, in all seeds, the module added nothing beyond the comparator.

| | Seed 30 | Seed 31 | Seed 32 |
| --- | --- | --- | --- |
| Hit rate, alarm direct | 80.9 % | 81.2 % | 81.1 % |
| − 1c with alarm | 0.0 pp (z 0.0) | −0.05 (−0.06) | +0.6 (0.7) |
| − Baseline | +1.9 pp (z 2.1) | +2.6 (2.9) | +2.2 (2.5) |

The rule was met, but it is a non-inferiority rule whose margin (2 SE ≈ 1.8 pp) is almost the whole effect. Pooled over seeds, alarm direct minus 1c with alarm is +0.19 pp (95 % CI about −0.8 to +1.2 pp from test noise; between seeds +0.2 ± 0.2 pp). The module therefore added at most about 1 pp beyond the comparator inputs. The comparator inputs alone beat Baseline by more than 2 SE in all three seeds (between seeds +2.2 ± 0.2 pp).

## 5 Measurement lessons

Measures of "does the schema point at the intended location" were repeatedly true or false by construction; five such artefacts were caught before locking, each by synthetic channels or by the independent reviewer.

| Stage | Measure | Artefact | Found by |
| --- | --- | --- | --- |
| 5a | Null model weakened by an exponent on the observer's likelihoods | Assigns exactly zero to the intended window at H5 events, so any schema passes | Review |
| 5a | Null model concentrated on candidates, schema diffuse | Schema's diffuse mass lowers its share automatically; biased towards failure | Review |
| 6 | "Om" (does error mass favour intended) against a null model | Sign tracked the channel's accuracy, not its content: +4.7 to −10.7 SE for channels with no role information | Synthetic calibration |
| 6 | Decoy window, probe trained on all steps | Intended window is never salient, decoy always is; a channel coding only salience passed (+54 SE) | Review |
| 6 | Decoy window, probe trained on capture steps | Channels built from environment maps (damped salience, salience on unvisited cells, priority map) passed (+4 to +102 SE) | Review |

The locked 1c measure requires the intended window to beat both a decoy (another wrong salient candidate) and a matched window (not a candidate, not salient, same colour and visited status as the intended window), each by > 2 SE in all seeds, read by a role-blind probe from the channel one step after the command. None of 14 kinds of synthetic channel without intention information passed; AST-like channels passed (+16 to +22 SE).

Practices we would keep:

- **Calibrate before locking.** Run the measure on synthetic channels with and without the property of interest, including channels that only encode the environment (in the spirit of control tasks for probes; Hewitt and Liang 2019).
- **Separate steering from description.** The channel at the step of the command carries the command itself; read it one step later.
- **Dose-check information, not only parameters.** The capacity control had the same parameters as 1c but collapsed to under 2.5 bits per step against 20–28, so it controlled parameters rather than information. An earlier control (H2 (a), sprint 4) was nearly empty for a similar reason.
- **Blind pilots, hashed locks.** Two pilot gates stopped without anyone looking at where errors pointed; locked scripts refuse to run on changed code.
- **Plan for power.** Per-seed criteria with three seeds are conservative for effects near 2 pp.

## 6 Discussion

Our reading is that this task did not require a separate learned model of attention: a memory of where the agent had looked, plus hand-coded comparator features, covered what it needed. The experiment was a reasonable sandbox for search with unreliable attention, but not a test that can separate an attention schema from simpler mechanisms. Everything in this section is interpretation.

**A given schema is an authority, not shown to be a model.** H2 shows that the 1a channel is causally upstream of where the agent looks. But 1a was never wrong during training, so following it when it lies is what any well-trained reader of a reliable signal would do. H2 does not show that the schema is a simplification. The asymmetry between H2 (a) and (b) is consistent with the agent mostly using the schema to avoid covered cells; the pilot's partial lesion pointed the same way.

**The self-trained module added nothing beyond the comparator inputs.** Neither 1b nor 1c helped without the matching signal and alarm, and step 0 shows these inputs help as much without any module (difference at most about 1 pp). Two caveats limit what this means. First, the comparator is hand-coded: the matching signal and alarm are deterministic functions of inputs the policy already has, so the contrast is "hand-coded feature versus learned module", not "comparator versus self-model" in general. Second, 1c with and without alarm differ in both features, so we cannot say whether the per-cell map or the one-bit alarm carries the gain. What we can say is that a separate learned module added nothing measurable once the relevant mismatch was computed for the agent. Whether the recurrent policy itself represents where its attention went was not tested: equal performance does not tell us what the policy represents, so the results do not establish control without an attention model.

**The task put little pressure on modelling attention.** The policy chooses an absolute grid location for the next spotlight command, so it can issue that command without knowing where the spotlight currently is. Self-information is useful only for bookkeeping of what has been searched, which is consistent with the agent using only the "covered so far" part of 1a. A task where the current attentional state matters for the next decision, for instance relative attention movements or tracking a moving target, would test the role of a schema far more directly. Hit rate is also a weak diagnostic: an episode succeeds as soon as the spotlight covers the target, so substantial errors in self-localisation can go unnoticed.

**Where the gap might be.** A correct schema (1a) gave +11 pp at p_capture 0.5 in a single-seed 400 000-episode pilot, against about +2 pp for 1c and 1b-E at 1 M episodes. 1b-E had supervision and did not do better than 1c, which suggests the learning signal is not the main limit. One candidate limit is localising the spotlight after capture: in sprint 5a (p_capture 0.25), 1b-E was right at 57 % of capture events against 94 % for an ideal observer. The architecture, whose cells communicate only through a global mean, may make this hard. We did not test whether this explains the gap. Piefke et al. (2024) did obtain an emergent schema with RL; in their setup an agent trained without the extra resource did not reach their convergence criterion within the tested training budget, whereas our baseline solved 79–89 % of episodes with no schema. We changed task, architecture, memory and training together, so this contrast does not identify a cause. Our agent has recurrent memory and an efference copy of its previous command, and Piefke et al. themselves discuss memory as a factor that could reduce the need for the extra resource; their central result concerns how hard the attentional state is to infer, which is more specific than overall task difficulty.

**H5 errors pointed elsewhere.** In 1b the schema's errors went to other salient objects, the places captures actually go. That is an accurate generalisation, the opposite of a pull towards the intention. Since 1b was trained against the actual location and was not used for control, we read this as a consequence of the construction, not as evidence against AST. In 1c the strongly negative probe values are what any channel carrying the aimed location produces with a probe trained on capture steps. They show that the channel encodes the intended location, not that it corrects or confuses it. Finally, a bias towards the intended location is not a necessary consequence of a simplified model of attention; it depends on the model's inputs, its learning objective and how it weighs intention against sensory evidence. Neither the supervised module (below its localisation prerequisite at the judged levels) nor the reward-trained module (prerequisites failed) was a suitable test of it, so these results cannot distinguish "AST predicts the wrong thing" from "we did not obtain a schema to test".

**What would test AST's core.** A task where accurate inference of the agent's own attentional state is consequential: situations where the same mismatch signal requires different next actions depending on where attention actually went, combined with probes and interventions on the controller's internal state to test whether an attention representation causally supports those choices. Multi-agent settings, where agents must predict each other's attention (Farrell, Ziman and Graziano 2024), are another candidate. We have not run such a study.

## 7 Limitations

The negative results hold for this environment, this architecture, A2C and up to 1 million episodes; they do not show that a schema of the kind AST describes cannot emerge.

- **Small agents, one architecture.** 4 610 parameters shared across cells, with a global mean as the only exchange between cells. We did not vary size (planned hypothesis H4: the effect survives scaling) or architecture.
- **Easy task.** The baseline reaches 79–89 %, which leaves little room. This was flagged before sprint 3 and is the same trap as in our earlier v1 attempt.
- **Few seeds, low power.** Three seeds per condition; the per-seed criterion misses a true 2 pp effect about four times in five at sprint 6 noise levels, and its SE covers test-episode noise only.
- **Efference copy for all conditions.** This was chosen for biological realism, but it gives every agent its intention, which weakens H5 as a test.
- **Self-built and self-reviewed.** Code, analysis and reviews were produced by instances of the same model under one lead. The reviewer instances had not built the code, but they share its training and may share its blind spots. No human with ML training has checked the code.
- **Non-standard implementation.** Hand-written NumPy backpropagation, tested against finite differences but not against an established library.
- **Checkpoint policy differs between studies** (best validation in sprint 4, final elsewhere).
- **Hand-coded comparator.** The matching signal and alarm are computed for the agent from its own inputs; we did not test whether an agent can learn them.
- **Lesions show dependence only.** Acute lesions and test-time shuffling put agents outside their training distribution; the lesioned 1a fell 28–43 pp below the Baseline, which shows reliance on the channel, not what the channel represents.
- **Absolute action space.** The agent selects absolute grid locations, so it never needs to know where its spotlight currently is. This limits how much any schema can help and is probably the main design weakness for the question we asked.
- **The H5 prediction was not derived from AST.** We pre-registered an intention-biased error pattern as AST's signature, but did not show that the theory requires it.
- **No test of the policy's own representation.** We compared modules, not what the recurrent policy represents; probes and interventions on its hidden state would be needed to say whether it models its attention.

## 8 Reproducibility

All code, locked criteria, raw result files and reviews are in the repository under `v2/`. `v2/reproduce.py` rebuilds the main tables at three levels: `tables` (seconds, from the committed result files), `evaluate` (33 minutes on one core: re-runs every evaluation on the committed checkpoints and compares each number with the committed results) and `train` (roughly 10–15 hours on two cores: retrains every run from scratch with the arguments stored in its checkpoint, then evaluates). On 26 October 2026, `evaluate` reproduced all six result files exactly (4 809 values, no differences), and retraining one run from scratch (step 0, seed 30) gave bit-identical weights. The full retraining has not been run.

| What | Files |
| --- | --- |
| Environment, agent, training | `env.py`, `agent_rl.py`, `train_rl.py`, `schema_channel.py`, `schema_1c.py`, `train_1c.py`, `train_alarm_direct.py` |
| Gradient tests | `test_agent_rl.py`, `test_schema_channel.py`, `test_1c.py` |
| Test banks and H2 | `bank.py`, `eval_mvp.py` |
| H5 (1b) | `h5.py`, `h5_eval.py`, `h5_5a.py`, lock `results/h5_lock.json` |
| 1c probe and main run | `probe_1c.py`, `test_probe_1c.py` (calibration), `eval_main_1c.py`, lock `results/h5_1c_lock.json`, pre-registration `results/sprint6_main_prereg.json` |
| Step 0 | `eval_step0.py`, pre-registration `results/sprint6_step0_prereg.json` |
| Results | `results/sprint4_mvp.json`, `sprint5a_h5.json`, `sprint6_main.json`, `sprint6_step0.json` |
| Independent reviews | `results/*review*.md` |

Requirements: Python 3 and NumPy. One 1 M-episode run takes 23–70 minutes on one CPU core. Training and test banks are seeded. Determinism has been checked on one machine only; on other platforms float results may differ in the last digits.

## References

- Agarwal, R., Schwarzer, M., Castro, P. S., Courville, A., and Bellemare, M. G. (2021). Deep reinforcement learning at the edge of the statistical precipice. *NeurIPS*.
- Farrell, K. T., Ziman, K., and Graziano, M. S. A. (2024). [Testing components of the attention schema theory in artificial neural networks](https://arxiv.org/abs/2411.00983v3). arXiv:2411.00983.
- Graziano, M. S. A., and Webb, T. W. (2015). The attention schema theory: a mechanistic account of subjective awareness. *Frontiers in Psychology* 6, 500.
- Hewitt, J., and Liang, P. (2019). Designing and interpreting probes with control tasks. *EMNLP*.
- Liu, D., Bolotta, S., Zhu, H., Bengio, Y., and Dumas, G. (2023). [Attention schema in neural agents](https://ar5iv.labs.arxiv.org/html/2305.17375). arXiv:2305.17375.
- Mnih, V., Heess, N., Graves, A., and Kavukcuoglu, K. (2014). Recurrent models of visual attention. *NeurIPS*.
- Mnih, V., et al. (2016). Asynchronous methods for deep reinforcement learning. *ICML*.
- Piefke, L., Doerig, A., Kietzmann, T., and Thorat, S. (2024). [Computational characterization of the role of an attention schema in controlling visuospatial attention](https://arxiv.org/html/2402.01056v2). arXiv:2402.01056.
- Schulman, J., Moritz, P., Levine, S., Jordan, M., and Abbeel, P. (2016). High-dimensional continuous control using generalized advantage estimation. *ICLR*.
- Sperry, R. W. (1950). Neural basis of the spontaneous optokinetic response produced by visual inversion. *Journal of Comparative and Physiological Psychology* 43, 482–489.
- von Holst, E., and Mittelstaedt, H. (1950). Das Reafferenzprinzip. *Naturwissenschaften* 37, 464–476.
- Wilterson, A. I., and Graziano, M. S. A. (2021). [The attention schema theory in a neural network agent: controlling visuospatial attention using a descriptive model of attention](https://pmc.ncbi.nlm.nih.gov/articles/PMC8379943). *PNAS* 118(33), e2102421118.
- Wolpert, D. M., Ghahramani, Z., and Jordan, M. I. (1995). An internal model for sensorimotor integration. *Science* 269, 1880–1882.

References without links are cited from memory and should be checked before circulation.

## Appendix: deviations from plan and post-hoc analyses

All changes below were made before the affected result was computed, unless marked post hoc.

| # | Stage | Change | Why | Timing |
| --- | --- | --- | --- | --- |
| 1 | Sprint 3 | H1 slope computed only over episodes where the target was found | Slope over all episodes merely repeated the hit rate (reviewer) | Before lock |
| 2 | All | Criterion text corrected from "2 SD" to "2 SE" | Wording error; computations always used SE | Before lock |
| 3 | Sprint 4 | Supplementary H2 (a) analysis with a real lie in the Shuffled control | Locked control was nearly empty (0.76 vs 6.1 changed cells) | Post hoc |
| 4 | Sprint 5a | Alarm added to the 1b module (1b-E) | 1b failed the localisation prerequisite (20–21 % vs ≥ 50 %); direction of errors not inspected | After pilot, before H5 lock |
| 5 | Sprint 5a | Reference capacity set by a halving rule, averaged over seeds | Two pre-specified references disagreed by 2.4× (rule: > 1.5×) | Before H5 lock |
| 6 | Sprint 5a | H5 null model revised (mixing instead of exponent, glimpse replaced by capture prior, diffuse-share correction, fallback matching) | Review found the first versions biased by construction | Before H5 lock |
| 7 | Sprint 5a | Planned fresh-seed run (5b) cancelled; H5 run once on the pilot runs (seeds 20–21) | Decision to move on to 1c, with 1b as a reference | Before H5 computed |
| 8 | Sprint 6 | Main run at p_capture 0.5 with a fixed cost of 0.0001 per bit, instead of the pilot's capacity-level rule | Pilot failed prerequisite (a); post-hoc phase E (seeds 42–43) showed a gain at these settings | Post hoc choice of operating point; main run pre-registered on new seeds |
| 9 | Sprint 6 | Probe trained on capture steps only; null-model "om" replaced by decoy and matched-window controls | Calibration and two reviews showed the earlier forms passed or failed by construction | Before 1c lock |
| 10 | Sprint 6 | Main run interrupted by a 2-hour job limit and resumed from checkpoints | Infrastructure | – |
| 11 | After gate 6 | Step 0 (alarm direct) added | Reviewer suggested it to isolate the module's contribution | Pre-registered before training |
| 12 | Sprint 4 | Test bank evaluated on 8 runs before the ninth (Shuffled, seed 12) had finished training | Process slip; no code or argument changed afterwards (review) | During run |
| 13 | Sprint 6 | Pooled gain over seeds (+2.0 pp, z ≈ 3.9) and between-seed means reported | The locked criterion was per seed; pooling was not pre-registered | Post hoc |
| 14 | Step 0 | Confidence interval for alarm direct minus 1c reported beside the non-inferiority rule | The rule's margin is almost the whole effect (review) | Post hoc |
