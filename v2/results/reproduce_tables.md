# Main tables, rebuilt from result files

Source: `results/`

## Sprint 2: competence (seeds 0-2)

| Seed | Baseline hit rate | Random agent |
| --- | --- | --- |
| 0 | 89.0 % | 28.4 % |
| 1 | 89.9 % | 27.7 % |
| 2 | 88.5 % | 27.6 % |

## Sprint 4: main experiment (seeds 10-12, p_capture 0.25)

| Measure | Seed 10 | Seed 11 | Seed 12 |
| --- | --- | --- | --- |
| Hit rate, Baseline | 87.7 % | 88.2 % | 88.2 % |
| Hit rate, 1a | 95.1 % | 94.4 % | 95.1 % |
| Hit rate, Shuffled | 88.7 % | 88.4 % | 88.5 % |
| F: 1a - Baseline | +7.5 pp (z 12.0) | +6.2 pp (z 9.9) | +6.9 pp (z 11.3) |
| F: 1a - Shuffled | +6.4 pp (z 10.6) | +6.0 pp (z 9.6) | +6.6 pp (z 10.9) |
| H1: slope, 1a - Baseline | -0.021 (z -1.6) | -0.001 (z -0.1) | -0.003 (z -0.2) |
| H2 (a) false unvisited | +0.22 (SE 0.011) | +0.15 (SE 0.009) | +0.07 (SE 0.007) |
| H2 (a) false unvisited, Shuffled | +0.00 | +0.00 | -0.00 |
| H2 (b) false visited | -0.81 (SE 0.023) | -0.83 (SE 0.022) | -0.84 (SE 0.022) |
| H2 (b) false visited, Shuffled | -0.07 | -0.21 | +0.07 |
| H3a: revisits, Baseline - 1a | +5.7 pp (z 10.2) | +5.4 pp (z 9.6) | +5.5 pp (z 9.5) |
| H3b: acute lesion, hit change | -50.2 pp | -34.6 pp | -46.7 pp |

Verdicts: F True, H1 False, H2 (a) True, H2 (b) True, H3 True. Between seeds, F vs Baseline +6.9 +- 0.4 pp (mean +- SE).

## Sprint 5a: H5 in 1b-E (seeds 20, 21)

| Run | Bits per step | Hit rate | Schema share on intended | Null share | Difference (z) |
| --- | --- | --- | --- | --- | --- |
| p5_ehalf22.1_s20 | 22.6 | 87.6 % | 0.16 | 0.56 | -0.40 (-52) |
| p5_ehalf22.1_s21 | 22.2 | 87.9 % | 0.17 | 0.57 | -0.41 (-51) |
| p5_ehalf11.05_s20 | 11.1 | 87.8 % | 0.25 | 0.53 | -0.28 (-32) |
| p5_ehalf11.05_s21 | 11.1 | 87.3 % | 0.21 | 0.52 | -0.31 (-39) |
| p5_ehalf5.53_s20 | 5.3 | 87.2 % | 0.31 | 0.48 | -0.17 (-22) |
| p5_ehalf5.53_s21 | 5.8 | 86.8 % | 0.34 | 0.48 | -0.13 (-17) |
| p5_ehalf44.2_s20 | 44.8 | 88.0 % | 0.13 | 0.59 | -0.46 (-71) |
| p5_ehalf44.2_s21 | 44.5 | 88.5 % | 0.22 | 0.60 | -0.38 (-49) |
| p5_eref1e-4_s20 | 178.1 | 88.8 % | 0.15 | 0.59 | -0.44 (-65) |
| p5_eref1e-4_s21 | 172.1 | 88.4 % | 0.23 | 0.58 | -0.35 (-45) |

Verdicts per level: 22.1: ej klarad; 11.05: ej klarad; 5.53: redovisas (ej klarad); 44.2 (referens): redovisas (ej klarad); utan kostnad (0,0001): redovisas (ej klarad)

## Sprint 6: main run (seeds 30-32, p_capture 0.5)

| Condition | Seed 30 | Seed 31 | Seed 32 | Bits per step |
| --- | --- | --- | --- | --- |
| Baseline | 79.0 % | 78.6 % | 78.9 % | - |
| 1c with alarm | 80.9 % | 81.3 % | 80.5 % | 20.1-27.7 |
| 1c without alarm | 78.2 % | 79.0 % | 77.9 % | 2.1-3.8 |
| Capacity baseline | 78.5 % | 78.3 % | 79.0 % | 0.5-2.4 |
| 1b-E | 81.3 % | 81.2 % | 81.5 % | 138.9-148.5 |

Prerequisites for 1c with alarm, per seed:

- (a) vs Baseline: +1.9 pp (z 2.09, pass), +2.7 pp (z 2.99, pass), +1.6 pp (z 1.75, fail)
- (b) vs capacity baseline: +2.3 pp (z 2.59, pass), +2.9 pp (z 3.26, pass), +1.4 pp (z 1.56, fail)
- (c) probe above chance: z 18 (pass), z 41 (pass), z 29 (pass)
- Post hoc: pooled gain over Baseline +2.0 pp (z 3.9 from test noise); between seeds +2.0 +- 0.3 pp.

- H5 probe (descriptive), seed 30: intended - decoy -0.114 (z -25), intended - matched -0.049 (z -17); shadow module -0.004 / -0.003
- H5 probe (descriptive), seed 31: intended - decoy -0.350 (z -77), intended - matched -0.252 (z -95); shadow module -0.010 / -0.004
- H5 probe (descriptive), seed 32: intended - decoy -0.210 (z -39), intended - matched -0.124 (z -36); shadow module -0.032 / -0.020

## Step 0: comparator inputs to the policy, no module (seeds 30-32)

| | Seed 30 | Seed 31 | Seed 32 |
| --- | --- | --- | --- |
| Hit rate, alarm direct | 80.9 % | 81.2 % | 81.1 % |
| - 1c with alarm | +0.00 pp (z 0.00) | -0.05 pp (z -0.06) | +0.62 pp (z 0.71) |
| - Baseline | +1.9 pp (z 2.1) | +2.6 pp (z 2.9) | +2.2 pp (z 2.5) |

Pre-registered rule met (module added nothing): True. Post hoc: alarm direct - 1c, pooled +0.19 pp, 95 % CI [-0.8, +1.2] pp from test noise; between seeds +0.19 +- 0.22 pp. Alarm direct - Baseline between seeds +2.2 +- 0.2 pp.

