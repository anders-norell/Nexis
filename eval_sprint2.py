"""Sprint 2 verdicts for H0 and H1.

All conditions are tested on the SAME 4000 held-out episodes per seed, so
differences are paired episode by episode.

H0: A (schema) beats C (no schema, parameter-matched) by >= 10 points.
H1: B (A with schema ablated) falls >= 20 points below A.
A hypothesis passes only if the lower end of the 95% CI of the difference
reaches the threshold (spec: "met with 95% confidence"), and p < 0.01.
"""
import json
import numpy as np
from env import sample_batch
from diagnose import load

SEEDS = range(5)
N = 4000
THRESH = dict(H0=0.10, H1=0.20)


def per_episode(agent, batch):
    o = agent.forward(batch)
    return (o["guess"][:, -1] == batch["target_pixel"][:, -1]).astype(float), o


def hier_bootstrap(diffs, n_boot=4000, rng=None):
    """diffs: list of per-episode difference arrays, one per seed.
    Resample seeds, then episodes within each seed."""
    rng = rng or np.random.default_rng(0)
    k = len(diffs)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        seeds = rng.integers(0, k, k)
        stats[i] = np.mean([diffs[s][rng.integers(0, len(diffs[s]), len(diffs[s]))].mean() for s in seeds])
    return np.percentile(stats, [2.5, 97.5])


def sign_flip_p(diffs, n_perm=4000, rng=None):
    """Paired permutation test: randomly swap condition labels per episode."""
    rng = rng or np.random.default_rng(1)
    d = np.concatenate(diffs)
    obs = d.mean()
    null = np.array([(d * rng.choice([-1, 1], d.size)).mean() for _ in range(n_perm)])
    return float((np.abs(null) >= abs(obs)).mean() + 1 / n_perm)


def main():
    rows, dH0, dH1, cap_rows = [], [], [], []
    for s in SEEDS:
        b = sample_batch(N, np.random.default_rng(50_000 + s), p_captured=0.5)
        A = load(f"s2_A_s{s}")
        C = load(f"s2_C_s{s}")
        okA, oA = per_episode(A, b)
        A.ablate = True
        okB, _ = per_episode(A, b)
        A.ablate = False
        okC, _ = per_episode(C, b)
        cap = b["captured"]
        rows.append(dict(seed=s, A=okA.mean(), B=okB.mean(), C=okC.mean(),
                         A_cap=okA[cap].mean(), C_cap=okC[cap].mean(),
                         A_ord=okA[~cap].mean(), C_ord=okC[~cap].mean(),
                         schema_top1=float((oA["schema_pred"].argmax(-1) == oA["att"].argmax(-1)).mean())))
        dH0.append(okA - okC)
        dH1.append(okA - okB)
    res = dict(per_seed=rows)
    for name, d in (("H0", dH0), ("H1", dH1)):
        lo, hi = hier_bootstrap(d)
        p = sign_flip_p(d)
        mean = float(np.mean([x.mean() for x in d]))
        res[name] = dict(mean_diff=mean, ci95=[float(lo), float(hi)], p=p,
                         threshold=THRESH[name],
                         passes=bool(lo >= THRESH[name] and p < 0.01),
                         direction_ok=bool(lo > 0 and p < 0.01))
    json.dump(res, open("results/sprint2_verdict.json", "w"), indent=1, default=float)
    print("seed    A      B      C     A_cap  C_cap  schema_top1")
    for r in rows:
        print(f"{r['seed']:>3}  {r['A']:.3f}  {r['B']:.3f}  {r['C']:.3f}  {r['A_cap']:.3f}  {r['C_cap']:.3f}  {r['schema_top1']:.3f}")
    for h in ("H0", "H1"):
        x = res[h]
        print(f"{h}: diff {x['mean_diff']*100:.1f} pts, 95% CI [{x['ci95'][0]*100:.1f}, {x['ci95'][1]*100:.1f}], "
              f"p={x['p']:.4f}, threshold {x['threshold']*100:.0f} -> {'PASS' if x['passes'] else 'FAIL'}"
              f" (direction {'confirmed' if x['direction_ok'] else 'not confirmed'})")


if __name__ == "__main__":
    main()
