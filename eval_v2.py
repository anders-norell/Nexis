"""v2 verdicts, with the thresholds fixed in the spec before any v2 run.

H0 : A beats C  by >= 5 points (lower 95% CI bound), p < 0.01
H0b: A beats C2, lower 95% CI bound > 0,             p < 0.01
H1 : B falls >= 5 points below A (lower 95% CI bound), p < 0.01
"""
import json
import numpy as np
from env import sample_batch
from diagnose import load
from eval_sprint2 import per_episode, hier_bootstrap, sign_flip_p

SEEDS = range(5)
N = 4000
TESTS = dict(H0=("C", 0.05), H0b=("C2", 0.0), H1=("B", 0.05))


def main():
    rows, diffs = [], {k: [] for k in TESTS}
    for s in SEEDS:
        b = sample_batch(N, np.random.default_rng(60_000 + s), p_captured=0.5)
        A = load(f"v2_A_s{s}")
        ok = {}
        ok["A"], oA = per_episode(A, b)
        A.ablate = True
        ok["B"], _ = per_episode(A, b)
        A.ablate = False
        ok["C"], _ = per_episode(load(f"v2_C_s{s}"), b)
        ok["C2"], _ = per_episode(load(f"v2_C2_s{s}"), b)
        mis = oA["att"].argmax(-1) != oA["intended"].argmax(-1)
        sch = oA["schema_pred"].argmax(-1) == oA["att"].argmax(-1)
        rows.append(dict(seed=s, **{k: float(v.mean()) for k, v in ok.items()},
                         schema_top1=float(sch.mean()),
                         schema_top1_when_drifted=float(sch[mis].mean()),
                         drifted_share=float(mis.mean())))
        for h, (other, _) in TESTS.items():
            diffs[h].append(ok["A"] - ok[other])
    res = dict(per_seed=rows)
    for h, (other, thr) in TESTS.items():
        lo, hi = hier_bootstrap(diffs[h])
        p = sign_flip_p(diffs[h])
        res[h] = dict(vs=other, mean_diff=float(np.mean([d.mean() for d in diffs[h]])),
                      ci95=[float(lo), float(hi)], p=p, threshold=thr,
                      passes=bool(lo > thr if thr == 0 else lo >= thr) and p < 0.01)
    json.dump(res, open("results/v2_verdict.json", "w"), indent=1)
    print("seed    A      B      C      C2    schema_top1  (when drifted)")
    for r in rows:
        print(f"{r['seed']:>3}  {r['A']:.3f}  {r['B']:.3f}  {r['C']:.3f}  {r['C2']:.3f}   "
              f"{r['schema_top1']:.3f}        {r['schema_top1_when_drifted']:.3f}")
    for h in TESTS:
        x = res[h]
        print(f"{h} (A vs {x['vs']}): {x['mean_diff']*100:+.1f} pts, 95% CI "
              f"[{x['ci95'][0]*100:.1f}, {x['ci95'][1]*100:.1f}], p={x['p']:.4f}, "
              f"threshold {x['threshold']*100:.0f} -> {'PASS' if x['passes'] else 'FAIL'}")


if __name__ == "__main__":
    main()
