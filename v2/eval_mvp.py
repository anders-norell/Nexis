"""Sprint 4 (MVP): test the nine runs on the seed bank and judge them against the criteria locked
at gate 3 (spec, "Förslag: trösklar och körplan för MVP", "H2 i detalj", "Fröbank", H3a/H3b).

Runs: results/s4_{off,fixed,shuffled}_s{seed}.best.npz, training seeds 10, 11, 12 (a run redone
under the pre-registered rule, < 73 % hits, has seed + 1000; RUNS below says which run is used).

Significance (locked): the difference exceeds 2 standard errors and goes the same way in all three
seeds. SE of a difference between two conditions = sqrt(SE1^2 + SE2^2), each SE over the 4 000 bank
episodes within the seed (unpaired, as in the pilot and gate 2; chosen before any result was seen).
H2 is paired by definition: per episode, with intervention minus without, SE = SD / sqrt(n).
Revisit share and H1 slope: SE by bootstrap over episodes (1 000 resamples, fixed bootstrap seed).

  python3 eval_mvp.py            writes results/sprint4_mvp.json and prints the verdicts
"""
import json
import os
import sys
import numpy as np
from env import EnvConfig
from agent_rl import make_agent
from bank import run_bank, bank_seeds, h1_seeds, h1_overrides, H1_KS

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
SEEDS = (10, 11, 12)
CONDS = ("off", "fixed", "shuffled")      # Baslinje, Fast schema (1a), Blandat schema
NBOOT = 1000
RERUN_OFFSET = 1000
K2 = 0.73


def run_name(cond, seed):
    return f"s4_{cond}_s{seed}"


def load(name):
    path = os.path.join(RES, name + ".best.npz")
    z = np.load(path)
    args = json.loads(str(z["m_args"]))
    cfg = EnvConfig(reveal=args["reveal"], p_capture=args["p_capture"],
                    windows=tuple(int(x) for x in str(args["windows"]).split(",")))
    ag = make_agent(cfg, d=args["d"], dh=args["dh"], seed=args["seed"], schema_mode=args["schema"])
    meta = ag.load(path)
    return ag, {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in meta.items() if k != "args"}


def boot_se(fn, arrays, seed=0):
    rng = np.random.default_rng(seed)
    n = len(arrays[0])
    vals = []
    for _ in range(NBOOT):
        i = rng.integers(0, n, n)
        vals.append(fn(*[a[i] for a in arrays]))
    return float(np.std(vals))


def revisit_share(rev, steps):
    return float(rev.sum() / max(steps.sum(), 1))


def basic(r):
    f = r["found"].astype(float)
    return dict(found=float(f.mean()), found_se=float(f.std() / np.sqrt(len(f))),
                revisit=revisit_share(r["revisits"], r["steps"]),
                revisit_se=boot_se(revisit_share, [r["revisits"], r["steps"]]),
                mean_steps=float(r["steps"].mean()), n=int(len(f)))


def h1_profile(ag, mode=None):
    steps = {}
    for k in H1_KS:
        r = run_bank(ag, h1_seeds(k), mode=mode, overrides=h1_overrides(k))
        steps[k] = (r["steps"].astype(float), r["found"])
    ks = np.array(H1_KS, float)

    def slope(*arrs):
        means = [arrs[2 * j][arrs[2 * j + 1]].mean() for j in range(len(H1_KS))]
        return float(np.polyfit(ks, means, 1)[0])
    flat = [x for k in H1_KS for x in steps[k]]
    rng = np.random.default_rng(1)
    boot = []
    for _ in range(NBOOT):
        res = []
        for k in H1_KS:
            s, f = steps[k]
            i = rng.integers(0, len(s), len(s))
            res += [s[i], f[i]]
        boot.append(slope(*res))
    return dict(slope=slope(*flat), slope_se=float(np.std(boot)),
                steps_found=[float(steps[k][0][steps[k][1]].mean()) for k in H1_KS],
                found=[float(steps[k][1].mean()) for k in H1_KS])


def h2_effect(base, lie, kind):
    e = base["elig_" + kind]
    d = (lie["aim_" + kind][e] - base["aim_" + kind][e]).astype(float)
    n = len(d)
    return dict(n=int(n), with_=float(lie["aim_" + kind][e].mean()), without=float(base["aim_" + kind][e].mean()),
                mean=float(d.mean()) if n else float("nan"),
                se=float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan"))


def evaluate_run(name):
    ag, meta = load(name)
    seeds = bank_seeds()
    out = dict(checkpoint=meta, schema=ag.schema_mode)
    base = run_bank(ag, seeds)
    out["intact"] = basic(base)
    out["h1"] = h1_profile(ag)
    if ag.schema_mode != "off":
        out["lesion"] = basic(run_bank(ag, seeds, mode="off"))
        out["h2"] = {}
        for kind in ("a", "b", "c"):
            lie = run_bank(ag, seeds, intervention=kind)
            out["h2"][kind] = h2_effect(base, lie, kind)
            out["h2"][kind]["found_with"] = float(lie["found"].mean())
    return out


def diff(x, y, sx, sy):
    d = x - y
    se = float(np.hypot(sx, sy))
    return dict(diff=float(d), se=se, z=float(d / se) if se > 0 else float("nan"))


def judge(runs, used):
    """Apply the locked criteria. used[(cond, seed)] = run name used for that cell."""
    R = {(c, s): runs[used[(c, s)]] for c in CONDS for s in SEEDS}
    v = {}

    def all_seeds(rows, sign=+1):
        ok = all(sign * r["diff"] > 2 * r["se"] for r in rows)
        return ok

    # F: 1a hits more often than Baslinje and than Blandat
    rows_b = [diff(R["fixed", s]["intact"]["found"], R["off", s]["intact"]["found"],
                   R["fixed", s]["intact"]["found_se"], R["off", s]["intact"]["found_se"]) for s in SEEDS]
    rows_s = [diff(R["fixed", s]["intact"]["found"], R["shuffled", s]["intact"]["found"],
                   R["fixed", s]["intact"]["found_se"], R["shuffled", s]["intact"]["found_se"]) for s in SEEDS]
    v["F"] = dict(vs_baseline=rows_b, vs_blandat=rows_s, passed=all_seeds(rows_b) and all_seeds(rows_s))
    # H1: 1a flatter slope (steps in found episodes, with capture) than both
    rows_b = [diff(R["fixed", s]["h1"]["slope"], R["off", s]["h1"]["slope"],
                   R["fixed", s]["h1"]["slope_se"], R["off", s]["h1"]["slope_se"]) for s in SEEDS]
    rows_s = [diff(R["fixed", s]["h1"]["slope"], R["shuffled", s]["h1"]["slope"],
                   R["fixed", s]["h1"]["slope_se"], R["shuffled", s]["h1"]["slope_se"]) for s in SEEDS]
    v["H1"] = dict(vs_baseline=rows_b, vs_blandat=rows_s, passed=all_seeds(rows_b, -1) and all_seeds(rows_s, -1))
    # H2: (a) more aims at the falsely unvisited cells, (b) fewer at the falsely visited ones, > 2 SE
    # in all three 1a seeds; and 1a's effect larger than Blandat's by > 2 SE.
    h2 = {}
    for kind, sign in (("a", +1), ("b", -1)):
        own = [dict(diff=R["fixed", s]["h2"][kind]["mean"], se=R["fixed", s]["h2"][kind]["se"],
                    n=R["fixed", s]["h2"][kind]["n"]) for s in SEEDS]
        ctl = [dict(diff=R["shuffled", s]["h2"][kind]["mean"], se=R["shuffled", s]["h2"][kind]["se"],
                    n=R["shuffled", s]["h2"][kind]["n"]) for s in SEEDS]
        vs = [diff(R["fixed", s]["h2"][kind]["mean"], R["shuffled", s]["h2"][kind]["mean"],
                   R["fixed", s]["h2"][kind]["se"], R["shuffled", s]["h2"][kind]["se"]) for s in SEEDS]
        h2[kind] = dict(fixed=own, blandat=ctl, fixed_vs_blandat=vs,
                        effect_passed=all_seeds(own, sign), vs_blandat_passed=all_seeds(vs, sign))
    h2["c_reported"] = dict(fixed=[R["fixed", s]["h2"]["c"] for s in SEEDS],
                            blandat=[R["shuffled", s]["h2"]["c"] for s in SEEDS])
    h2["passed"] = all(h2[k]["effect_passed"] and h2[k]["vs_blandat_passed"] for k in ("a", "b"))
    v["H2"] = h2
    # H3a: Baslinje (schema zeroed through training) hits less often and revisits more than 1a
    hit = [diff(R["off", s]["intact"]["found"], R["fixed", s]["intact"]["found"],
                R["off", s]["intact"]["found_se"], R["fixed", s]["intact"]["found_se"]) for s in SEEDS]
    rev = [diff(R["off", s]["intact"]["revisit"], R["fixed", s]["intact"]["revisit"],
                R["off", s]["intact"]["revisit_se"], R["fixed", s]["intact"]["revisit_se"]) for s in SEEDS]
    v["H3a"] = dict(hits=hit, revisits=rev, passed=all_seeds(hit, -1) and all_seeds(rev, +1))
    # H3b: 1a with the schema zeroed at test vs intact 1a
    hit = [diff(R["fixed", s]["lesion"]["found"], R["fixed", s]["intact"]["found"],
                R["fixed", s]["lesion"]["found_se"], R["fixed", s]["intact"]["found_se"]) for s in SEEDS]
    rev = [diff(R["fixed", s]["lesion"]["revisit"], R["fixed", s]["intact"]["revisit"],
                R["fixed", s]["lesion"]["revisit_se"], R["fixed", s]["intact"]["revisit_se"]) for s in SEEDS]
    v["H3b"] = dict(hits=hit, revisits=rev, passed=all_seeds(hit, -1) and all_seeds(rev, +1))
    v["H3"] = dict(passed=v["H3a"]["passed"] and v["H3b"]["passed"])
    # reported, not a criterion: acutely lesioned 1a vs Baslinje
    v["lesion_vs_baseline"] = [diff(R["fixed", s]["lesion"]["found"], R["off", s]["intact"]["found"],
                                    R["fixed", s]["lesion"]["found_se"], R["off", s]["intact"]["found_se"])
                               for s in SEEDS]
    return v


def main():
    out_path = os.path.join(RES, "sprint4_mvp.json")
    res = json.load(open(out_path)) if os.path.exists(out_path) else {"runs": {}}
    names = sys.argv[1:] or [run_name(c, s + o) for c in CONDS for s in SEEDS for o in (0, RERUN_OFFSET)]
    for name in names:
        if name in res["runs"] or not os.path.exists(os.path.join(RES, name + ".final.npz")):
            continue
        r = evaluate_run(name)
        res["runs"][name] = r
        json.dump(res, open(out_path, "w"), indent=1)
        print(name, f"found {r['intact']['found']:.4f}", flush=True)
    # which run is used per cell: the original, unless it is below K2 and a rerun exists
    used, reruns = {}, []
    for c in CONDS:
        for s in SEEDS:
            o, r = run_name(c, s), run_name(c, s + RERUN_OFFSET)
            if o in res["runs"] and res["runs"][o]["intact"]["found"] < K2:
                reruns.append(o)
                if r in res["runs"]:
                    used[(c, s)] = r
            elif o in res["runs"]:
                used[(c, s)] = o
    res["below_k2"] = reruns
    res["used"] = {f"{c}|{s}": n for (c, s), n in used.items()}
    if len(used) == len(CONDS) * len(SEEDS):
        res["verdict"] = judge(res["runs"], used)
        for k in ("F", "H1", "H2", "H3a", "H3b", "H3"):
            print(k, "passed" if res["verdict"][k]["passed"] else "NOT passed")
    json.dump(res, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main()
