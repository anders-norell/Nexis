"""Sprint 3 pilot: search slope WITH capture (proposed H1 measure), 8 x 8, window 1, standard
salient distractors and capture, k = 4, 8, 12, 16, budget ceil(1.2 (k+1)), 2 000 episodes per k.
Slope = least-squares slope of mean steps over all episodes (found or not), and of hit rate.
Standard error of the slope by bootstrap over episodes (200 resamples)."""
import json, os, sys
import numpy as np
from env import EnvConfig, SearchEnv
from metrics import run_metrics
import pilot_eval as pe

KS = (4, 8, 12, 16)


def slope_with_capture(ag, seed, mode=None, n=2000):
    cfg0 = ag.cfg
    steps, found = {}, {}
    for k in KS:
        m = run_metrics(SearchEnv(cfg0, seed=710_000 + 10 * seed + k), pe.SchemaAdapter(ag, np.random.default_rng(k), mode),
                        n, overrides=dict(n=8, w=1, k=k, budget=int(np.ceil(1.2 * (k + 1)))))
        steps[k], found[k] = m["steps"].astype(float), m["found"].astype(float)
    ks = np.array(KS, float)
    fit = lambda d: float(np.polyfit(ks, [d[k].mean() for k in KS], 1)[0])
    rng = np.random.default_rng(0)
    boot = [fit({k: rng.choice(steps[k], len(steps[k])) for k in KS}) for _ in range(200)]
    return dict(slope_steps=fit(steps), slope_steps_se=float(np.std(boot)), slope_found=fit(found),
                steps=[float(steps[k].mean()) for k in KS], found=[float(found[k].mean()) for k in KS])


if __name__ == "__main__":
    out = {}
    for name in sys.argv[1:]:
        ag, args, _ = pe.load_run(os.path.join(pe.RES, name + ".best.npz"))
        out[name] = slope_with_capture(ag, args["seed"])
        if args["schema"] != "off":
            out[name + "|lesion"] = slope_with_capture(ag, args["seed"], mode="off")
        print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[name].items()}, flush=True)
    p = os.path.join(pe.RES, "sprint3_slope_capture.json")
    old = json.load(open(p)) if os.path.exists(p) else {}
    old.update(out)
    json.dump(old, open(p, "w"), indent=1)
