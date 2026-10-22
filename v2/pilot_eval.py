"""Sprint 3 pilot evaluation.

For every finished pilot run (results/p3_*.best.npz) the checkpoint chosen on validation
episodes is tested on 4 000 new episodes (seed 900000 + training seed), in the environment the
run was trained in, with the agent's stochastic policy:

  intact     the agent as trained
  lesion     schema conditions only: the schema channel is zeroed at test (agent unchanged)

Per working point it reports the gain of Fixed schema (1a) over the baseline and the lesion
cost (1a intact minus 1a lesioned), with standard errors. Episodes are paired: the same test
seed gives the same episodes in every condition at a working point.

  python3 pilot_eval.py            evaluate all finished runs, write results/sprint3_pilot.json
"""
import glob
import json
import os
import sys
import numpy as np
from env import EnvConfig, SearchEnv
from agent_rl import PriorityMapAgent
from train_rl import rollout

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
N_TEST = 4000


def load_run(path):
    z = np.load(path)
    args = json.loads(str(z["m_args"]))
    cfg = EnvConfig(reveal=args["reveal"], p_capture=args["p_capture"],
                    windows=tuple(int(x) for x in str(args["windows"]).split(",")))
    ag = PriorityMapAgent(cfg, d=args["d"], dh=args["dh"], seed=args["seed"], schema_mode=args["schema"])
    meta = ag.load(path)
    return ag, args, {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in meta.items() if k != "args"}


def test(ag, seed, mode=None, n=N_TEST, B=500):
    saved = ag.schema_mode
    if mode is not None:
        ag.schema_mode = mode
    env = SearchEnv(ag.cfg, seed=900_000 + seed)
    rng = np.random.default_rng(900_000 + seed)
    found, steps, caps = [], [], []
    for _ in range(n // B):
        _, st = rollout(ag, env, B, rng, keep=False)
        found.append(st["found"]); steps.append(st["steps"])
    ag.schema_mode = saved
    f = np.concatenate(found)
    return dict(found=float(f.mean()), found_se=float(f.std() / np.sqrt(len(f))),
                mean_steps=float(np.concatenate(steps).mean()))


def point_key(args):
    return f"reveal={args['reveal']}|p_capture={args['p_capture']}|windows={args['windows']}"


def main():
    out_path = os.path.join(RES, "sprint3_pilot.json")
    res = json.load(open(out_path)) if os.path.exists(out_path) else {"runs": {}}
    for path in sorted(glob.glob(os.path.join(RES, "p3_*.best.npz"))):
        name = os.path.basename(path)[:-len(".best.npz")]
        if not os.path.exists(os.path.join(RES, name + ".final.npz")):
            continue                                     # still training
        if name in res["runs"]:
            continue
        ag, args, meta = load_run(path)
        r = dict(point=point_key(args), schema=args["schema"], seed=args["seed"], checkpoint=meta,
                 intact=test(ag, args["seed"]))
        if args["schema"] != "off":
            r["lesion"] = test(ag, args["seed"], mode="off")
        res["runs"][name] = r
        print(name, r["point"], r["schema"], f"intact {r['intact']['found']:.3f}",
              f"lesion {r['lesion']['found']:.3f}" if "lesion" in r else "", flush=True)
        json.dump(res, open(out_path, "w"), indent=1)
    # per working point and seed: gain and lesion cost
    pts = {}
    for name, r in res["runs"].items():
        pts.setdefault((r["point"], r["seed"]), {})[r["schema"]] = r
    table = []
    for (pt, seed), d in sorted(pts.items()):
        row = dict(point=pt, seed=seed)
        if "off" in d:
            row["base"] = d["off"]["intact"]["found"]
        for cond in ("fixed", "shuffled", "learned"):
            if cond in d:
                row[cond] = d[cond]["intact"]["found"]
                row[cond + "_lesion"] = d[cond]["lesion"]["found"]
        if "off" in d and "fixed" in d:
            se = np.hypot(d["off"]["intact"]["found_se"], d["fixed"]["intact"]["found_se"])
            row["gain_1a"] = row["fixed"] - row["base"]
            row["gain_1a_se"] = float(se)
            row["lesion_cost_1a"] = row["fixed"] - row["fixed_lesion"]
        table.append(row)
    res["table"] = table
    json.dump(res, open(out_path, "w"), indent=1)
    for row in table:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})


if __name__ == "__main__":
    main()
