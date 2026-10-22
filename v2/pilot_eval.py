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
from agent_rl import PriorityMapAgent, make_agent
from train_rl import rollout

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
N_TEST = 4000


def load_run(path):
    z = np.load(path)
    args = json.loads(str(z["m_args"]))
    cfg = EnvConfig(reveal=args["reveal"], p_capture=args["p_capture"],
                    windows=tuple(int(x) for x in str(args["windows"]).split(",")))
    ag = make_agent(cfg, d=args["d"], dh=args["dh"], seed=args["seed"], schema_mode=args["schema"])
    meta = ag.load(path)
    return ag, args, {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in meta.items() if k != "args"}


def test(ag, seed, mode=None, n=N_TEST, B=500, zero_channel=None):
    """zero_channel: 0 zeroes the 'window last step' channel, 1 the 'covered so far' map
    (partial lesions, to see which part of the schema the agent relies on)."""
    import train_rl
    import agent_rl
    saved, saved_fn = ag.schema_mode, train_rl.schema_input
    if mode is not None:
        ag.schema_mode = mode
    if zero_channel is not None:
        def part(*a, **k):
            s, v = agent_rl.schema_input(*a, **k)
            if s is not None:
                s = s.copy(); s[..., zero_channel] = 0.0
            return s, v
        train_rl.schema_input = part
    try:
        return _test(ag, seed, n, B)
    finally:
        ag.schema_mode, train_rl.schema_input = saved, saved_fn


def _test(ag, seed, n, B):
    env = SearchEnv(ag.cfg, seed=900_000 + seed)
    rng = np.random.default_rng(900_000 + seed)
    found, steps, caps = [], [], []
    for _ in range(n // B):
        _, st = rollout(ag, env, B, rng, keep=False)
        found.append(st["found"]); steps.append(st["steps"])
    f = np.concatenate(found)
    return dict(found=float(f.mean()), found_se=float(f.std() / np.sqrt(len(f))),
                mean_steps=float(np.concatenate(steps).mean()))


class SchemaAdapter:
    """Drives a trained agent (any schema mode) through metrics.run_metrics."""

    def __init__(self, agent, rng, mode=None):
        self.ag, self.rng, self.mode = agent, rng, mode or agent.schema_mode

    def reset(self, env):
        G = env.cfg.G
        self.h = self.ag.init_state(env.B)
        mod = getattr(self.ag, "module", None)
        self.hm = mod.init_state(env.B) if mod is not None else None
        self.union = np.zeros((env.B, G, G))
        self.t, self.schema, self.visited = 0, None, np.zeros((env.B, G, G))
        self.donor = self.rng.permutation(env.B)

    def act(self, env, obs):
        mod = getattr(self.ag, "module", None)
        if mod is not None:
            G = env.cfg.G
            xm, mm = mod.features(env, obs, self.t, None)
            self.hm, _, _, slm, _ = mod.step(xm, mm, self.hm)
            cur = np.exp(slm).reshape(env.B, G, G) if self.t > 0 else np.zeros((env.B, G, G))
            self.union = np.maximum(self.union, cur)
            if self.mode == "module":
                self.schema = np.stack([cur, self.union], -1)
        x, m = self.ag.features(env, obs, self.t, self.schema)
        self.h, logp, V, slogp, _ = self.ag.step(x, m, self.h)
        self.pred = np.exp(slogp)
        self.t += 1
        return (logp - np.log(-np.log(self.rng.random(logp.shape)))).argmax(1)

    def observe(self, env, info):
        from agent_rl import schema_input
        if self.mode != "module":
            self.schema, self.visited = schema_input(self.mode, env, info, self.visited, self.pred, self.donor)


def search_profile(ag, seed, mode=None):
    """H1/H3 measures in the standard environment: revisit share and hit rate (4 000 episodes),
    and the search slope on 8 x 8, window 1, no capture, k = 4..16 (as in sprint 2)."""
    from metrics import run_metrics, summarise
    cfg0 = ag.cfg
    m = run_metrics(SearchEnv(cfg0, seed=900_000 + seed), SchemaAdapter(ag, np.random.default_rng(seed), mode), N_TEST)
    s = summarise(m)
    steps = {}
    ag.cfg = EnvConfig(n_salient=(0,), reveal=cfg0.reveal, p_capture=cfg0.p_capture, windows=cfg0.windows)
    for k in (4, 8, 12, 16):
        mk = run_metrics(SearchEnv(ag.cfg, seed=700_000 + 10 * seed + k), SchemaAdapter(ag, np.random.default_rng(k), mode),
                         2000, overrides=dict(n=8, w=1, k=k, budget=int(np.ceil(1.2 * (k + 1)))))
        steps[k] = float(mk["steps"][mk["found"]].mean())
    ag.cfg = cfg0
    ks = np.array(sorted(steps))
    slope = float(np.polyfit(ks, [steps[k] for k in ks], 1)[0])
    return dict(found=s["found"], revisit_share=s["revisit_share"], aim_repeat_share=s["aim_repeat_share"],
                slope=slope, steps_per_k=steps)


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
        if r["point"] == "reveal=1.0|p_capture=0.25|windows=1,2":   # chosen working point: spread of H1/H3 measures
            r["profile"] = search_profile(ag, args["seed"])
            cfg0 = ag.cfg                               # same agent, environment without capture
            ag.cfg = EnvConfig(n_salient=(0,), reveal=cfg0.reveal, p_capture=cfg0.p_capture, windows=cfg0.windows)
            r["no_capture"] = test(ag, args["seed"])
            ag.cfg = cfg0
            if args["schema"] in ("fixed", "shuffled"):
                r["lesion_window"] = test(ag, args["seed"], zero_channel=0)
                r["lesion_covered"] = test(ag, args["seed"], zero_channel=1)
            if args["schema"] != "off":
                r["profile_lesion"] = search_profile(ag, args["seed"], mode="off")
        if args["schema"] in ("learned", "module"):      # H5 measurability check (not the H5 test)
            from h5 import run_h5, summarise_h5
            ev = run_h5(ag, SearchEnv(ag.cfg, seed=950_000 + args["seed"]), N_TEST,
                        np.random.default_rng(950_000 + args["seed"]))
            s = summarise_h5(ev)
            # Blind: keep only how many eligible events there are. Where the head points in them
            # is the H5 result itself and is not looked at before the thresholds are locked.
            r["h5_check"] = dict(n_events=s["n"], n_with_other_salient=s.get("n_with_other_salient", 0),
                                 episodes=N_TEST)
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
