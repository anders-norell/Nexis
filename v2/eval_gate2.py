"""Sprint 2 gate evaluation (criterion locked 21 Oct 2026, before training).

For each training seed: the checkpoint chosen on validation episodes (seed 500000+s) is tested
on 4 000 new episodes (seed 900000+s), standard settings, cue level weak, with the agent's own
stochastic policy. The random agent is run on the same test episodes.

K1  hit rate exceeds the random agent's by more than 2 SD, in all seeds.
K2  hit rate >= 73 % in all seeds (best memoryless reference 71.5 %).

Diagnostics (not part of the gate): return, aim-repeat share, revisit share, first fixation on
the target colour, captures per episode, greedy hit rate, steps and hit rate per distractor count.

Writes results/sprint2_gate.json and results/sprint2.png.
"""
import json
import os
import sys
import numpy as np
from env import EnvConfig, SearchEnv
from agents import RandomAgent
from agent_rl import PriorityMapAgent
from metrics import run_metrics, summarise

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
N_TEST = 4000
K2_THRESHOLD = 0.73
SEEDS = (0, 1, 2)


class RLAdapter:
    def __init__(self, agent, rng, greedy=False):
        self.ag, self.rng, self.greedy = agent, rng, greedy

    def reset(self, env):
        self.h = self.ag.init_state(env.B)
        self.t = 0

    def act(self, env, obs):
        x, m = self.ag.features(env, obs, self.t, None)
        self.h, logp, V, sl, _ = self.ag.step(x, m, self.h)
        self.t += 1
        if self.greedy:
            return logp.argmax(1)
        return (logp - np.log(-np.log(self.rng.random(logp.shape)))).argmax(1)

    def observe(self, env, info):
        pass


def load(seed, which="best"):
    ag = PriorityMapAgent(EnvConfig(), seed=seed, schema_mode="off")
    meta = ag.load(os.path.join(RES, f"s2_base_s{seed}.{which}.npz"))
    return ag, {k: v.tolist() for k, v in meta.items()}


def per_k(ag, seed):
    """In-distribution search profile: n=8, w=1, no capture, budget ceil(1.2(k+1))."""
    cfg = EnvConfig(n_salient=(0,))
    out = {}
    for k in (4, 8, 12, 16):
        env = SearchEnv(cfg, seed=700_000 + 10 * seed + k)
        ag.cfg = cfg
        m = run_metrics(env, RLAdapter(ag, np.random.default_rng(k)), 2000,
                        overrides=dict(n=8, w=1, k=k, budget=int(np.ceil(1.2 * (k + 1)))))
        out[k] = dict(found=float(m["found"].mean()), mean_steps_found=float(m["steps"][m["found"]].mean()))
    ag.cfg = EnvConfig()
    return out


def main():
    seeds = [int(s) for s in sys.argv[1:]] or list(SEEDS)
    cfg = EnvConfig()
    res = dict(criterion=dict(K1="hit rate > random by > 2 SD in all seeds",
                              K2=f"hit rate >= {K2_THRESHOLD:.0%} in all seeds",
                              n_test=N_TEST, policy="stochastic (sampled)"), seeds={})
    for s in seeds:
        ag, meta = load(s)
        test_seed = 900_000 + s
        m = run_metrics(SearchEnv(cfg, seed=test_seed), RLAdapter(ag, np.random.default_rng(test_seed)), N_TEST)
        mr = run_metrics(SearchEnv(cfg, seed=test_seed), RandomAgent(np.random.default_rng(test_seed)), N_TEST)
        mg = run_metrics(SearchEnv(cfg, seed=test_seed), RLAdapter(ag, None, greedy=True), N_TEST)
        a, r = summarise(m), summarise(mr)
        z = (a["found"] - r["found"]) / np.sqrt(a["found_se"] ** 2 + r["found_se"] ** 2)
        res["seeds"][s] = dict(checkpoint=meta, agent=a, random=r, greedy=summarise(mg),
                               z_vs_random=float(z), K1=bool(z > 2), K2=bool(a["found"] >= K2_THRESHOLD),
                               per_k=per_k(ag, s))
        print(s, f"found {a['found']:.3f} ± {a['found_se']:.3f}  random {r['found']:.3f}  z {z:.1f}  "
              f"K1 {z > 2}  K2 {a['found'] >= K2_THRESHOLD}  greedy {summarise(mg)['found']:.3f}", flush=True)
    sd = res["seeds"]
    res["gate"] = dict(K1=all(v["K1"] for v in sd.values()), K2=all(v["K2"] for v in sd.values()),
                       all_seeds=sorted(sd.keys()) == list(SEEDS))
    res["gate"]["pass"] = res["gate"]["K1"] and res["gate"]["K2"] and res["gate"]["all_seeds"]
    with open(os.path.join(RES, "sprint2_gate.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("GATE", res["gate"])
    plot(res)


def plot(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    refs = json.load(open(os.path.join(RES, "sprint2_references.json")))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for s in res["seeds"]:
        ev = [json.loads(l) for l in open(os.path.join(RES, f"s2_base_s{s}.log.jsonl")) if '"eval"' in l]
        ax[0].plot([e["episodes"] / 1e6 for e in ev], [100 * e["eval"]["found"] for e in ev], label=f"frö {s}")
    lines = [("random", "slump"), ("memoryless_guided", "färgstyrd utan minne"),
             ("guided_intended", "styrd, minne av avsikt"), ("guided_inferred", "styrd, härleder fångst"),
             ("guided_actual", "styrd, perfekt schema")]
    for key, lab in lines:
        v = 100 * np.mean([x["found"] for x in refs[key]])
        ax[0].axhline(v, ls=":", lw=1, color="gray")
        ax[0].text(0, v + 0.5, f"{lab} {v:.0f} %", fontsize=7, color="gray")
    ax[0].axhline(100 * K2_THRESHOLD, color="crimson", lw=1)
    ax[0].text(0, 100 * K2_THRESHOLD - 2.5, "K2 73 %", fontsize=8, color="crimson")
    ax[0].set_xlabel("träningsepisoder (miljoner)"); ax[0].set_ylabel("hittar målet, valideringsepisoder (%)")
    ax[0].set_title("Inlärning, baslinje utan schema"); ax[0].legend(fontsize=8, loc="lower right")
    ks = (4, 8, 12, 16)
    for s, v in res["seeds"].items():
        ax[1].plot(ks, [100 * v["per_k"][k]["found"] for k in ks], marker="o", label=f"frö {s}")
    ax[1].set_xlabel("antal distraktorer (8×8, fönster 1, ingen fångst)")
    ax[1].set_ylabel("hittar målet inom budget (%)"); ax[1].set_title("Sökprofil")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "sprint2.png"), dpi=130)


if __name__ == "__main__":
    main()
