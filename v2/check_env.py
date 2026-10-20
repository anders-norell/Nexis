"""Sprint 1 environment check with the hand-coded reference agents.

A. Search slope: steps to find the target per added distractor, per cue level.
   Fixed n=8, w=1, generous budget (200), no capture, so every agent finishes.
B. Default episodes (random parameters, cue level weak): success rate and return.
C. Capture: guided agent that tracks its intended windows vs one that tracks actual ones.

Writes results/sprint1_env_check.json and results/sprint1_env_check.png.
"""
import json
import os
import numpy as np
from env import EnvConfig, SearchEnv
from agents import RandomAgent, ItemScanAgent, GuidedAgent, run

OUT = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT, exist_ok=True)
N_EP = 4000
KS = (4, 8, 12, 16)
AGENTS = {"random": RandomAgent, "item_scan": ItemScanAgent, "guided": GuidedAgent}


def slope(ks, steps, rng, n_boot=1000):
    """OLS slope of steps on k, with a bootstrap SD over episodes."""
    ks, steps = np.asarray(ks, float), np.asarray(steps, float)
    est = np.polyfit(ks, steps, 1)[0]
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(ks), len(ks))
        boots.append(np.polyfit(ks[i], steps[i], 1)[0])
    return float(est), float(np.std(boots))


def part_a():
    res, rng = {}, np.random.default_rng(0)
    for level in ("none", "weak", "strong"):
        cfg = EnvConfig(cue_level=level, p_salient=0.0)
        for name, cls in AGENTS.items():
            all_k, all_s, means, found = [], [], [], []
            for k in KS:
                env = SearchEnv(cfg, seed=100 + k)
                r = run(env, cls(np.random.default_rng(k)), N_EP,
                        overrides=dict(n=8, w=1, k=k, budget=200))
                all_k += [k] * N_EP
                all_s += list(r["steps"])
                means.append(float(r["steps"].mean()))
                found.append(float(r["found"].mean()))
            s, sd = slope(all_k, all_s, rng)
            res[f"{level}/{name}"] = dict(mean_steps=means, found=found, slope=s, slope_sd=sd)
            print(f"A {level:6s} {name:9s} steps {np.round(means, 1)}  slope {s:.3f} ± {sd:.3f}")
    return res


def part_b():
    res = {}
    cfg = EnvConfig(cue_level="weak")
    for name, cls in AGENTS.items():
        env = SearchEnv(cfg, seed=200)
        r = run(env, cls(np.random.default_rng(1)), N_EP)
        ret = np.where(r["found"], 1.0, 0.0) - cfg.step_cost * (r["steps"] - r["found"])
        res[name] = dict(found=float(r["found"].mean()), steps=float(r["steps"].mean()),
                         ret=float(ret.mean()))
        print(f"B {name:9s} found {res[name]['found']:.3f}  steps {res[name]['steps']:.2f}  "
              f"return {res[name]['ret']:.3f}")
    return res


def part_c():
    res = {}
    cfg = EnvConfig(cue_level="weak", p_salient=1.0)
    for track in ("intended", "actual"):
        env = SearchEnv(cfg, seed=300)
        r = run(env, GuidedAgent(np.random.default_rng(2), track=track), N_EP)
        f = r["found"]
        se = float(np.sqrt(f.mean() * (1 - f.mean()) / len(f)))
        res[track] = dict(found=float(f.mean()), se=se, steps=float(r["steps"].mean()),
                          captures_per_ep=float(r["n_capture"].mean()))
        print(f"C guided/{track:8s} found {f.mean():.3f} ± {se:.3f}  "
              f"captures/episode {r['n_capture'].mean():.2f}")
    return res


def plot(a, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    colors = {"random": "#9a9a9a", "item_scan": "#4a7bb7", "guided": "#d07a2d"}
    for ax, level in zip(axes, ("none", "weak", "strong")):
        for name in AGENTS:
            d = a[f"{level}/{name}"]
            ax.plot(KS, d["mean_steps"], "o-", color=colors[name],
                    label=f"{name} ({d['slope']:.2f}/distr.)")
        ax.set_title(f"cue level: {level}")
        ax.set_xlabel("distractors")
        ax.set_xticks(KS)
        ax.legend(fontsize=7, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("mean steps to find target")
    fig.suptitle("Nexis v2 sprint 1: search slope of reference agents (8x8 grid, window 1)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    out = dict(n_episodes=N_EP, A_search_slope=part_a(), B_default_episodes=part_b(),
               C_capture=part_c())
    with open(os.path.join(OUT, "sprint1_env_check.json"), "w") as f:
        json.dump(out, f, indent=1)
    plot(out["A_search_slope"], os.path.join(OUT, "sprint1_env_check.png"))
