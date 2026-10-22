"""Tests for the H5 ideal observer and event collection. Run: python3 test_h5.py"""
import numpy as np
from env import EnvConfig, SearchEnv
from agent_rl import PriorityMapAgent
from h5 import observer_posterior, candidates, excludes_intended, run_h5, summarise_h5

FAILS = []


def check(name, cond):
    print(("ok   " if cond else "FAIL ") + name)
    if not cond:
        FAILS.append(name)


def collect(reveal, n=1500, seed=3):
    env = SearchEnv(EnvConfig(reveal=reveal, p_capture=0.5), seed=seed)
    env.reset(n)
    rng = np.random.default_rng(seed)
    rows = []
    while not env.done.all():
        act = rng.integers(0, env.cfg.G ** 2, env.B)
        _, _, _, info = env.step(act)
        for b in np.nonzero(info["active"])[0]:
            ir0, ic0 = info["intended"][b]
            post = observer_posterior(env, b, ir0, ic0, env.last_window[b], bool(info["revealed"][b]))
            prior = candidates(env, b, ir0, ic0)
            rows.append(dict(post=post, prior=prior, actual=tuple(info["actual"][b]),
                             cap=bool(info["captured"][b]), rev=bool(info["revealed"][b]),
                             excl=excludes_intended(env, b, ir0, ic0)))
    return rows


def main():
    rows = collect(1.0)
    check("posterior sums to 1", all(abs(sum(p for _, p, _ in r["post"]) - 1) < 1e-9 for r in rows))
    check("actual window always keeps posterior mass when shown",
          all(sum(p for o, p, _ in r["post"] if tuple(o) == r["actual"]) > 0 for r in rows))
    ex = [r for r in rows if r["excl"]]
    check(f"glimpse excluding the intended place gives it zero mass ({len(ex)} cases)",
          len(ex) > 50 and all(sum(p for _, p, k in r["post"] if k == "intended") == 0 for r in ex))
    check("exclusion only happens under capture", all(r["cap"] for r in ex))
    caps = [r for r in rows if r["cap"]]
    share_ex = len(ex) / max(len(caps), 1)
    print(f"     captures {len(caps)}, of which the glimpse excludes the intended place {share_ex:.2f}")
    # calibration: expected capture chance from the prior equals the observed capture rate
    exp_cap = np.mean([1 - next(p for _, p, k in r["prior"] if k == "intended") for r in rows])
    obs_cap = np.mean([r["cap"] for r in rows])
    check(f"observer prior matches capture rate ({exp_cap:.3f} vs {obs_cap:.3f})", abs(exp_cap - obs_cap) < 0.01)
    # calibration of the posterior: mass on the true window averages higher than prior mass
    pm = np.mean([sum(p for o, p, _ in r["post"] if tuple(o) == r["actual"]) for r in rows])
    check(f"observer is right more often than not ({pm:.3f})", pm > 0.8)

    rows0 = collect(0.0)
    check("masked glimpse: posterior equals prior",
          all(all(abs(a[1] - b[1]) < 1e-9 for a, b in zip(r["post"], r["prior"])) for r in rows0))
    check("masked glimpse never excludes the intended place", not any(r["excl"] for r in rows0))

    # event collection runs with an untrained 1b agent
    cfg = EnvConfig(p_capture=0.5)
    ag = PriorityMapAgent(cfg, seed=0, schema_mode="learned")
    ev = run_h5(ag, SearchEnv(cfg, seed=4), 500, np.random.default_rng(0))
    s = summarise_h5(ev)
    ok = s["n"] > 20 and all(abs(sum(e["head"].values()) - 1) < 1e-4 for e in ev) \
        and all(e["observer"]["intended"] == 0 for e in ev)
    check(f"H5 events collected ({s['n']}), head mass partitions, observer gives intended 0", ok)
    print("     untrained head:", {k: round(v, 3) for k, v in s["head_mass"].items()})

    print(f"\n{len(FAILS)} failed")
    raise SystemExit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
