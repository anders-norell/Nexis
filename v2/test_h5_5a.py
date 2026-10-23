"""Tests for sprint 5a's H5 machinery: restricted observer, mixing null model (condition a, synthetic
cases), bank runner for module agents, H2 (b) in the module's map. Untrained agents only.
  python3 test_h5_5a.py"""
import numpy as np
from env import EnvConfig, SearchEnv
from agent_rl import make_agent
from h5 import observer_restricted, observer_mixed, excludes_intended_restricted
from h5_bank import run_bank_module, summary, N_MIN
from bank import bank_seeds, H2_T

FAILS = []


def check(name, cond, extra=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  {extra}" if extra else ""))
    if not cond:
        FAILS.append(name)


def synthetic_events(n=300, seed=7):
    """Real environment states with a capture that the restricted observer can detect."""
    env = SearchEnv(EnvConfig(p_capture=0.6, n_salient=(2, 3)), seed=seed)
    env.reset(600)
    rng = np.random.default_rng(seed)
    ev = []
    while not env.done.all() and len(ev) < n:
        _, _, _, info = env.step(rng.integers(0, env.cfg.G ** 2, env.B))
        for b in np.nonzero(info["captured"] & info["active"] & ~info["hit"])[0]:
            ir0, ic0 = info["intended"][b]
            if excludes_intended_restricted(env, b, ir0, ic0):
                ev.append((env, b, ir0, ic0, env.last_window[b].copy(), tuple(info["actual"][b])))
        if len(ev) >= n:
            break
    return ev[:n]


def mass(post, key):
    return sum(p for o, p, k in post if (k == "intended" if key == "intended" else tuple(o) == key and k != "intended"))


def test_null_model_mixing():
    import copy
    # freeze env states: re-simulate is expensive, so evaluate immediately per event
    env = SearchEnv(EnvConfig(p_capture=0.6, n_salient=(2, 3)), seed=11)
    env.reset(400)
    rng = np.random.default_rng(1)
    zero_ok, pos, acc = True, [], {0.0: [], 0.3: [], 0.8: []}
    n = 0
    while not env.done.all() and n < 150:
        _, _, _, info = env.step(rng.integers(0, env.cfg.G ** 2, env.B))
        for b in np.nonzero(info["captured"] & info["active"] & ~info["hit"])[0]:
            ir0, ic0 = info["intended"][b]
            if not excludes_intended_restricted(env, b, ir0, ic0):
                continue
            g, act = env.last_window[b], tuple(info["actual"][b])
            p0 = observer_restricted(env, b, ir0, ic0, g)
            zero_ok &= mass(p0, "intended") == 0.0
            pm = observer_mixed(env, b, ir0, ic0, g, 0.3, np.random.default_rng(n), K=100)
            pos.append(mass(pm, "intended"))
            for q in acc:
                acc[q].append(mass(observer_mixed(env, b, ir0, ic0, g, q, np.random.default_rng(n), K=100), act))
            n += 1
    check(f"undamped observer gives the intended place exactly 0 at H5 events ({n} synthetic cases)", zero_ok)
    check("mixing null model (q = 0.3) puts mass on the intended place", np.mean(pos) > 0.05 and min(pos) >= 0,
          f"mean {np.mean(pos):.3f}")
    a = [np.mean(acc[q]) for q in sorted(acc)]
    check("accuracy falls as q rises, so q can be matched to the schema's accuracy", a[0] > a[1] > a[2],
          " ".join(f"{x:.3f}" for x in a))
    check("damping by a power would keep the intended place at 0 (0 ** lambda = 0)", 0.0 ** 0.3 == 0.0)


def test_bank_runner_reproducible_and_independent():
    ag = make_agent(EnvConfig(), seed=5, schema_mode="channel")
    s = bank_seeds(40)
    r1 = run_bank_module(ag, s, block=20)
    r2 = run_bank_module(ag, s, block=20)
    check("module bank runs are reproducible", all(np.array_equal(r1[k], r2[k]) for k in ("found", "steps", "bits")))
    one = run_bank_module(ag, s[7:8], block=1)
    check("an episode is the same whatever else is in its batch",
          one["found"][0] == r1["found"][7] and one["steps"][0] == r1["steps"][7] and np.isclose(one["bits"][0], r1["bits"][7]))
    sm = summary(r1)
    check("summary fields in range", 0 <= sm["acc_all_steps"] <= 1 and sm["bits_per_step"] > 0)
    check("blind run stores no group labels", all("group_wrong" not in e for e in r1["events"])
          and "group_counts" not in r1 and set(r1["dose_groups"]) == {"right_ok", "wrong_ok"})
    base = make_agent(EnvConfig(), seed=5, schema_mode="off")
    rb = run_bank_module(base, s, block=20)
    check("baseline runs through the same runner", len(rb["found"]) == 40 and len(rb["events"]) == 0)


def test_h2b_in_module_map():
    ag = make_agent(EnvConfig(), seed=6, schema_mode="channel")
    s = bank_seeds(60)
    base = run_bank_module(ag, s, block=30, record=True)
    lie = run_bank_module(ag, s, intervention="b", block=30, record=True)
    same, exact = True, True
    for tb, tl, Bm in zip(base["trace"], lie["trace"], lie["cellsB"]):
        for k in range(min(H2_T, len(tb))):
            same &= np.array_equal(tb[k]["a"], tl[k]["a"])
            if tb[k]["schema"] is not None:
                same &= np.array_equal(tb[k]["schema"], tl[k]["schema"])
        if len(tb) > H2_T:
            sb, sl = tb[H2_T]["schema"], tl[H2_T]["schema"]
            diff = sb[..., 1] != sl[..., 1]
            exact &= not (diff & ~Bm).any() and bool((sl[..., 1][Bm] == 1).all() if Bm.any() else True)
            exact &= np.array_equal(sb[..., 0], sl[..., 0])
    check("H2 (b) on 1b: identical before the 4th command", same)
    check("H2 (b) on 1b: only the chosen cells change, shown as covered, in the running map", exact)
    check("H2 (b) on 1b: commands counted at most 3", lie["aim_b"].max() <= 3 and base["aim_b"].max() == 0)


if __name__ == "__main__":
    import sys
    for f in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        try:
            f()
        except Exception as e:
            import traceback; traceback.print_exc()
            FAILS.append(f.__name__); print("FAIL", f.__name__, repr(e))
    print(f"{len(FAILS)} failed")
    sys.exit(1 if FAILS else 0)
