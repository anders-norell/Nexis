"""Tests for the H5 evaluation (h5_eval.py, gate 5a decisions 24 Oct 2026). Untrained agents and synthetic
schemas built from the null model only; no trained agent is touched.
  python3 test_h5_eval.py"""
import numpy as np
from env import EnvConfig
from agent_rl import make_agent
from h5 import observer_restricted
from h5_bank import run_bank_module
from bank import bank_seeds
import h5_eval as H

FAILS = []
SEEDS = bank_seeds(600, 9_100_000)          # test-only seeds, disjoint from 2 000 000+ and 4 000 000+


def check(name, cond, extra=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  {extra}" if extra else ""))
    if not cond:
        FAILS.append(name)


def untrained_events(blind):
    ag = make_agent(EnvConfig(), schema_mode="channel_e", seed=5)
    return run_bank_module(ag, SEEDS, blind=blind, detail=True)["events"]


def test_detail(events):
    ok_frac = all(np.allclose(e["det"]["frac"].sum(1), 1.0) for e in events)
    check("every candidate window lies in I + A + O (frac rows sum to 1)", ok_frac)
    check("glimpse excludes the intended window (L[src, intended] = 0)",
          all(e["det"]["L"][e["det"]["src"], 0] == 0 for e in events))
    check("regions are disjoint", all(not (e["det"]["I"] & e["det"]["A"]).any() and
                                      not ((e["det"]["I"] | e["det"]["A"]) & e["det"]["O"]).any() for e in events))
    single = [e for e in events if len(e["det"]["origins"]) == 2]
    check("events with only the capturer as salient candidate have no other region",
          len(single) > 0 and all(not e["det"]["has_other"] for e in single), f"n={len(single)}")
    pr = [H.prior(e["det"]["sal"], e["det"]["p_capture"], e["det"]["cap"]) for e in events]
    check("prior sums to 1, intended >= 1 - cap", all(abs(p.sum() - 1) < 1e-9 and p[0] >= 0.4 - 1e-9 for p in pr))


def test_null_q0_is_observer():
    """At q = 0 the null is the restricted ideal observer (checked on live env states)."""
    from env import SearchEnv
    from h5 import excludes_intended_restricted
    env = SearchEnv(EnvConfig(p_capture=0.6, n_salient=(2, 3)), seed=21)
    env.reset(300)
    rng = np.random.default_rng(2)
    worst, n = 0.0, 0
    while not env.done.all() and n < 120:
        _, _, _, info = env.step(rng.integers(0, env.cfg.G ** 2, env.B))
        for b in np.nonzero(info["captured"] & info["active"] & ~info["hit"])[0]:
            ir0, ic0 = info["intended"][b]
            if not excludes_intended_restricted(env, b, ir0, ic0):
                continue
            ar0, ac0 = info["actual"][b]
            det = H.event_detail(env, b, ir0, ic0, ar0, ac0, int(info["capturer"][b]))
            post = np.array([p for _, p, _ in observer_restricted(env, b, ir0, ic0, env.last_window[b])])
            regs, _ = H.null_curve(det, n)
            worst = max(worst, np.abs(regs[0] - post @ det["frac"]).max())
            n += 1
    check("null at q = 0 equals the restricted observer", n > 50 and worst < 1e-9, f"n={n} worst={worst:.1e}")


def test_prior_predictive(events):
    """At q = 1 with all contents distinct the posterior is one-hot on the drawn glimpse source, so the
    null's mean answer must equal the capture prior (glimpse drawn after the prior, not evenly)."""
    errs = []
    for e in events[:200]:
        det = dict(e["det"])
        nc = len(det["origins"])
        det["L"] = np.eye(nc)
        regs, _ = H.null_curve(det, 1, qs=np.array([1.0]), K=20000)
        want = H.prior(det["sal"], det["p_capture"], det["cap"]) @ det["frac"]
        errs.append(np.abs(regs[0] - want).max())
    check("q = 1 replacement follows the capture prior", max(errs) < 0.02, f"max err {max(errs):.3f}")


def synthetic(d, q0, move=0.0):
    """Schema = the null itself at q0 (other random numbers), optionally with a fraction `move` of the error
    mass moved from O to I (move > 0) or from I to O (move < 0)."""
    s = dict(d)
    regs = np.zeros((d["n"], 3))
    argA = np.zeros(d["n"])
    for i, e in enumerate(d["_events"]):
        r, a = H.null_curve(e["det"], [e["seed"], e["t"], 99], qs=np.array([q0]))
        regs[i], argA[i] = r[0], a[0]
    I, A, O = regs.T.copy()
    if move > 0:
        I, O = I + move * O, (1 - move) * O
    elif move < 0:
        I, O = (1 + move) * I, O - move * I
    s.update(sI=I, sA=A, sO=O, s_argA=argA, s_rest=np.zeros(d["n"]))
    return s


def test_synthetic(events):
    d = H.build(events)
    d["_events"] = [e for e in events if e["det"]["has_other"]]
    for q0 in (0.3, 0.7):
        s = synthetic(d, q0)
        r = H.evaluate(s, "mass", n_boot=300)
        check(f"schema = null at q0 = {q0}: q recovered, no difference (mass matching)",
              abs(r["q"] - q0) < 0.05 and abs(r["diff"]) < 2 * r["se"] + 0.005,
              f"q={r['q']:.3f} diff={r['diff']:+.4f} se={r['se']:.4f}")
    # The null's argmax curve is flat for q < ~0.5 (the observer's most probable cell barely moves), so argmax
    # matching identifies q only where the curve is steep. It is the reserve, used when mass matching fails,
    # i.e. at low accuracy, which is the steep part. Tested there.
    for q0 in (0.75, 0.9):
        s = synthetic(d, q0)
        r = H.evaluate(s, "argmax", n_boot=300)
        check(f"schema = null at q0 = {q0}: q recovered, no difference (argmax matching, steep part)",
              abs(r["q"] - q0) < 0.05 and abs(r["diff"]) < 2 * r["se"] + 0.005,
              f"q={r['q']:.3f} diff={r['diff']:+.4f} se={r['se']:.4f}")
    r = H.evaluate(synthetic(d, 0.5, +0.3), "mass", n_boot=300)
    check("errors moved toward intended: passes", r["passed"], f"diff={r['diff']:+.3f} z={r['z']:.1f}")
    r = H.evaluate(synthetic(d, 0.5, -0.3), "mass", n_boot=300)
    check("errors moved away from intended: fails, negative", (not r["passed"]) and r["diff"] < 0,
          f"diff={r['diff']:+.3f} z={r['z']:.1f}")
    return d


def test_solve_and_reserve():
    c = np.linspace(0.9, 0.2, len(H.QS))
    check("solve_q inside range", abs(H.solve_q(c, 0.55) - 0.5) < 1e-9)
    check("solve_q outside range gives None", H.solve_q(c, 0.1) is None and H.solve_q(c, 0.95) is None)
    f = lambda m, a: dict(mass=dict(possible=m), argmax=dict(possible=a))
    check("reserve order: mass first", H.level_method([f(True, True), f(True, False)]) == "mass")
    check("reserve order: argmax when mass fails in a seed", H.level_method([f(True, True), f(False, True)]) == "argmax")
    check("reserve order: not assessable", H.level_method([f(False, True), f(True, False)]) is None)


def test_blind(events_blind):
    check("blind run stores no I/O split", all("sI" not in e and "s_argregion" not in e for e in events_blind))
    d = H.build(events_blind)
    try:
        H.evaluate(d, "mass", n_boot=10)
        check("evaluate refuses blind data", False)
    except AssertionError:
        check("evaluate refuses blind data", True)
    f = H.feasibility(d)
    check("feasibility works on blind data", "mass" in f and "argmax" in f, str({k: f[k]["possible"] for k in ("mass", "argmax")}))


if __name__ == "__main__":
    ev = untrained_events(blind=False)
    print(f"untrained agent: {len(ev)} H5 events, {sum(e['det']['has_other'] for e in ev)} with another salient candidate")
    test_detail(ev)
    test_null_q0_is_observer()
    test_prior_predictive([e for e in ev if e["det"]["has_other"]])
    test_solve_and_reserve()
    d = test_synthetic(ev)
    r = H.evaluate(d, H.level_method([H.feasibility(d)]) or "mass", n_boot=300)
    print("descriptive, untrained (diffuse) schema:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    test_blind(untrained_events(blind=True))
    print("\nALL PASSED" if not FAILS else f"\n{len(FAILS)} FAILED: {FAILS}")
