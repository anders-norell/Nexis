"""Tests for the sprint 6 probe, null model and measures. Untrained agents and synthetic channels only.
   python3 test_probe_1c.py"""
import os
import tempfile
import numpy as np
import probe_1c as P
from env import EnvConfig
from agent_rl import make_agent
from schema_1c import make_agent_1c

FAILS = []
CFG = EnvConfig()


def check(name, cond, extra=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  {extra}" if extra else ""))
    if not cond:
        FAILS.append(name)


def test_probe_gradient():
    rng = np.random.default_rng(0)
    E, K = 30, 4
    F = rng.normal(0, 1, (E, K, P.NF))
    M = np.ones((E, K)); M[::3, 3] = 0
    y = rng.integers(0, 3, E)
    pb = P.Probe(seed=1)
    nf = len(pb.cols)
    pb.mu, pb.sd = rng.normal(0, 0.1, nf), rng.uniform(0.5, 2, nf)
    L, g = pb.loss_grad(F, M, y)
    worst = 0.0
    for k, v in pb.p.items():
        flat = v.reshape(-1)
        for i in range(flat.size):
            o = flat[i]
            flat[i] = o + 1e-6; Lp, _ = pb.loss_grad(F, M, y)
            flat[i] = o - 1e-6; Lm, _ = pb.loss_grad(F, M, y)
            flat[i] = o
            num = (Lp - Lm) / 2e-6
            worst = max(worst, abs(num - g[k].reshape(-1)[i]) / (abs(num) + abs(g[k].reshape(-1)[i]) + 1e-8))
    check("probe gradient matches finite differences", worst < 1e-5, f"worst rel {worst:.1e}")


def test_role_blind():
    rng = np.random.default_rng(1)
    pb = P.Probe(seed=2)
    F = rng.normal(0, 1, (5, 4, P.NF)); M = np.ones((5, 4))
    perm = np.array([2, 0, 3, 1])
    p1 = pb.probs(F, M)
    p2 = pb.probs(F[:, perm], M)
    check("probe is equivariant to candidate order (no role information)", np.allclose(p1[:, perm], p2))
    check("probe has no per-candidate constant (same function for every candidate)",
          set(pb.p) == {"W", "b", "v"} and pb.p["v"].ndim == 1)
    same = np.repeat(F[:, :1], 4, 1)
    check("identical channel values -> uniform", np.allclose(pb.probs(same, M), 0.25))


def runs():
    ag1c = make_agent_1c(CFG, seed=3, alarm=True)
    ag1b = make_agent(CFG, seed=3, schema_mode="channel_e")
    agoff = make_agent(CFG, seed=3, schema_mode="off")
    seeds = [8_000_000 + i for i in range(300)]
    return ag1c, ag1b, agoff, seeds


def test_collect():
    ag1c, ag1b, agoff, seeds = runs()
    r1 = P.collect(ag1c, "1c", seeds, block=150)
    rb = P.collect(ag1b, "1b", seeds, block=150)
    ro = P.collect(agoff, "off", seeds, block=150)
    ev = r1["events"]
    n_h5 = sum(e["h5"] for e in ev)
    check("1c: events collected, with and without capture, H5 events present",
          len(ev) > 100 and any(e["captured"] for e in ev) and any(not e["captured"] for e in ev) and n_h5 > 10,
          f"{len(ev)} events, {sum(e['captured'] for e in ev)} with capture, {n_h5} H5")
    check("1c: intended first, actual index valid, K >= 2",
          all(e["origins"][0] is not None and 0 <= e["actual"] < len(e["origins"]) and len(e["origins"]) >= 2 for e in ev))
    check("no capture -> actual is the intended window", all(e["actual"] == 0 for e in ev if not e["captured"]))
    check("capture -> actual is not the intended window", all(e["actual"] != 0 for e in ev if e["captured"]))
    check("H5 events: the intended window's content differs from the glimpse (L[actual, 0] = 0)",
          all(e["L"][e["actual"], 0] == 0 for e in ev if e["h5"]))
    check("1b runs and gives events", len(rb["events"]) > 100)
    check("baseline runs (no events)", len(ro["events"]) == 0 and len(ro["found"]) == 300)
    b = P.bits_summary(r1)
    check("bits split by capture, both groups non-empty", min(b["n_steps"]) > 0, str(b))
    rbb = P.collect(ag1c, "1c", seeds, block=150)
    check("bank run reproducible (same agent, same seeds)", np.array_equal(r1["found"], rbb["found"])
          and len(rbb["events"]) == len(ev) and np.allclose(rbb["events"][5]["F"], ev[5]["F"]))
    cap = make_agent_1c(CFG, seed=3, alarm=True, shuffle=True)
    rc = P.collect(cap, "1c", seeds, block=150)
    check("capacity baseline runs in the bank", len(rc["events"]) > 100)
    rt = P.collect(ag1c, "1c", seeds, block=150, test_shuffle=True)
    check("test-time shuffle (sanity check) runs", len(rt["found"]) == 300)
    return r1


def test_null(res):
    ev = [e for e in res["events"] if e["h5"]]
    c = np.array([P.null_curves(e) for e in ev])
    check("null at q = 0: no mass on the intended window at H5 events", np.allclose(c[:, 0, 0], 0))
    expect = np.array([1.0 / e["L"][e["actual"]].sum() for e in ev])
    check("null at q = 0: actual gets 1 / (candidates with the glimpse's content)", np.allclose(c[:, 0, 1], expect))
    check("null masses sum to 1", np.allclose(c[..., :3].sum(2), 1))
    check("null mass on actual falls with q (on average)", c[:, -1, 1].mean() < c[:, 0, 1].mean())


def synth(res, mode, rng, r=0.6):
    """Synthetic channel at t + 1 (Z1; Z0 is noise unless stated):
      actual    marks the actual window              (veridical)
      noise     nothing
      intended  always the intended window           (no capture information)
      marker    the actual window with prob r, else nothing             (no role information)
      wrong     the actual window with prob r, else a random wrong candidate, intended included (no role info)
      ast       the actual window with prob r, else the intended window (AST-like)
      anti      the actual window with prob r, else another wrong place, never the intended window
      steer     like marker at t + 1, and the intended window marked in Z0 (the command signal at t)"""
    G = CFG.G
    out = []
    for e in res["events"]:
        e = dict(e)
        Z1 = rng.normal(0, 1, (G, G)); Z0 = rng.normal(0, 1, (G, G))
        w = e["w"]
        a, Kc = e["actual"], len(e["origins"])
        hit = rng.random() < r
        wrong = [i for i in range(Kc) if i != a]
        k = None
        if mode == "actual":
            k = a
        elif mode == "intended":
            k = 0
        elif mode in ("marker", "steer"):
            k = a if hit else None
        elif mode == "wrong":
            k = a if hit else wrong[rng.integers(len(wrong))]
        elif mode == "ast":
            k = a if hit else 0
        elif mode == "anti":
            others = [i for i in wrong if i != 0]
            k = a if (hit or not others) else others[rng.integers(len(others))]
        if k is not None:
            r0, c0 = e["origins"][k]
            Z1[r0:r0 + w, c0:c0 + w] += 3.0
        if mode == "steer":
            r0, c0 = e["origins"][0]
            Z0[r0:r0 + w, c0:c0 + w] += 3.0
        e["F"] = P.window_features(Z1, Z0, e["origins"], w, e["n"])
        out.append(e)
    return dict(events=out, found=res["found"])


def test_measures(res):
    rng = np.random.default_rng(5)
    ev = res["events"]
    half = len(ev) // 2
    tmp = tempfile.mkdtemp()
    old = P.LOCK
    P.LOCK = os.path.join(tmp, "lock.json")
    try:
        try:
            P.measure(P.Probe(), res)
            check("measure refuses without the lock", False)
        except RuntimeError:
            check("measure refuses without the lock", True)
        P.write_lock("synthetic test")
        cases = [("actual", 1.0), ("noise", 0.0), ("intended", 0.0)] + \
                [("marker", r) for r in (0.5, 0.6, 0.7, 0.8)] + [("wrong", r) for r in (0.5, 0.6, 0.7)] + \
                [("ast", 0.6), ("ast", 0.7), ("anti", 0.6), ("steer", 0.6)]
        for mode, r in cases:
            s = synth(res, mode, rng, r)
            tr = P.balanced(s["events"][:half])
            te = dict(events=s["events"][half:], found=res["found"])
            out = {}
            for name, cols in (("primary", P.COLS_PRIMARY), ("sens", P.COLS_SENS)):
                pb = P.Probe(seed=0, cols=cols)
                pb.fit(tr, iters=600)
                out[name] = (P.blind_summary(pb, te), P.measure(pb, te))
            bl, m = out["primary"]
            ms = out["sens"][1]
            print(f"     {mode:8s} r={r:.1f} assessable {m['assessable']!s:5s} (c_k3) z {bl['c_k3']['z']:+6.1f}; "
                  f"om {m['om']['value']:+.3f} ({m['om']['z']:+5.1f} SE); control {m['control']['value']:+.3f} "
                  f"({m['control']['z']:+5.1f} SE, n {m['n_with_decoy']}); with null {m['control_with_null']['z']:+5.1f}; "
                  f"sens control {ms['control']['z']:+5.1f} SE")
            noinfo = mode in ("noise", "marker", "wrong", "steer")
            if mode == "intended":      # carries the intention but nothing about capture: (c) fails, H5 not judged
                check("intention-only channel: (c) fails, so H5 is not judged", not bl["c_k3"]["passed"])
            if noinfo:
                check(f"calibration: no role information ({mode}, r={r}) -> control |z| < 2 or not assessable (primary)",
                      (not m["assessable"]) or abs(m["control"]["z"]) < 2)
                check(f"calibration: ({mode}, r={r}) does not pass", not m["control"]["passed"])
            if mode == "actual":
                check("veridical channel: (c) passes, not assessable, does not pass",
                      bl["c_k3"]["passed"] and not m["assessable"] and not m["control"]["passed"])
            if mode == "ast":
                check(f"AST-like channel (r={r}): control passes", m["control"]["passed"])
            if mode == "anti":
                check("channel erring elsewhere: control fails", not m["control"]["passed"])
            if mode == "steer":
                check("command signal at t does not reach the primary probe (|z| < 2)", abs(m["control"]["z"]) < 2)
                check("... but it does reach the sensitivity probe (t + 1 and t): control > 2 SE", ms["control"]["z"] > 2)
    finally:
        P.LOCK = old


if __name__ == "__main__":
    test_probe_gradient()
    test_role_blind()
    r = test_collect()
    test_null(r)
    ag1c = make_agent_1c(CFG, seed=3, alarm=True)
    big = P.collect(ag1c, "1c", [8_100_000 + i for i in range(2000)])     # larger sample for the calibration
    test_measures(big)
    print("\nALL OK" if not FAILS else f"\n{len(FAILS)} FAILED: {FAILS}")
