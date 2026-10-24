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
    pb.mu, pb.sd = rng.normal(0, 0.1, P.NF), rng.uniform(0.5, 2, P.NF)
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
    check("null masses sum to 1", np.allclose(c.sum(2), 1))
    check("null mass on actual falls with q (on average)", c[:, -1, 1].mean() < c[:, 0, 1].mean())


def synth(res, mode, rng):
    """Replace the channel by a synthetic one: 'actual' marks the actual window, 'intended' always the
    intended window, 'noise' nothing. Features recomputed from synthetic maps."""
    G = CFG.G
    out = []
    for e in res["events"]:
        e = dict(e)
        Z1 = rng.normal(0, 1, (G, G)); Z0 = rng.normal(0, 1, (G, G))
        w = e["w"]
        k = None
        if mode == "actual":
            k = e["actual"]
        elif mode == "intended":
            k = 0
        elif mode == "ast":                    # right 60 %, otherwise falls back on the intended window
            k = e["actual"] if rng.random() < 0.6 else 0
        elif mode == "anti":                   # right 60 %, otherwise another place, never the intended window
            others = [i for i in range(len(e["origins"])) if i not in (0, e["actual"])]
            k = e["actual"] if (rng.random() < 0.6 or not others) else others[rng.integers(len(others))]
        if k is not None:
            r0, c0 = e["origins"][k]
            Z1[r0:r0 + w, c0:c0 + w] += 3.0
        e["F"] = P.window_features(Z1, Z0, e["origins"], w, e["n"])
        out.append(e)
    return dict(events=out, found=res["found"])


def test_measures(res):
    rng = np.random.default_rng(5)
    n = len(res["events"])
    half = n // 2
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
        for mode in ("actual", "noise", "intended", "ast", "anti"):
            s = synth(res, mode, rng)
            tr = dict(events=P.balanced(s["events"][:half]))
            te = dict(events=s["events"][half:], found=res["found"])
            pb = P.Probe(seed=0)
            pb.fit(tr["events"], iters=600)
            bl = P.blind_summary(pb, te)
            m = P.measure(pb, te)
            print(f"     {mode:9s} assessable {m['assessable']} (c) {bl['c']['mass_actual']:.2f} vs {bl['c']['uniform']:.2f} z {bl['c']['z']:.1f};"
                  f" q {m['q']}; om probe {m['om']['probe_mass_intended']:.3f} null {m['om']['null_mass_intended']:.3f}"
                  f" diff {m['om']['diff']:+.3f} ({m['om']['z']:+.1f} SE); var {m['var']['diff']:+.3f}")
            if mode == "actual":
                check("synthetic veridical channel: (c) passes; 'om' not assessable (probe beats the observer)",
                      bl["c"]["passed"] and not m["assessable"] and not m["om"]["passed"])
            if mode == "noise":
                check("synthetic empty channel: (c) fails, 'om' does not pass", not bl["c"]["passed"] and not m["om"]["passed"])
            if mode == "intended":      # no information about capture at all: (c) must fail, so H5 is not judged
                check("synthetic intention-only channel: (c) fails", not bl["c"]["passed"])
            if mode == "ast":
                check("synthetic AST-like channel (errs towards intention): 'om' passes", m["om"]["passed"])
            if mode == "anti":
                check("synthetic channel erring elsewhere: 'om' fails", not m["om"]["passed"])
    finally:
        P.LOCK = old


if __name__ == "__main__":
    test_probe_gradient()
    test_role_blind()
    r = test_collect()
    test_null(r)
    test_measures(r)
    print("\nALL OK" if not FAILS else f"\n{len(FAILS)} FAILED: {FAILS}")
