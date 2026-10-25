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


def synth_maps(e, mode, rng, r, G):
    """Synthetic channel maps (Z1 at t + 1, Z0 at t) for one event:
      actual    marks the actual window              (veridical)
      noise     nothing
      intended  always the intended window           (no capture information)
      marker    the actual window with prob r, else nothing             (no role information)
      wrong     the actual window with prob r, else a random wrong candidate, intended included (no role info)
      salient   every salient candidate window, never the intended window (codes salience only; review finding 1)
      ast       the actual window with prob r, else the intended window (AST-like)
      anti      the actual window with prob r, else another wrong place, never the intended window
      steer     like marker at t + 1, and the intended window marked in Z0 (the command signal at t)
      passive   noise plus a weak mark (+0.3) on the intended window (passive transmission of intended-window input)
    Channels built from the environment's maps (second pre-lock review, finding A; need collect(store_maps=True)):
      damped    salient objects, damped where the spotlight actually was
      unvis_sal salient objects on cells not yet covered
      priority  target colour and salience on cells not yet covered (where to look next)
      visited   the true coverage (cells actually covered, this step included)
      effcov    coverage as the efference copy has it: covered before, plus the actual AND the intended window
                (a memory of where one aimed; AST-consistent, reported, expected to pass)
      astsal    ast (r) plus the salience map (the review's mixed case)"""
    Z1 = rng.normal(0, 1, (G, G)); Z0 = rng.normal(0, 1, (G, G))
    w = e["w"]
    a, Kc = e["actual"], len(e["origins"])
    hit = rng.random() < r
    wrong = [i for i in range(Kc) if i != a]
    ks = []
    if mode == "actual":
        ks = [a]
    elif mode == "intended":
        ks = [0]
    elif mode in ("marker", "steer"):
        ks = [a] if hit else []
    elif mode == "wrong":
        ks = [a if hit else wrong[rng.integers(len(wrong))]]
    elif mode == "salient":
        ks = list(range(1, Kc))
    elif mode == "ast":
        ks = [a if hit else 0]
    elif mode == "anti":
        others = [i for i in wrong if i != 0]
        ks = [a if (hit or not others) else others[rng.integers(len(others))]]
    for k in ks:
        r0, c0 = e["origins"][k]
        Z1[r0:r0 + w, c0:c0 + w] += 3.0
    if mode == "steer":
        r0, c0 = e["origins"][0]
        Z0[r0:r0 + w, c0:c0 + w] += 3.0
    if mode == "passive":
        r0, c0 = e["origins"][0]
        Z1[r0:r0 + w, c0:c0 + w] += 0.3
    if mode in ("damped", "unvis_sal", "priority", "visited", "effcov", "astsal"):
        mp = e["maps"]
        vis_after = np.maximum(mp["vis_before"], mp["actual"])
        if mode == "damped":
            Z1 += 3.0 * mp["sal"] * (1 - mp["actual"])
        elif mode == "unvis_sal":
            Z1 += 3.0 * mp["sal"] * (1 - vis_after)
        elif mode == "priority":
            Z1 += 3.0 * np.minimum(mp["tcol"] + mp["sal"], 1) * (1 - vis_after)
        elif mode == "visited":
            Z1 += 3.0 * vis_after
        elif mode == "effcov":
            Z1 += 3.0 * np.maximum(vis_after, mp["intended"])
        elif mode == "astsal":
            k = a if hit else 0
            r0, c0 = e["origins"][k]
            Z1[r0:r0 + w, c0:c0 + w] += 3.0
            Z1 += 1.5 * mp["sal"]
    return Z1, Z0


def synth(res, mode, rng, r=0.6, shadow_mode=None):
    """Events with the synthetic channel as F / Fd (and, with shadow_mode, a second channel as F_sh / Fd_sh)."""
    G = CFG.G
    out = []
    for e in res["events"]:
        e = dict(e)
        for fk, fdk, md in (("F", "Fd", mode), ("F_sh", "Fd_sh", shadow_mode)):
            if md is None:
                continue
            Z1, Z0 = synth_maps(e, md, rng, r, G)
            e[fk] = P.window_features(Z1, Z0, e["origins"], e["w"], e["n"])
            if e["matched"] is not None:
                e["Fm" if fk == "F" else "Fm_sh"] = P.window_features(Z1, Z0, [e["matched"]], e["w"], e["n"])
        out.append(e)
    return dict(events=out, found=res["found"])


def test_decoys(res):
    ev = [e for e in res["events"] if e["h5"]]
    have = [e for e in ev if e["decoy"] is not None]
    check("decoy is a wrong salient candidate, not intended, not actual, content differs from the glimpse",
          all(e["decoy"] not in (0, e["actual"]) and e["L"][e["actual"], e["decoy"]] == 0 for e in have),
          f"{len(have)} of {len(ev)} H5 events have one")


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
                [("salient", 0.6), ("passive", 0.0), ("ast", 0.6), ("ast", 0.7), ("anti", 0.6), ("steer", 0.6),
                 ("damped", 0.0), ("unvis_sal", 0.0), ("priority", 0.0), ("visited", 0.0), ("effcov", 0.0),
                 ("astsal", 0.6)]
        for mode, r in cases:
            s = synth(res, mode, rng, r)
            tr = P.training_set(s["events"][:half])
            te = dict(events=s["events"][half:], found=res["found"])
            out = {}
            for name, cols in (("primary", P.COLS_PRIMARY), ("sens", P.COLS_SENS)):
                pb = P.Probe(seed=0, cols=cols)
                pb.fit(tr, iters=600)
                out[name] = (P.blind_summary(pb, te), P.measure(pb, te))
            bl, m = out["primary"]
            ms = out["sens"][1]
            cr = m["criterion"]
            print(f"     {mode:9s} r={r:.1f} (c_k3) z {bl['c_k3']['z']:+6.1f}; decoy {cr['decoy']['z']:+6.1f} SE, "
                  f"matched {cr['matched']['z']:+6.1f} SE (n {cr['n']}) -> {'PASS' if cr['passed'] else 'no'}; "
                  f"sens {ms['criterion']['decoy']['z']:+5.1f}/{ms['criterion']['matched']['z']:+5.1f}")
            noinfo = mode in ("noise", "marker", "wrong", "steer", "salient", "passive", "damped", "unvis_sal",
                              "priority", "visited", "intended", "anti")
            if noinfo:
                check(f"calibration: {mode} (r={r}) does not pass the joint criterion", not cr["passed"])
            if mode in ("ast",):
                check(f"AST-like channel (r={r}) passes the joint criterion", cr["passed"])
            if mode == "actual":
                check("veridical channel: (c) passes, joint criterion does not", bl["c_k3"]["passed"] and not cr["passed"])
            if mode == "steer":
                check("steering signal at t does not pass in the sensitivity probe either", not ms["criterion"]["passed"])
        # paired comparison with a shadow channel (the judged quantity)
        for tm, sm, expect in (("ast", "passive", True), ("passive", "passive", False), ("salient", "noise", False)):
            s = synth(res, tm, rng, 0.6, shadow_mode=sm)
            tr = P.training_set(s["events"][:half])
            te = dict(events=s["events"][half:], found=res["found"])
            pt = P.Probe(seed=0); pt.fit(tr, iters=600)
            ps = P.Probe(seed=0); ps.fit(tr, iters=600, fk="F_sh")
            mp = P.measure_pair(pt, ps, te)
            a = mp["all"]
            print(f"     pair {tm:8s} vs shadow {sm:8s}: {a['value']:+.3f} ({a['z']:+5.1f} SE, n {a['n']}); "
                  f"trained {a['trained']:+.3f}, untrained {a['untrained']:+.3f}")
            if expect:     # reported only, not judged
                check(f"paired (reported): {tm} vs shadow {sm} > 2 SE", mp["passed"])
            else:
                check(f"paired (reported): {tm} vs shadow {sm} does not pass", not mp["passed"])
    finally:
        P.LOCK = old


if __name__ == "__main__":
    test_probe_gradient()
    test_role_blind()
    r = test_collect()
    test_null(r)
    ag1c = make_agent_1c(CFG, seed=3, alarm=True)
    big = P.collect(ag1c, "1c", [8_100_000 + i for i in range(2000)], store_maps=True)   # calibration sample
    test_decoys(big)
    test_measures(big)
    print("\nALL OK" if not FAILS else f"\n{len(FAILS)} FAILED: {FAILS}")
