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
    def covered(e):
        det = e["det"]
        sh = det["masks"][0] & ~det["I"]                     # cells of the intended window dropped as shared
        sh_frac = (det["masks"] & sh).sum((1, 2)) / det["w"] ** 2
        return np.allclose(det["frac"].sum(1) + sh_frac, 1.0)
    check("every candidate window lies in I + A + O + dropped shared cells", all(covered(e) for e in events))
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


def schema_map(e, q0, u=0.0, move=0.0, peaked=False, diffuse_on=None):
    """A synthetic schema cell map with the null's own beliefs at q0 (other random numbers):
    u      share of uniform (diffuse) mass over the valid cells, as a noisy softmax has;
    move   fraction of the O-region mass moved to the I region (> 0) or of I moved to O (< 0);
    peaked salient candidates' mass on the object's own cell instead of spread over its window."""
    det = e["det"]
    cm, post = H.null_cellmap(det, [e["seed"], e["t"], 99], q0, return_post=True)
    if peaked:
        G = cm.shape[0]
        cm = det["masks"][0] * post[0] / det["w"] ** 2
        for k in range(1, len(post)):
            cm[det["objs"][k]] += post[k]
    I, O = det["I"], det["O"]
    if move > 0 and I.any():
        m = move * cm[O].sum(); cm[O] *= 1 - move; cm[I] += m / I.sum()
    elif move < 0 and O.any():
        m = -move * cm[I].sum(); cm[I] *= 1 + move; cm[O] += m / O.sum()
    valid = np.zeros_like(cm); valid[:det["n"], :det["n"]] = 1.0
    if diffuse_on == "occupied":              # diffuse mass only on cells with objects (review R1)
        valid = valid * (det["cls"] >= 1)
    return (1 - u) * cm + u * valid / valid.sum()


def synth_events(events, **kw):
    out = []
    for e in events:
        e2 = dict(e)
        sb, sf = H.schema_regions(schema_map(e, **kw), e["det"])
        e2.update(sb); e2.update(sf)
        out.append(e2)
    return out


def test_synthetic(events):
    ev = [e for e in events if e["det"]["has_other"]]
    d0 = H.build(ev)
    raw = d0["_raw"]
    run = lambda method="mass", **kw: H.evaluate(H.build(synth_events(ev, **kw), raw=raw), method, n_boot=300)
    for q0 in (0.3, 0.7):
        for u in (0.0, 0.45, 0.85):
            r = run(q0=q0, u=u)
            check(f"schema = null at q0={q0}, diffuse share {u}: q recovered, no difference (mass)",
                  abs(r["q"] - q0) < 0.05 and abs(r["diff"]) < 2 * r["se"] + 0.005,
                  f"q={r['q']:.3f} diff={r['diff']:+.4f} se={r['se']:.4f} rest={r['schema_rest_mass']:.2f}")
    for q0 in (0.4, 0.8):
        r = run(q0=q0, u=0.65, diffuse_on="occupied")
        check(f"diffuse mass on occupied cells only, q0={q0}: no difference (class densities)",
              abs(r["diff"]) < 2 * r["se"] + 0.005, f"q={r['q']:.3f} diff={r['diff']:+.4f} se={r['se']:.4f}")
    r = run(q0=0.5, u=0.45, peaked=True)
    check("object-peaked schema with the null's beliefs: no difference (shared cells dropped)",
          abs(r["diff"]) < 2 * r["se"] + 0.005, f"diff={r['diff']:+.4f} se={r['se']:.4f}")
    # Argmax reserve: tested where the null's argmax curve is steep; on its plateau q is not identified.
    for q0 in (0.75, 0.9):
        r = run("argmax", q0=q0, u=0.45)
        check(f"schema = null at q0={q0}, diffuse: q recovered, no difference (argmax, steep part)",
              abs(r["q"] - q0) < 0.05 and abs(r["diff"]) < 2 * r["se"] + 0.005,
              f"q={r['q']:.3f} diff={r['diff']:+.4f} se={r['se']:.4f}")
    f = H.feasibility(H.build(synth_events(ev, q0=0.2, u=0.0), raw=raw))
    check("argmax on the plateau (q0 = 0.2) is not identified", not f["argmax"]["possible"],
          f"schema argA={f['argmax']['schema']:.3f}")
    r = run(q0=0.5, u=0.45, move=0.3)
    check("errors moved toward intended: passes", r["passed"], f"diff={r['diff']:+.3f} z={r['z']:.1f}")
    r = run(q0=0.5, u=0.45, move=-0.3)
    check("errors moved away from intended: fails, negative", (not r["passed"]) and r["diff"] < 0,
          f"diff={r['diff']:+.3f} z={r['z']:.1f}")
    # cluster bootstrap: duplicating every event within its own episode leaves the SE unchanged
    d = H.build(synth_events(ev, q0=0.5, u=0.45), raw=raw)
    dd = {k: (np.concatenate([v] * 3) if isinstance(v, np.ndarray) else v) for k, v in d.items() if k != "_raw"}
    dd["n"] = 3 * d["n"]
    a, b_ = H.evaluate(d, "mass", n_boot=400), H.evaluate(dd, "mass", n_boot=400)
    check("bootstrap resamples episodes (duplicates within episode keep the SE)", abs(a["se"] - b_["se"]) < 0.25 * a["se"],
          f"se {a['se']:.4f} vs {b_['se']:.4f}")
    return d0


def test_argmax_ties():
    """Null argmax ties are split evenly over the tied candidate cells."""
    G = 4
    I = np.zeros((G, G), bool); I[0, 0:2] = True
    A = np.zeros((G, G), bool); A[1, 0:2] = True
    O = np.zeros((G, G), bool); O[2, 0:2] = True
    masks = np.stack([I, A, O])
    det = dict(sal=np.array([0.5, 0.5]), p_capture=0.25, cap=0.6, src=1, L=np.eye(3), masks=masks,
               frac=np.stack([(masks & R).sum((1, 2)) for R in (I, A, O)], 1) / 2.0, w=1, I=I, A=A, O=O, s_min=0.3)
    det["masks"] = masks
    # q = 1 with equal salience: posterior = prior, actual and other tie; each window has 2 cells (w treated as
    # sqrt of cells only for the spread); intended has the highest prior so it wins alone
    _, a = H.null_curve(dict(det, w=np.sqrt(2)), 0, qs=np.array([1.0]), K=4000)
    check("argmax: intended wins alone at q = 1 (prior >= 0.4)", a[0] == 0.0, f"argA={a[0]}")
    det2 = dict(det, w=np.sqrt(2), sal=np.array([1.0, 1.0]), p_capture=0.5, cap=1.0, L=np.ones((3, 3)))
    _, a = H.null_curve(det2, 0, qs=np.array([0.0]), K=10)
    check("argmax: a tie between A and O counts one half", abs(a[0] - 0.5) < 1e-12, f"argA={a[0]}")


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
    test_argmax_ties()
    d = test_synthetic(ev)
    r = H.evaluate(d, H.level_method([H.feasibility(d)]) or "mass", n_boot=300)
    print("descriptive, untrained (diffuse) schema:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    test_blind(untrained_events(blind=True))
    print("\nALL PASSED" if not FAILS else f"\n{len(FAILS)} FAILED: {FAILS}")
