"""Tests for the sprint 5a schema channel (schema_channel.py) and the matching signal.
Gradients are checked against finite differences in float64.   python3 test_schema_channel.py"""
import numpy as np
from env import EnvConfig, SearchEnv
from agent_rl import PriorityMapAgent, matching_signal, make_agent
from schema_channel import ChannelSchema, channel_grads, LN2

FAILS = []


def check(name, cond, extra=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  {extra}" if extra else ""))
    if not cond:
        FAILS.append(name)


def rollout_records(mod, T=4, B=3, seed=0):
    """A short deterministic rollout of the module on random inputs, with fixed noise and targets."""
    rng = np.random.default_rng(seed)
    G2 = mod.cfg.G ** 2
    xs = [rng.normal(0, 1, (B, G2, mod.net.dx)) for _ in range(T)]
    m = np.ones((B, G2)); m[:, -5:] = 0.0
    eps = [rng.normal(0, 1, (B, G2)) for _ in range(T)]
    tg = []
    for t in range(T):
        q = np.zeros((B, G2))
        if t > 0:
            for b in range(B):
                q[b, rng.choice(G2 - 5, 2, replace=False)] = 1.0
            tg.append(q)
        else:
            tg.append(None)
    return xs, m, eps, tg


def total_loss(mod, xs, m, eps, tg):
    B = xs[0].shape[0]
    h = mod.init_state(B)
    recs = []
    for t in range(len(xs)):
        h, logp, bits, cache = mod.forward(xs[t], m, h, eps[t])
        recs.append(dict(idx=np.arange(B), cache=cache, logp=logp, target=tg[t], bits=bits))
    n_tot = sum(int((r["target"].sum(1) > 0).sum()) for r in recs if r["target"] is not None)
    L = 0.0
    for r in recs:
        if r["target"] is None:
            continue
        q = r["target"] / r["target"].sum(1, keepdims=True)
        L += -(q * np.where(q > 0, r["logp"], 0)).sum() + mod.beta * r["bits"].sum()
    return L / n_tot, recs


def test_gradients():
    cfg = EnvConfig(grid_sizes=(4,), n_distractors=(2, 3))
    mod = ChannelSchema(cfg, d=5, dh=6, seed=1, dtype=np.float64)
    mod.beta = 0.07
    xs, m, eps, tg = rollout_records(mod)
    L, recs = total_loss(mod, xs, m, eps, tg)
    grads, ce, bits = channel_grads(mod, recs)
    worst = 0.0
    rng = np.random.default_rng(5)
    for k in ("c_wmu", "c_bmu", "c_wls", "c_bls", "c_wd", "c_bd", "Wz", "Un", "bz"):
        P = mod.p[k]
        flat = P.reshape(-1)
        for i in rng.choice(flat.size, min(6, flat.size), replace=False):
            old = flat[i]
            e = 1e-6
            flat[i] = old + e; Lp, _ = total_loss(mod, xs, m, eps, tg)
            flat[i] = old - e; Lm, _ = total_loss(mod, xs, m, eps, tg)
            flat[i] = old
            num = (Lp - Lm) / (2 * e)
            ana = grads[k].reshape(-1)[i]
            err = abs(num - ana) / max(1e-8, abs(num) + abs(ana))
            worst = max(worst, err if abs(num) + abs(ana) > 1e-9 else 0.0)
    check("channel gradients match finite differences (CE + beta * bits, BPTT)", worst < 1e-5, f"worst rel err {worst:.2e}")
    check("reported bits = loss's bits", abs(bits - sum(r["bits"].sum() for r in recs if r["target"] is not None) /
                                          sum(len(r["bits"]) for r in recs if r["target"] is not None)) < 1e-9)


def test_bits_zero_for_pure_noise():
    cfg = EnvConfig()
    mod = ChannelSchema(cfg, seed=2, dtype=np.float64)
    for k in ("c_wmu", "c_wls"):
        mod.p[k][:] = 0
    mod.p["c_bmu"][:] = 0; mod.p["c_bls"][:] = 0
    B, G2 = 2, cfg.G ** 2
    x = np.random.default_rng(0).normal(0, 1, (B, G2, mod.net.dx))
    _, logp, bits, _ = mod.forward(x, np.ones((B, G2)), mod.init_state(B), np.zeros((B, G2)))
    check("a channel that sends N(0,1) carries 0 bits", np.allclose(bits, 0))


def test_decoder_sees_only_channel():
    """Two different inputs with the same z give the same schema map."""
    cfg = EnvConfig()
    mod = ChannelSchema(cfg, seed=3, dtype=np.float64)
    B, G2 = 2, cfg.G ** 2
    rng = np.random.default_rng(1)
    x1, x2 = rng.normal(0, 1, (2, B, G2, mod.net.dx))
    m = np.ones((B, G2))
    h1, _, _, c1 = mod.forward(x1, m, mod.init_state(B), np.zeros((B, G2)))
    h2, _, _, c2 = mod.forward(x2, m, mod.init_state(B), np.zeros((B, G2)))
    z1 = c1[6]
    eps2 = (z1 - c2[2]) / c2[4]                       # noise that makes z equal for input 2
    _, lp1, _, _ = mod.forward(x1, m, mod.init_state(B), np.zeros((B, G2)))
    _, lp2, _, _ = mod.forward(x2, m, mod.init_state(B), eps2)
    check("schema map depends on the input only through the channel value z", np.allclose(lp1, lp2))


def test_matching_signal_detects_exactly_detectable_captures():
    """Dose check: the matching signal shows a mismatch at exactly those captures where the restricted
    ideal observer (occupied + colour) can tell the glimpse did not come from the intended window."""
    from h5 import excludes_intended_restricted
    env = SearchEnv(EnvConfig(p_capture=0.6), seed=4)
    obs = env.reset(400)
    ag = PriorityMapAgent(env.cfg, matching=True)
    rng = np.random.default_rng(4)
    agree, n_cap, n_det, n_mis_nocap = 0, 0, 0, 0
    while not env.done.all():
        a = rng.integers(0, env.cfg.G ** 2, env.B)
        obs, _, _, info = env.step(a)
        x, _ = ag.features(env, obs, 1, None)
        msig = x[..., -1].reshape(env.B, env.cfg.G, env.cfg.G)
        for b in np.nonzero(info["active"] & ~info["hit"])[0]:
            mis = bool((msig[b] < 0).any())
            ir0, ic0 = info["intended"][b]
            if info["captured"][b]:
                n_cap += 1
                det = excludes_intended_restricted(env, b, ir0, ic0)
                n_det += det
                agree += (mis == det)
            else:
                n_mis_nocap += mis
    check(f"matching mismatch == restricted observer can detect capture ({n_cap} captures, {n_det} detectable)",
          agree == n_cap)
    check("no mismatch without capture", n_mis_nocap == 0, f"{n_mis_nocap}")


def test_alarm_signal_alternative_e():
    """The alarm is 1 exactly when the matching signal shows a mismatch anywhere, the same on every cell,
    and only the alternative-E module gets it."""
    from h5 import excludes_intended_restricted
    env = SearchEnv(EnvConfig(p_capture=0.6), seed=8)
    obs = env.reset(300)
    ag = PriorityMapAgent(env.cfg, matching=True, alarm=True)
    rng = np.random.default_rng(8)
    ok_any, ok_flat, agree, n_cap = True, True, 0, 0
    while not env.done.all():
        obs, _, _, info = env.step(rng.integers(0, env.cfg.G ** 2, env.B))
        x, _ = ag.features(env, obs, 1, None)
        G = env.cfg.G
        ms, al = x[..., -2].reshape(env.B, G, G), x[..., -1].reshape(env.B, G, G)
        ok_any &= np.array_equal(al[:, 0, 0] > 0, (ms < 0).any((1, 2)))
        ok_flat &= bool((al == al[:, :1, :1]).all())
        for b in np.nonzero(info["active"] & info["captured"] & ~info["hit"])[0]:
            n_cap += 1
            agree += (al[b, 0, 0] > 0) == excludes_intended_restricted(env, b, *info["intended"][b])
    check("alarm = mismatch anywhere", ok_any)
    check("alarm is the same on every cell (broadcast, says nothing about where)", ok_flat)
    check(f"alarm goes off exactly at detectable captures ({n_cap} captures)", agree == n_cap)
    e = make_agent(EnvConfig(), schema_mode="channel_e"); c = make_agent(EnvConfig(), schema_mode="channel")
    check("only the E module has the alarm channel", e.module.net.dx == c.module.net.dx + 1 and not c.module.net.alarm)


def test_existing_agents_unchanged():
    """Policy and module features without matching are exactly as before (sprint 4 checkpoints load)."""
    import os, json
    from eval_mvp import load
    ag, _ = load("s4_fixed_s10")
    C, S = ag.cfg.n_colors, ag.cfg.n_shapes
    check("policy has no matching channel", ag.matching is False and ag.dx == (3 + C) + (C + S) + (2 + C + S) + 5)
    # bit-identical to the sprint 4 agent code on a real batch
    import importlib.util
    path = os.environ.get("OLD_AGENT")
    if not path:                                   # agent_rl.py as of sprint 4 (unchanged since sprint 3)
        import subprocess, tempfile
        src = subprocess.run(["git", "show", "5b999ed:v2/agent_rl.py"], capture_output=True, text=True,
                             cwd=os.path.dirname(os.path.abspath(__file__)), check=True).stdout
        d = tempfile.mkdtemp(); path = os.path.join(d, "agent_rl_s4.py"); open(path, "w").write(src)
    spec = importlib.util.spec_from_file_location("agent_rl_s4", path)
    old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    oa = old.PriorityMapAgent(ag.cfg, schema_mode="fixed"); oa.p = {k: v.copy() for k, v in ag.p.items()}
    env = SearchEnv(ag.cfg, seed=9); obs = env.reset(64)
    sch = np.random.default_rng(0).random((64, ag.cfg.G, ag.cfg.G, 2))
    x1, m1 = ag.features(env, obs, 2, sch); x2, m2 = oa.features(env, obs, 2, sch)
    h = ag.init_state(64)
    o1 = ag.step(x1, m1, h); o2 = oa.step(x2, m2, h)
    check("features and step identical to sprint 4 code", np.array_equal(x1, x2) and
          all(np.array_equal(a, b) for a, b in zip(o1[:4], o2[:4])))
    ag2 = make_agent(EnvConfig(), schema_mode="channel")
    check("channel module has one extra input channel", ag2.module.net.dx == ag.dx + 1)
    from pilot_5a import load as pload, RES as PRES
    if os.path.exists(os.path.join(PRES, "p5_ref1e-4_s20.best.npz")):
        a3, _, _ = pload(os.path.join(PRES, "p5_ref1e-4_s20.best.npz"))
        check("first-round pilot modules still load (no alarm)", a3.module.net.alarm is False)


if __name__ == "__main__":
    import sys
    for f in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        try:
            f()
        except Exception as e:
            FAILS.append(f.__name__); print("FAIL", f.__name__, repr(e))
    print(f"{len(FAILS)} failed")
    sys.exit(1 if FAILS else 0)
