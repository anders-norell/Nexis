"""Tests for the sprint 2 agent: hand-written gradients against finite differences, masking,
efference remapping, schema channel off in the baseline, and the A2C gradient on a tiny case.

Run: python3 test_agent_rl.py
"""
import numpy as np
from env import EnvConfig, SearchEnv
from agent_rl import PriorityMapAgent, schema_input
from train_rl import rollout, a2c_grads

FAILS = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def unrolled_loss(agent, xs, ms, coefs):
    h = agent.init_state(xs[0].shape[0])
    L, caches = 0.0, []
    for x, m, (cl, cv, cs) in zip(xs, ms, coefs):
        h, logp, V, slogp, cache = agent.step(x, m, h)
        L += (cl * np.where(m > 0, logp, 0)).sum() + (cv * V).sum() + (cs * np.where(m > 0, slogp, 0)).sum()
        caches.append((cache, logp, slogp))
    return L, caches


def test_gradients():
    cfg = EnvConfig()
    agent = PriorityMapAgent(cfg, d=5, dh=6, seed=3, dtype=np.float64)
    env = SearchEnv(cfg, seed=1)
    rng = np.random.default_rng(0)
    B, T = 3, 3
    obs = env.reset(B)
    xs, ms, coefs = [], [], []
    for t in range(T):
        x, m = agent.features(env, obs, t, None)
        x = x + rng.normal(0, 0.3, x.shape) * m[..., None]     # make every input dimension matter
        xs.append(x); ms.append(m)
        coefs.append((rng.normal(size=m.shape) * m, rng.normal(size=B), rng.normal(size=m.shape) * m))
        obs, *_ = env.step(rng.integers(0, cfg.G ** 2, B))
    L, caches = unrolled_loss(agent, xs, ms, coefs)
    grads = agent.zero_grads()
    dh = np.zeros_like(caches[0][0][2])
    for (cache, logp, slogp), (cl, cv, cs), m in reversed(list(zip(caches, coefs, ms))):
        p, sp = np.exp(logp), np.exp(slogp)
        dlogit = cl - p * cl.sum(1, keepdims=True)
        dslog = cs - sp * cs.sum(1, keepdims=True)
        dh = agent.step_backward(cache, dlogit, cv, dslog, dh, grads)
    eps, worst = 1e-6, 0.0
    for k, v in agent.p.items():
        idx = [tuple(rng.integers(0, s) for s in v.shape) for _ in range(6)] if v.ndim else [()]
        for i in idx:
            old = v[i]
            v[i] = old + eps; Lp, _ = unrolled_loss(agent, xs, ms, coefs)
            v[i] = old - eps; Lm, _ = unrolled_loss(agent, xs, ms, coefs)
            v[i] = old
            num = (Lp - Lm) / (2 * eps)
            rel = abs(num - grads[k][i]) / max(1e-6, abs(num) + abs(grads[k][i]))
            worst = max(worst, rel)
            if rel > 1e-4:
                print("   ", k, i, num, grads[k][i])
    check("BPTT gradients match finite differences", worst < 1e-4, f"worst rel err {worst:.2e}")


def test_masking_and_remap():
    cfg = EnvConfig()
    agent = PriorityMapAgent(cfg, seed=0)
    env = SearchEnv(cfg, seed=5)
    obs = env.reset(64, overrides=dict(n=6, w=2))
    x, m = agent.features(env, obs, 0, None)
    h, logp, *_ = agent.step(x, m, agent.init_state(64))
    check("policy puts zero mass outside the grid", np.exp(logp)[m == 0].max() < 1e-12)
    check("no remapped fovea before first command", np.abs(x[..., 15:25]).sum() == 0)
    a = np.full(64, 2 * cfg.G + 3)                                # aim at (2, 3)
    obs, r, d, info = env.step(a, force_capture=np.zeros(64, bool))
    x, m = agent.features(env, obs, 1, None)
    xg = x.reshape(64, cfg.G, cfg.G, -1)
    inwin = xg[..., 15]                                           # remapped in-window flag
    exp = np.zeros((cfg.G, cfg.G)); exp[2:4, 3:5] = 1
    act = ~info["hit"]
    check("remap puts fovea on the intended 2x2 window", np.all(inwin[act] == exp))
    a = np.full(64, 5 * cfg.G + 5)                                # corner of 6x6: window shifted
    obs, r, d, info = env.step(a, force_capture=np.zeros(64, bool))
    x, m = agent.features(env, obs, 2, None)
    inwin = x.reshape(64, cfg.G, cfg.G, -1)[..., 15]
    exp = np.zeros((cfg.G, cfg.G)); exp[4:6, 4:6] = 1
    act = info["active"] & ~info["hit"]
    check("remap follows the env's edge shift", np.all(inwin[act] == exp))
    # remapped content equals the true grid content when no capture happened
    C, S = cfg.n_colors, cfg.n_shapes
    rem_col = x.reshape(64, cfg.G, cfg.G, -1)[..., 17:17 + C]
    ok = True
    for b in np.nonzero(act)[0]:
        for (rr, cc) in [(4, 4), (4, 5), (5, 4), (5, 5)]:
            col = env.color[b, rr, cc]
            want = np.zeros(C) if col < 0 else np.eye(C)[col]
            ok &= np.array_equal(rem_col[b, rr, cc], want)
    check("remapped colour matches the grid without capture", ok)


def test_schema_off_and_modes():
    cfg = EnvConfig()
    env = SearchEnv(cfg, seed=2)
    rng = np.random.default_rng(0)
    base = PriorityMapAgent(cfg, seed=1, schema_mode="off")
    recs, _ = rollout(base, env, 32, rng)
    sch = np.concatenate([r["cache"][0][..., -2:].ravel() for r in recs])
    check("baseline schema channel is all zeros", np.all(sch == 0))
    grads, _ = a2c_grads(base, recs, 0.99, 0.95, 0.01, 0.5, 0.0)
    check("1b head gets no gradient when off", np.all(grads["ws"] == 0) and np.all(grads["bs"] == 0))
    zero_rows = grads["Wz"][base.dx - 2:base.dx]
    check("schema input weights get no gradient when off", np.all(zero_rows == 0))
    for mode in ("fixed", "learned", "shuffled"):
        ag = PriorityMapAgent(cfg, seed=1, schema_mode=mode)
        recs, st = rollout(ag, SearchEnv(cfg, seed=3), 16, rng)
        s = np.concatenate([r["cache"][0][..., -2:].ravel() for r in recs[1:]]) if len(recs) > 1 else np.zeros(1)
        g, info = a2c_grads(ag, recs, 0.99, 0.95, 0.01, 0.5, 1.0 if mode == "learned" else 0.0)
        check(f"schema mode '{mode}' runs and feeds a non-zero channel", np.abs(s).sum() > 0)
    # fixed schema marks exactly the actual window
    env = SearchEnv(cfg, seed=4)
    env.reset(8, overrides=dict(n=8, w=2))
    _, _, _, info = env.step(np.full(8, 0))
    sc, vis = schema_input("fixed", env, info, np.zeros((8, cfg.G, cfg.G)))
    ok = all(sc[b, ..., 0].sum() == 4 and sc[b, info["actual"][b][0], info["actual"][b][1], 0] == 1
             for b in range(8))
    check("fixed schema marks the actual 2x2 window", ok)


def test_a2c_direction():
    """A gradient step on the A2C loss raises the log-prob of actions with positive advantage."""
    cfg = EnvConfig()
    agent = PriorityMapAgent(cfg, seed=7, dtype=np.float64)
    env = SearchEnv(cfg, seed=9)
    rng = np.random.default_rng(1)
    recs, st = rollout(agent, env, 64, rng)
    grads, _ = a2c_grads(agent, recs, 0.99, 0.95, 0.0, 0.0, 0.0)
    T = len(recs)
    act = np.stack([r["active"] for r in recs]).astype(float)
    # objective = sum adv * logp(a) under the same adv: compare before/after a tiny step
    def obj():
        h = agent.init_state(64); tot = 0.0
        for t, rc in enumerate(recs):
            idx = rc["idx"]
            hs, logp, V, sl, c = agent.step(rc["cache"][0], rc["cache"][1], h[idx])
            h[idx] = hs
            tot += (WADV[t][idx] * logp[np.arange(len(idx)), rc["a"]]).sum()
        return tot
    # recompute normalised advantages as a2c_grads does
    rew = np.stack([r["r"] for r in recs]) * act
    V = np.stack([r["V"] for r in recs]) * act
    adv = np.zeros_like(rew); last = np.zeros(64)
    for t in reversed(range(T)):
        nv = V[t + 1] * act[t + 1] if t + 1 < T else 0
        last = rew[t] + 0.99 * nv - V[t] + 0.99 * 0.95 * (act[t + 1] if t + 1 < T else 0) * last
        adv[t] = last * act[t]
    n = act.sum(); mu = adv.sum() / n; sd = np.sqrt((((adv - mu) * act) ** 2).sum() / n) + 1e-8
    global WADV
    WADV = (adv - mu) / sd * act
    before = obj()
    for k in agent.p:
        agent.p[k] -= 1e-3 * grads[k]
    after = obj()
    check("policy step increases advantage-weighted log-prob", after > before, f"{before:.4f} -> {after:.4f}")


def test_a2c_full_gradient():
    """a2c_grads equals the finite-difference gradient of the A2C surrogate (advantages and
    returns held fixed), including the running-episode subsets and the 1b loss."""
    cfg = EnvConfig()
    agent = PriorityMapAgent(cfg, d=4, dh=5, seed=11, dtype=np.float64)
    rng = np.random.default_rng(2)
    recs, _ = rollout(agent, SearchEnv(cfg, seed=12), 6, rng)
    ce, cv, cs = 0.05, 0.5, 0.7
    grads, _ = a2c_grads(agent, recs, 0.99, 0.95, ce, cv, cs)
    T = len(recs); B = 6
    act = np.stack([r["active"] for r in recs]).astype(float)
    rew = np.stack([r["r"] for r in recs]) * act
    V0 = np.stack([r["V"] for r in recs]) * act
    adv = np.zeros_like(rew); last = np.zeros(B)
    for t in reversed(range(T)):
        nv = V0[t + 1] * act[t + 1] if t + 1 < T else 0
        last = rew[t] + 0.99 * nv - V0[t] + 0.99 * 0.95 * (act[t + 1] if t + 1 < T else 0) * last
        adv[t] = last * act[t]
    rets = adv + V0
    n = act.sum(); mu = adv.sum() / n; sd = np.sqrt((((adv - mu) * act) ** 2).sum() / n) + 1e-8
    nadv = (adv - mu) / sd * act

    def surrogate():
        h = agent.init_state(B); L = 0.0
        for t, rc in enumerate(recs):
            idx = rc["idx"]
            hs, logp, V, slogp, c = agent.step(rc["cache"][0], rc["cache"][1], h[idx])
            h[idx] = hs
            p = np.exp(logp)
            H = -np.where(p > 0, p * logp, 0).sum(1)
            L += (-nadv[t][idx] * logp[np.arange(len(idx)), rc["a"]] - ce * H).sum()
            L += 0.5 * cv * ((V - rets[t][idx]) ** 2).sum()
            if rc["sl_target"] is not None:
                tg = rc["sl_target"]; z = tg.sum(1, keepdims=True)
                q = np.where(z > 0, tg / np.maximum(z, 1e-12), 0)
                L += cs * -(q * np.where(q > 0, slogp, 0)).sum()
        return L / n
    eps, worst = 1e-6, 0.0
    for k, v in agent.p.items():
        for _ in range(4):
            i = tuple(rng.integers(0, s_) for s_ in v.shape)
            old = v[i]
            v[i] = old + eps; Lp = surrogate()
            v[i] = old - eps; Lm = surrogate()
            v[i] = old
            num = (Lp - Lm) / (2 * eps)
            # relative error with an absolute floor: finite differences carry ~1e-9 noise here
            rel = abs(num - grads[k][i]) / (abs(num) + abs(grads[k][i]) + 1e-4)
            worst = max(worst, rel)
    check("A2C gradient (policy, entropy, value, 1b) matches finite differences", worst < 1e-4,
          f"worst rel err {worst:.2e}")


if __name__ == "__main__":
    test_a2c_full_gradient()
    test_gradients()
    test_masking_and_remap()
    test_schema_off_and_modes()
    test_a2c_direction()
    print("\nALL PASS" if not FAILS else f"\n{len(FAILS)} FAILED: {FAILS}")
    raise SystemExit(1 if FAILS else 0)
