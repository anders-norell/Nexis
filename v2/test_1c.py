"""Tests for sprint 6 (1c), on untrained agents only.   python3 test_1c.py

Gradients: the A2C surrogate (advantages and returns held fixed, as in the update) plus beta * bits is
recomputed from a recorded rollout, and the hand-written gradients for the policy AND the module (through
the channel) are compared with central finite differences in float64.
"""
import numpy as np
from env import EnvConfig, SearchEnv
from agent_rl import PriorityMapAgent, matching_signal, alarm_signal
from schema_channel import ChannelSchema
from schema_1c import make_agent_1c, schema_from_z, module_grads_1c, LN2, derangement
from train_1c import rollout, a2c_grads_1c, critic_diagnostic

FAILS = []


def check(name, cond, extra=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  {extra}" if extra else ""))
    if not cond:
        FAILS.append(name)


CFG = EnvConfig(grid_sizes=(4, 5), n_distractors=(2, 4), n_salient=(1, 2), p_capture=0.5)


def record(agent, B=4, seed=3):
    """Run a rollout and keep what is needed to recompute the loss as a function of the parameters."""
    env = SearchEnv(CFG, seed=seed)
    rng = np.random.default_rng(seed)
    mod = agent.module
    G = CFG.G
    obs = env.reset(B)
    h, hm = agent.init_state(B), mod.init_state(B)
    donor = derangement(rng, B)
    steps = []
    t = 0
    while not env.done.all():
        idx = np.nonzero(~env.done)[0]
        xm, mm = mod.features(env, obs, t)
        src = donor[idx] if mod.shuffle else idx
        eps = rng.normal(0, 1, (len(idx), G * G))
        x0, m = agent.features(env, obs, t, None)
        valid = obs["action_mask"]
        steps.append(dict(idx=idx, xm=xm[src], mm=mm[src], cm=mm[idx], eps=eps, x0=x0[idx], m=m[idx], valid=valid[idx],
                          open=t > 0))
        # act with the agent's own policy so the episode is realistic
        hms, z, _, _ = mod.forward(xm[src], mm[src], hm[idx], eps, open_=t > 0, cm=mm[idx])
        hm[idx] = hms
        x = x0[idx].copy()
        x[..., agent.zcol] = z * valid[idx]
        hs, logp, _, _, _ = agent.step(x, m[idx], h[idx])
        h[idx] = hs
        a_s = (logp - np.log(-np.log(rng.random(logp.shape)))).argmax(1)
        a = np.zeros(B, int); a[idx] = a_s
        active = ~env.done
        obs, r, done, info = env.step(a)
        steps[-1].update(a=a_s, r=r, active=active)
        t += 1
    return steps, B


def forward_all(agent, steps, B):
    """Recompute the rollout's forward pass with the current parameters. Returns recs, mrecs."""
    mod = agent.module
    G2 = CFG.G ** 2
    h, hm = agent.init_state(B), mod.init_state(B)
    recs, mrecs = [], []
    for s in steps:
        idx = s["idx"]
        hms, z, bits, mc = mod.forward(s["xm"], s["mm"], hm[idx], s["eps"], open_=s["open"], cm=s["cm"])
        hm[idx] = hms
        x = s["x0"].copy()
        x[..., agent.zcol] = z * s["valid"]
        hs, logp, Vs, _, cache = agent.step(x, s["m"], h[idx])
        h[idx] = hs
        V = np.zeros(B); V[idx] = Vs
        recs.append(dict(idx=idx, cache=cache, logp=logp, a=s["a"], V=V, r=s["r"], active=s["active"],
                         zmask=s["valid"]))
        mrecs.append(dict(idx=idx, cache=mc, bits=bits, open=s["open"]))
    return recs, mrecs


def frozen_targets(recs, gamma, lam):
    T = len(recs)
    B = recs[0]["V"].shape[0]
    act = np.stack([r["active"] for r in recs]).astype(float)
    rew = np.stack([r["r"] for r in recs]) * act
    V = np.stack([r["V"] for r in recs]) * act
    adv = np.zeros((T, B)); last = np.zeros(B)
    for t in reversed(range(T)):
        nv = V[t + 1] * act[t + 1] if t + 1 < T else np.zeros(B)
        delta = rew[t] + gamma * nv - V[t]
        last = delta + gamma * lam * (act[t + 1] if t + 1 < T else 0.0) * last
        adv[t] = last * act[t]
    rets = adv + V
    n = act.sum()
    am = (adv * act).sum() / n
    sd = np.sqrt((((adv - am) * act) ** 2).sum() / n) + 1e-8
    return (adv - am) / sd * act, rets, n


def surrogate(agent, steps, B, nadv, rets, n, ent, vf, actor=1.0, critic=1.0):
    recs, mrecs = forward_all(agent, steps, B)
    L = 0.0
    for t, (rc, mr) in enumerate(zip(recs, mrecs)):
        idx = rc["idx"]
        logp = rc["logp"]
        p = np.exp(logp)
        H = -np.where(p > 0, p * logp, 0).sum(1)
        L += actor * (-(nadv[t][idx] * logp[np.arange(len(idx)), rc["a"]]).sum() - ent * H.sum()) / n
        L += critic * vf * 0.5 * ((rc["V"][idx] - rets[t][idx]) ** 2).sum() / n
        L += agent.module.beta * mr["bits"].sum() / n
    return L


def grad_check(alarm, shuffle, beta, label):
    ag = make_agent_1c(CFG, d=4, dh=5, seed=2, alarm=alarm, shuffle=shuffle, dtype=np.float64)
    ag.module.beta = beta
    rng = np.random.default_rng(0)
    for k in ag.module.p:                             # make the channel matter (otherwise tiny gradients)
        if k.startswith("c_"):
            ag.module.p[k] = ag.module.p[k] + rng.normal(0, 0.3, ag.module.p[k].shape)
    steps, B = record(ag)
    gamma, lam, ent, vf = 0.99, 0.95, 0.01, 0.5
    recs, mrecs = forward_all(ag, steps, B)
    nadv, rets, n = frozen_targets(recs, gamma, lam)
    g, dzs, info = a2c_grads_1c(ag, recs, gamma, lam, ent, vf)
    mg = module_grads_1c(ag.module, mrecs, dzs, 1.0 / info["n_act"])
    worst = {"policy": 0.0, "module": 0.0}
    for which, params, grads in (("policy", ag.p, g), ("module", ag.module.p, mg)):
        for k in params:
            flat = params[k].reshape(-1)
            for i in rng.choice(flat.size, min(10, flat.size), replace=False):
                old = flat[i]
                e = 1e-4        # magnitudes down to 1e-9: smaller steps are swamped by rounding
                flat[i] = old + e; Lp = surrogate(ag, steps, B, nadv, rets, n, ent, vf)
                flat[i] = old - e; Lm = surrogate(ag, steps, B, nadv, rets, n, ent, vf)
                flat[i] = old
                fd = (Lp - Lm) / (2 * e)
                an = grads[k].reshape(-1)[i]
                rel = abs(fd - an) / (abs(fd) + abs(an) + 1e-8)
                worst[which] = max(worst[which], rel)
    check(f"FD policy grads ({label})", worst["policy"] < 1e-4, f"worst rel {worst['policy']:.2e}")
    check(f"FD module grads through channel ({label})", worst["module"] < 1e-4, f"worst rel {worst['module']:.2e}")
    return ag, steps, B


def test_gradients():
    grad_check(False, False, 0.0, "without alarm, no cost")
    grad_check(False, False, 0.05, "without alarm, cost")
    grad_check(True, False, 0.05, "with alarm, cost")
    grad_check(True, True, 0.05, "capacity baseline, cost")


def test_critic_split():
    """Module gradient is linear in the policy's loss: actor-only + critic-only = total (no cost)."""
    ag = make_agent_1c(CFG, d=4, dh=5, seed=5, alarm=True, dtype=np.float64)
    ag.module.beta = 0.0
    steps, B = record(ag, seed=7)
    recs, mrecs = forward_all(ag, steps, B)
    _, dzs, info = a2c_grads_1c(ag, recs, 0.99, 0.95, 0.01, 0.5)
    gt = module_grads_1c(ag.module, mrecs, dzs, 1 / info["n_act"])
    d = critic_diagnostic(ag, recs, mrecs, 0.99, 0.95, 0.01, 0.5)
    tot = np.sqrt(sum((v ** 2).sum() for v in gt.values()))
    check("critic + actor = total module gradient", abs(d["total_norm"] - tot) < 1e-9 * max(1, tot),
          f"{d['total_norm']:.3e} vs {tot:.3e}; critic share {d['critic_share']:.2f}, cos {d['cos']:.2f}")
    check("diagnostic leaves beta unchanged", ag.module.beta == 0.0)


def test_inputs():
    """Alarm factor: 'without' = exactly the policy's tensor; 'with' = exactly 1b-E's input."""
    env = SearchEnv(CFG, seed=11)
    obs = env.reset(6)
    for _ in range(2):
        obs, *_ = env.step(np.random.default_rng(1).integers(0, CFG.G ** 2, 6))
    ag0 = make_agent_1c(CFG, seed=1, alarm=False)
    ag1 = make_agent_1c(CFG, seed=1, alarm=True)
    xp, mp = ag0.features(env, obs, 2, None)
    x0, m0 = ag0.module.features(env, obs, 2)
    check("without: module input == policy input (schema columns 0)", np.array_equal(xp, x0) and np.array_equal(mp, m0))
    e = ChannelSchema(CFG, seed=1, alarm=True)
    xe, _ = e.features(env, obs, 2)
    x1, _ = ag1.module.features(env, obs, 2)
    check("with: module input == 1b-E's input", np.array_equal(xe, x1))
    check("with = without + matching + alarm", np.array_equal(x1[..., :x0.shape[-1]], x0) and x1.shape[-1] == x0.shape[-1] + 2)
    base = PriorityMapAgent(CFG, seed=1, schema_mode="off")
    check("policy initialised exactly as the baseline (same seed)",
          all(np.array_equal(base.p[k], ag0.p[k]) for k in base.p))
    check("policy has the baseline's parameter count", sum(v.size for v in base.p.values()) == sum(v.size for v in ag0.p.values()))
    cap = make_agent_1c(CFG, seed=1, alarm=True, shuffle=True)
    check("capacity baseline: same module parameter count as 'with'", cap.module.n_params() == ag1.module.n_params(),
          f"{cap.module.n_params()} (without: {ag0.module.n_params()})")


def test_channel_basics():
    ag = make_agent_1c(CFG, seed=3, alarm=True)
    env = SearchEnv(CFG, seed=4)
    rng = np.random.default_rng(4)
    G = CFG.G
    obs = env.reset(5)
    xm, mm = ag.module.features(env, obs, 0)
    hm = ag.module.init_state(5)
    _, z, bits, _ = ag.module.forward(xm, mm, hm, rng.normal(0, 1, (5, G * G)), open_=False)
    check("t = 0: channel closed (z = 0, no bits)", not z.any() and not bits.any())
    _, z, bits, _ = ag.module.forward(xm, mm, hm, rng.normal(0, 1, (5, G * G)), open_=True)
    check("open: z zero on invalid cells", not (z * (1 - mm)).any())
    check("open: bits > 0 and finite", np.all(bits > 0) and np.all(np.isfinite(bits)))
    # zero-information channel: mu = 0, sigma = 1 -> 0 bits
    for k in ("c_wmu", "c_bmu", "c_wls", "c_bls"):
        ag.module.p[k][:] = 0
    _, z, bits, _ = ag.module.forward(xm, mm, hm, rng.normal(0, 1, (5, G * G)), open_=True)
    check("mu = 0, sigma = 1 -> 0 bits", np.allclose(bits, 0, atol=1e-5))


def test_rollout_and_doses():
    """Untrained agents: rollouts run in all three variants, bits split by capture, the capacity baseline's
    donor input differs from the own input (dose), and the training step changes the module."""
    G = CFG.G
    for alarm, shuffle, label in ((False, False, "without"), (True, False, "with"), (True, True, "capacity")):
        ag = make_agent_1c(CFG, seed=1, alarm=alarm, shuffle=shuffle)
        env = SearchEnv(CFG, seed=5)
        recs, st = rollout(ag, env, 64, np.random.default_rng(5))
        ok = np.isfinite(st["bits"]) and st["bits"] > 0 and np.isfinite(st["bits_cap"]) and np.isfinite(st["bits_nocap"])
        check(f"rollout {label}: bits finite, split by capture", ok,
              f"bits {st['bits']:.1f}, after capture {st['bits_cap']:.1f}, otherwise {st['bits_nocap']:.1f}"
              + (f", donor running {st['donor_running']:.2f}" if shuffle else ""))
        g, dzs, info = a2c_grads_1c(ag, recs, 0.99, 0.95, 0.01, 0.5)
        mg = module_grads_1c(ag.module, st["mrecs"], dzs, 1 / info["n_act"])
        nz = np.sqrt(sum((v ** 2).sum() for v in mg.values()))
        check(f"{label}: module gets a nonzero gradient through the channel", nz > 0, f"|g| {nz:.2e}")
    # dose check for the capacity baseline: how much of the module's input changes against the own input
    ag = make_agent_1c(CFG, seed=1, alarm=True, shuffle=True)
    env = SearchEnv(CFG, seed=9)
    rng = np.random.default_rng(9)
    obs = env.reset(200)
    donor = rng.permutation(200)
    changed, base = [], []
    t = 0
    while not env.done.all() and t < 6:
        idx = np.nonzero(~env.done)[0]
        xm, mm = ag.module.features(env, obs, t)
        diff = (np.abs(xm[donor[idx]] - xm[idx]).sum(-1) > 0) & (mm[idx] > 0)
        changed.append(diff.sum() / max(mm[idx].sum(), 1))
        obs, *_ = env.step(rng.integers(0, G * G, 200))
        t += 1
    d = derangement(np.random.default_rng(0), 256)
    check("donors form a derangement (no episode is its own donor)", not (d == np.arange(256)).any()
          and np.array_equal(np.sort(d), np.arange(256)))
    ag2 = make_agent_1c(CFG, seed=1, alarm=True, shuffle=True)
    env2 = SearchEnv(CFG, seed=12)
    _, st2 = rollout(ag2, env2, 128, np.random.default_rng(12))
    ag3 = make_agent_1c(CFG, seed=1, alarm=True)
    _, st3 = rollout(ag3, SearchEnv(CFG, seed=12), 128, np.random.default_rng(12))
    check("capacity baseline: channel and bits on the own cells (bits comparable to 'with')",
          abs(st2["bits"] - st3["bits"]) < 0.15 * st3["bits"], f"{st2['bits']:.1f} vs {st3['bits']:.1f}")
    check("capacity baseline dose: donor input differs on most valid cells", np.mean(changed) > 0.5,
          f"share of valid cells with different input: {np.mean(changed):.2f}")


if __name__ == "__main__":
    test_inputs()
    test_channel_basics()
    test_gradients()
    test_critic_split()
    test_rollout_and_doses()
    print("\nALL OK" if not FAILS else f"\n{len(FAILS)} FAILED: {FAILS}")
