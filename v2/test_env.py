"""Unit tests for the v2 search environment. Run: python3 test_env.py"""
import numpy as np
from env import EnvConfig, SearchEnv, window_origin, in_window
from agents import RandomAgent, GuidedAgent, run

FAILS = []


def check(name, cond):
    print(("ok   " if cond else "FAIL ") + name)
    if not cond:
        FAILS.append(name)


def test_layout():
    env = SearchEnv(EnvConfig(), seed=1)
    env.reset(500)
    cfg = env.cfg
    inside = all(not (env.color[b][env.n[b]:, :] >= 0).any() and not (env.color[b][:, env.n[b]:] >= 0).any()
                 for b in range(500))
    check("objects only inside the episode's grid", inside)
    check("exactly k+1 objects", all((env.color[b] >= 0).sum() == env.k[b] + 1 for b in range(500)))
    tr, tc = env.target.T
    b = np.arange(500)
    check("target has the cued colour and shape",
          (env.color[b, tr, tc] == env.target_color).all() and (env.shape[b, tr, tc] == env.target_shape).all())
    same = 0
    for bb in range(500):
        m = env.color[bb] >= 0
        m[tr[bb], tc[bb]] = False
        dup = (env.color[bb][m] == env.target_color[bb]) & (env.shape[bb][m] == env.target_shape[bb])
        same += dup.sum()
    check("no distractor identical to the target", same == 0)


def share_fraction(level, B=800):
    env = SearchEnv(EnvConfig(cue_level=level), seed=2)
    env.reset(B)
    tot = shared = 0
    for b in range(B):
        m = env.color[b] >= 0
        m[tuple(env.target[b])] = False
        shared += (env.color[b][m] == env.target_color[b]).sum()
        tot += m.sum()
    return shared / tot


def test_cue_levels():
    f_weak, f_strong = share_fraction("weak"), share_fraction("strong")
    check(f"weak: most distractors share target colour ({f_weak:.2f}, want ~0.75)", abs(f_weak - 0.75) < 0.03)
    check(f"strong: few distractors share target colour ({f_strong:.2f}, want ~0.25)", abs(f_strong - 0.25) < 0.03)
    env = SearchEnv(EnvConfig(cue_level="none"), seed=3)
    obs = env.reset(50)
    check("none: periphery carries no colour", obs["periphery"][..., 3:].sum() == 0)


def test_periphery_no_shape():
    """Two grids that differ only in distractor shapes give the same periphery."""
    env = SearchEnv(EnvConfig(), seed=4)
    env.reset(20)
    p1 = env.periphery()
    env.shape = np.where(env.shape >= 0, (env.shape + 1) % env.cfg.n_shapes, -1)
    check("periphery shows colour but not shape", np.array_equal(p1, env.periphery()))
    check("periphery shows occupancy", np.array_equal(p1[..., 1] > 0, env.color >= 0))


def test_fovea_window():
    cfg = EnvConfig(n_salient=(0,))
    env = SearchEnv(cfg, seed=5)
    env.reset(200)
    G = cfg.G
    a = env.rng.integers(0, G * G, 200)
    obs, r, done, info = env.step(a)
    ok_local = ok_shape = True
    for b in range(200):
        w = env.w[b]
        r0, c0 = info["actual"][b]
        f = obs["fovea"][b]
        ok_local &= f[..., 0].sum() == w * w                         # w*w in-window cells
        for i in range(w):
            for j in range(w):
                col, shp = env.color[b, r0 + i, c0 + j], env.shape[b, r0 + i, c0 + j]
                if col >= 0:
                    ok_shape &= f[i, j, 2 + col] == 1 and f[i, j, 2 + cfg.n_colors + shp] == 1
                else:
                    ok_shape &= f[i, j, 1] == 0
    check("fovea is the actual w x w window, egocentric", ok_local)
    check("fovea shows colour and shape of covered cells", ok_shape)
    check("fovea is a fixed W x W patch (no coordinate channels; whether position can be "
          "inferred from content is measured in check_env part C)", obs["fovea"].shape[1:3] == (cfg.W, cfg.W))


def test_window_inside_grid():
    ok = True
    for n in (6, 9):
        for w in (1, 2):
            for r in range(n):
                for c in range(n):
                    r0, c0 = window_origin(r, c, w, n)
                    ok &= 0 <= r0 <= n - w and 0 <= c0 <= n - w and in_window(r, c, r0, c0, w)
    check("window stays inside grid and covers the chosen cell", ok)


def test_params_vary():
    env = SearchEnv(EnvConfig(), seed=6)
    env.reset(2000)
    check("grid size varies over all values", set(env.n) == set(env.cfg.grid_sizes))
    check("window varies over all values", set(env.w) == set(env.cfg.windows))
    check("distractor count spans its range", env.k.min() == 4 and env.k.max() == 16)
    check("budget varies", len(set(env.budget)) > 5)


def test_prior():
    env = SearchEnv(EnvConfig(), seed=7)
    env.reset(4000)
    h = (env.n + 1) // 2
    hot = ((env.target[:, 0] < h) & (env.target[:, 1] < h)).mean()
    expect = 0.4 + 0.6 * (h ** 2 / env.n ** 2).mean()
    check(f"target prior: hot quadrant {hot:.2f}, want ~{expect:.2f}", abs(hot - expect) < 0.03)


def test_capture():
    cfg = EnvConfig(n_salient=(1, 2, 3))
    env = SearchEnv(cfg, seed=8)
    env.reset(3000)
    cnt = (env.sal_pos[..., 0] >= 0).sum(1)
    check("1-3 salient distractors per episode when asked", set(cnt) == {1, 2, 3})
    vals = env.sal_val[env.sal_pos[..., 0] >= 0]
    check("salience is graded in [s_min, 1]", vals.min() >= cfg.s_min and vals.max() <= 1 and vals.std() > 0.1)
    tgt = np.zeros(3000, bool)
    for b in range(3000):
        for (r, c) in env.sal_pos[b]:
            tgt[b] |= r >= 0 and (r, c) == tuple(env.target[b])
    check("salient objects are distractors, never the target", not tgt.any())
    # noisy salience in the periphery, fresh each step
    p1 = env.periphery()[..., 2]
    env.step(np.zeros(3000, int))
    p2 = env.periphery()[..., 2]
    d = (p1 - env.salient)[env.valid]
    check(f"periphery salience is noisy (SD {d.std():.3f}, want ~{cfg.sal_noise})", abs(d.std() - cfg.sal_noise) < 0.01)
    check("salience noise is redrawn every step", not np.allclose(p1, p2))

    # capture rate follows salience: one salient object, several fixed salience values
    G = cfg.G
    rates = []
    for s in (0.4, 1.0):
        env = SearchEnv(EnvConfig(n_salient=(1,), s_min=s), seed=9)
        env.reset(6000)
        env.sal_val[:, 0] = s
        a = np.zeros(6000, int)
        for b in range(6000):
            n, (sr, sc) = env.n[b], env.sal_pos[b, 0]
            cand = [(r, c) for r in range(n) for c in range(n) if abs(r - sr) > 2 or abs(c - sc) > 2]
            r, c = cand[env.rng.integers(len(cand))]
            a[b] = r * G + c
        _, _, _, info = env.step(a)
        rates.append(info["captured"].mean())
    check(f"capture chance = p_capture * salience ({rates[0]:.3f} vs 0.10, {rates[1]:.3f} vs 0.25)",
          abs(rates[0] - 0.10) < 0.015 and abs(rates[1] - 0.25) < 0.02)

    # several salient objects: capturer chosen in proportion to salience, total capped
    env = SearchEnv(EnvConfig(n_salient=(3,), p_capture=1.0), seed=10)
    env.reset(8000)
    env.sal_val[:] = [0.3, 0.6, 0.9]
    a = np.zeros(8000, int)
    _, _, _, info = env.step(a)
    inside = in_window(env.sal_pos[..., 0], env.sal_pos[..., 1], 0, 0, env.w[:, None])
    ok = ~inside.any(1)
    rate = info["captured"][ok].mean()
    check(f"total capture chance capped ({rate:.3f}, want {env.cfg.capture_cap})", abs(rate - env.cfg.capture_cap) < 0.02)
    who = info["capturer"][ok & info["captured"]]
    share = np.bincount(who, minlength=3) / len(who)
    check(f"capturer drawn in proportion to salience ({np.round(share, 2)}, want [0.17 0.33 0.5])",
          np.abs(share - np.array([1, 2, 3]) / 6).max() < 0.03)

    cap = info["captured"]
    sp = env.sal_pos[np.arange(8000), np.maximum(info["capturer"], 0)]
    sr0, sc0 = window_origin(sp[:, 0], sp[:, 1], env.w, env.n)
    check("captured: actual window is on the capturing object",
          ((info["actual"][cap, 0] == sr0[cap]) & (info["actual"][cap, 1] == sc0[cap])).all())
    check("not captured: actual window equals intended", (info["actual"][~cap] == info["intended"][~cap]).all())
    ci = np.nonzero(cap)[0]
    check("salient object inside the intended window never captures",
          inside.any() and not inside[ci, info["capturer"][ci]].any())

    # forced capture for H5
    env = SearchEnv(EnvConfig(n_salient=(0, 1, 2)), seed=11)
    env.reset(2000)
    a = np.zeros(2000, int)
    ir0, ic0 = window_origin(a // G, a % G, env.w, env.n)
    exists = env.sal_pos[..., 0] >= 0
    elig = exists & ~in_window(env.sal_pos[..., 0], env.sal_pos[..., 1], ir0[:, None], ic0[:, None], env.w[:, None])
    _, _, _, info = env.step(a, force_capture=np.ones(2000, bool))
    check("force_capture=True captures exactly when an eligible salient object exists",
          np.array_equal(info["captured"], elig.any(1)))
    check("forced capturer is an eligible object", elig[np.nonzero(info["captured"])[0], info["capturer"][info["captured"]]].all())
    env.reset(2000)
    _, _, _, info = env.step(a, force_capture=np.zeros(2000, bool))
    check("force_capture=False never captures", not info["captured"].any())
    env.reset(300)
    a = env.target[:, 0] * G + env.target[:, 1]
    env.step(a, force_capture=np.zeros(300, bool))   # all found
    v = env.visits.copy()
    _, r, _, info = env.step(a, force_capture=np.ones(300, bool))
    check("finished episodes: no capture, no reward, no visits",
          not info["captured"].any() and (r == 0).all() and np.array_equal(v, env.visits))


def test_efference():
    env = SearchEnv(EnvConfig(), seed=12)
    obs = env.reset(500)
    G = env.cfg.G
    check("efference copy is zero at the first step", obs["efference"].sum() == 0)
    a = env.rng.integers(0, G * G, 500)
    obs, _, _, info = env.step(a, force_capture=np.ones(500, bool))
    cell = obs["efference"].argmax(1)
    ar, ac = np.divmod(a, G)
    want = np.minimum(ar, env.n - 1) * G + np.minimum(ac, env.n - 1)
    act = info["active"]
    check("efference copy = intended cell, also when captured",
          np.array_equal(cell[act], want[act]) and info["captured"].any())
    env = SearchEnv(EnvConfig(efference_copy=False), seed=12)
    env.reset(50)
    obs, _, _, _ = env.step(np.zeros(50, int))
    check("efference copy can be switched off", obs["efference"].sum() == 0)


def test_reward_and_budget():
    cfg = EnvConfig(n_salient=(0,))
    env = SearchEnv(cfg, seed=9)
    env.reset(300)
    G = cfg.G
    a = env.target[:, 0] * G + env.target[:, 1]
    _, r, done, info = env.step(a)
    check("looking at the target finds it, +1, episode ends", (r == 1).all() and done.all() and env.found.all())
    env.reset(300)
    far = np.zeros(300, int)
    for b in range(300):
        far[b] = ((env.target[b, 0] + 4) % env.n[b]) * G + (env.target[b, 1] + 4) % env.n[b]
    while not env.done.all():
        _, r, done, info = env.step(far)       # same cell every step: revisits
        act = info["active"]
        check_ok = np.allclose(r[act & ~info["hit"]], -cfg.step_cost)
        if not check_ok:
            break
    check("each step costs step_cost, revisits included", check_ok)
    check("episode ends at the budget", (env.steps[~env.found] == env.budget[~env.found]).all())
    check("finished episodes take no more steps", (env.steps <= env.budget).all())


def test_clamp_and_k_limit():
    cfg = EnvConfig(grid_sizes=(6, 9), n_salient=(0,))
    env = SearchEnv(cfg, seed=10)
    env.reset(200, overrides=dict(n=6, w=1))
    G = cfg.G
    _, _, _, info = env.step(np.full(200, G * G - 1))     # padded corner (8, 8)
    check("out-of-grid action is clamped to the grid edge", (info["actual"] == 5).all())
    env = SearchEnv(EnvConfig(grid_sizes=(6,), n_distractors=(40, 40)), seed=10)
    env.reset(50)
    check("k is limited to n*n - 1", (env.k == 35).all() and all((env.color[b] >= 0).sum() == 36 for b in range(50)))


def test_reproducible():
    out = []
    for _ in range(2):
        env = SearchEnv(EnvConfig(), seed=11)
        res = run(env, RandomAgent(np.random.default_rng(3)), 200)
        out.append((res["steps"].copy(), res["found"].copy(), env.color.copy()))
    check("same seed gives identical episodes and outcomes",
          all(np.array_equal(x, y) for x, y in zip(out[0], out[1])))


def test_agents_order():
    env = SearchEnv(EnvConfig(), seed=12)
    rnd = run(env, RandomAgent(np.random.default_rng(1)), 2000)["found"].mean()
    env = SearchEnv(EnvConfig(), seed=12)
    gui = run(env, GuidedAgent(np.random.default_rng(1)), 2000)["found"].mean()
    check(f"guided agent beats random agent ({gui:.2f} vs {rnd:.2f})", gui > rnd + 0.1)


if __name__ == "__main__":
    for f in [test_layout, test_cue_levels, test_periphery_no_shape, test_fovea_window,
              test_window_inside_grid, test_params_vary, test_prior, test_capture,
              test_efference, test_reward_and_budget, test_clamp_and_k_limit, test_reproducible, test_agents_order]:
        f()
    print(f"\n{len(FAILS)} failed")
    raise SystemExit(1 if FAILS else 0)
