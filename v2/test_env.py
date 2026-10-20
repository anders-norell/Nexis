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
    cfg = EnvConfig(p_salient=0.0)
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
    cfg = EnvConfig(p_salient=1.0, p_capture=0.2)
    env = SearchEnv(cfg, seed=8)
    env.reset(3000)
    has = env.sal_pos[:, 0] >= 0
    check(f"salient distractor present when asked ({has.mean():.3f})", has.mean() > 0.97)
    ok_excl = True
    for b in np.nonzero(has)[0]:
        r0, c0 = window_origin(env.sal_pos[b, 0], env.sal_pos[b, 1], env.w[b], env.n[b])
        ok_excl &= not in_window(env.target[b, 0], env.target[b, 1], r0, c0, env.w[b])
    check("capture window never contains the target", ok_excl)
    # choose a cell far from the salient one so capture is possible
    G = cfg.G
    a = np.zeros(3000, int)
    for b in range(3000):
        n = env.n[b]
        cand = [(r, c) for r in range(n) for c in range(n)
                if abs(r - env.sal_pos[b, 0]) > 2 or abs(c - env.sal_pos[b, 1]) > 2]
        r, c = cand[env.rng.integers(len(cand))]
        a[b] = r * G + c
    obs, rew, done, info = env.step(a)
    rate = info["captured"][has].mean()
    check(f"capture rate {rate:.3f}, want ~0.20", abs(rate - 0.2) < 0.03)
    cap = info["captured"]
    check("captured: actual window differs from intended", (info["actual"][cap] != info["intended"][cap]).any(1).all())
    check("not captured: actual window equals intended", (info["actual"][~cap] == info["intended"][~cap]).all())
    sr0, sc0 = window_origin(env.sal_pos[:, 0], env.sal_pos[:, 1], env.w, env.n)
    check("captured: actual window is on the salient distractor",
          ((info["actual"][cap, 0] == sr0[cap]) & (info["actual"][cap, 1] == sc0[cap])).all())
    check("capture never finds the target", not info["hit"][cap].any())
    # forced capture for H5
    env.reset(500)
    has = env.sal_pos[:, 0] >= 0
    a = np.zeros(500, int)
    ir0, ic0 = window_origin(a // G, a % G, env.w, env.n)
    sal_in = has & in_window(env.sal_pos[:, 0], env.sal_pos[:, 1], ir0, ic0, env.w)
    _, _, _, info = env.step(a, force_capture=np.ones(500, bool))
    check("force_capture=True captures exactly when a salient one is outside the chosen window",
          np.array_equal(info["captured"], has & ~sal_in))
    env.reset(500)
    _, _, _, info = env.step(a, force_capture=np.zeros(500, bool))
    check("force_capture=False never captures", not info["captured"].any())
    # finished episodes are untouched by forced capture
    env.reset(300)
    a = env.target[:, 0] * G + env.target[:, 1]
    env.step(a, force_capture=np.zeros(300, bool))   # all found
    assert env.done.all()
    v = env.visits.copy()
    _, r, _, info = env.step(a, force_capture=np.ones(300, bool))
    check("finished episodes: no capture, no reward, no visits",
          not info["captured"].any() and (r == 0).all() and np.array_equal(v, env.visits))


def test_reward_and_budget():
    cfg = EnvConfig(p_salient=0.0)
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
    cfg = EnvConfig(grid_sizes=(6, 9), p_salient=0.0)
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
              test_reward_and_budget, test_clamp_and_k_limit, test_reproducible, test_agents_order]:
        f()
    print(f"\n{len(FAILS)} failed")
    raise SystemExit(1 if FAILS else 0)
