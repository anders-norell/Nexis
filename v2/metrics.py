"""Episode metrics shared by the sprint 2 gate evaluation (reference agents and RL agents).

run_metrics drives any agent that has reset(env), act(env, obs) and observe(env, info)
and returns per-episode arrays:

found      target found within budget
steps      steps used
ret        episode return (+1 on hit, -step_cost per non-hit step)
revisits   steps whose actual window covered no cell that had not been covered before
aim_repeats commands whose intended window lies wholly inside earlier intended windows
           (memory signature: an agent that remembers where it aimed seldom re-aims there)
n_capture  number of capture events
first_tc   first command aimed at a cell with the target's colour (periphery colour)
"""
import numpy as np


def run_metrics(env, agent, B, overrides=None):
    cfg = env.cfg
    obs = env.reset(B, overrides)
    agent.reset(env)
    G = cfg.G
    covered = np.zeros((B, G, G), bool)
    revisits = np.zeros(B, int)
    aimed = np.zeros((B, G, G), bool)
    aim_rep = np.zeros(B, int)
    n_cap = np.zeros(B, int)
    ret = np.zeros(B)
    first_tc = np.zeros(B, bool)
    t = 0
    target_col_map = env.color == env.target_color[:, None, None]
    while not env.done.all():
        a = agent.act(env, obs)
        if t == 0:
            r, c = np.divmod(np.asarray(a), G)
            first_tc = target_col_map[np.arange(B), r, c]
        obs, rew, done, info = env.step(a)
        if hasattr(agent, "observe"):
            agent.observe(env, info)
        ret += rew
        n_cap += info["captured"]
        r0, c0 = info["actual"].T
        i0, j0 = info["intended"].T
        for b in np.nonzero(info["active"])[0]:
            w = env.w[b]
            if aimed[b, i0[b]:i0[b] + w, j0[b]:j0[b] + w].all():
                aim_rep[b] += 1
            aimed[b, i0[b]:i0[b] + w, j0[b]:j0[b] + w] = True
            sl = covered[b, r0[b]:r0[b] + w, c0[b]:c0[b] + w]
            if sl.all():
                revisits[b] += 1
            covered[b, r0[b]:r0[b] + w, c0[b]:c0[b] + w] = True
        t += 1
    return dict(found=env.found.copy(), steps=env.steps.copy(), ret=ret, revisits=revisits, aim_repeats=aim_rep,
                n_capture=n_cap, first_tc=first_tc, k=env.k.copy(), n=env.n.copy(),
                w=env.w.copy(), budget=env.budget.copy())


def summarise(m):
    steps = m["steps"].sum()
    return dict(found=float(m["found"].mean()),
                found_se=float(m["found"].std() / np.sqrt(len(m["found"]))),
                ret=float(m["ret"].mean()),
                revisit_share=float(m["revisits"].sum() / max(steps, 1)),
                aim_repeat_share=float(m["aim_repeats"].sum() / max(steps, 1)),
                first_on_target_colour=float(m["first_tc"].mean()),
                mean_steps=float(m["steps"].mean()),
                captures_per_ep=float(m["n_capture"].mean()))
