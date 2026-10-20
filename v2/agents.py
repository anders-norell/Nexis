"""Hand-coded reference agents for the v2 search environment.

None of these learn. They bracket what a learned agent should achieve in sprint 2:

RandomAgent    uniform over valid cells, no memory. The floor.
ItemScanAgent  visits occupied cells (seen in the periphery) in random order, skipping cells
               it believes it has covered. Ignores colour: optimal-ish at cue level "none".
GuidedAgent    like ItemScan but only target-coloured cells, hot quadrant first.

Scan and Guided keep track of covered cells in one of two ways:
  track="intended"  marks the window it chose (knows its commands, not where attention went)
  track="actual"    marks the window attention really covered (a perfect attention schema)
Under capture the two differ; the gap is a first, hand-coded hint of what knowing your
own attention is worth in this task.
"""
import numpy as np
from env import window_origin


class RandomAgent:
    def __init__(self, rng):
        self.rng = rng

    def reset(self, env):
        pass

    def act(self, env, obs):
        mask = obs["action_mask"]
        u = self.rng.random(mask.shape) * mask
        return u.argmax(1)

    def observe(self, env, info):
        pass


class ItemScanAgent:
    use_colour = False
    use_prior = False

    def __init__(self, rng, track="intended"):
        self.rng, self.track = rng, track

    def reset(self, env):
        G = env.cfg.G
        self.covered = np.zeros((env.B, G, G), bool)
        self.order = self.rng.random((env.B, G, G))   # random tie-breaking per episode

    def _candidates(self, env, obs):
        per = obs["periphery"]
        cand = per[..., 1] > 0                        # occupied
        if self.use_colour and env.cfg.cue_level != "none":
            C = env.cfg.n_colors
            want = obs["cue"][:, :C]                  # (B, C)
            cand &= (per[..., 3:3 + C] * want[:, None, None, :]).sum(-1) > 0
        return cand

    def act(self, env, obs):
        G = env.cfg.G
        cand = self._candidates(env, obs) & ~self.covered
        score = self.order.copy()
        if self.use_prior:
            rr, cc = np.meshgrid(np.arange(G), np.arange(G), indexing="ij")
            h = ((env.n + 1) // 2)[:, None, None]
            score = score + 1.0 * ((rr < h) & (cc < h))
        score = np.where(cand, score + 10.0, np.where(env.valid & ~self.covered, score, -1.0))
        return score.reshape(env.B, -1).argmax(1)

    def observe(self, env, info):
        r0, c0 = (info["actual"] if self.track == "actual" else info["intended"]).T
        for b in np.nonzero(info["active"])[0]:
            w = env.w[b]
            self.covered[b, r0[b]:r0[b] + w, c0[b]:c0[b] + w] = True


class GuidedAgent(ItemScanAgent):
    use_colour = True
    use_prior = True


def run(env, agent, B, overrides=None, force_capture=None):
    """Run B episodes to completion. Returns per-episode arrays."""
    obs = env.reset(B, overrides)
    agent.reset(env)
    n_cap = np.zeros(B, int)
    while not env.done.all():
        a = agent.act(env, obs)
        fc = None if force_capture is None else force_capture(env)
        obs, r, done, info = env.step(a, fc)
        agent.observe(env, info)
        n_cap += info["captured"]
    return dict(found=env.found.copy(), steps=env.steps.copy(), k=env.k.copy(), n=env.n.copy(),
                w=env.w.copy(), budget=env.budget.copy(), salient=env.sal_pos[:, 0] >= 0,
                n_capture=n_cap)
