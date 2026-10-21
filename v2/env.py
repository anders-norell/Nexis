"""Nexis v2 search environment (sprint 1).

Visual search on a grid, batched in NumPy. One target among distractors.

What the agent gets each step
-----------------------------
cue        (B, C+S)          which colour and shape is sought (one-hot each)
periphery  (B, G, G, 3+C)    whole grid, coarse: valid cell, occupied, salient, colour
                              (colour is blanked at cue level "none"). Never shape.
fovea      (B, W, W, 2+C+S)  egocentric patch of what the spotlight actually covers:
                              in-window flag, occupied, colour, shape. It carries no
                              coordinates: where the spotlight is must be remembered or
                              inferred, which is what the schema is for.

Action: a cell index (row * G + col) that the spotlight is centred on. The window has a
fixed size per episode (no zoom in the MVP) and is shifted to stay inside the grid.

efference  (B, G*G)            copy of the agent's previous command (one-hot over cells; zero
                              at the first step). Biological corollary discharge: the agent
                              knows what it aimed at, not where attention landed. Given to all
                              conditions (cfg.efference_copy).

Capture (bottom-up, graded): 0-3 distractors per episode are salient, each with a salience
s in [s_min, 1]. The periphery shows salience with fresh Gaussian noise every step, on every
cell, so the agent senses salience but cannot read it exactly. On each step every salient
object outside the intended window competes to pull the spotlight, with probability
p_capture * s (total capped at capture_cap); the winner is drawn in proportion to salience.
Which object captures, and when, is therefore not known in advance. Intended and actual
windows then differ (needed for H5). A capture that happens to cover the target finds it,
as it would in an animal.

Reward: +1 when the actual window covers the target (episode ends), -step_cost per step
otherwise. The episode also ends when the step budget is used up. Revisits are allowed but
cost steps like any other step.

Episode parameters (grid size, window, distractor count, budget, salience) are drawn at
random per episode, so the agent cannot memorise them.
"""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class EnvConfig:
    grid_sizes: tuple = (6, 7, 8, 9)          # n, grid is n x n
    windows: tuple = (1, 2)                   # w, spotlight is w x w (fixed within episode)
    n_distractors: tuple = (4, 16)            # inclusive range for k
    budget_frac: tuple = (0.6, 1.2)           # budget = ceil(frac * (k+1)), at least min_budget
    min_budget: int = 3
    n_colors: int = 4
    n_shapes: int = 4
    cue_level: str = "weak"                   # "none" | "weak" | "strong"
    share_color: dict = field(default_factory=lambda: {"none": 0.75, "weak": 0.75, "strong": 0.25})
    prior_p: float = 0.4                      # chance target is put in the hot quadrant
    n_salient: tuple = (0, 1, 2, 3)           # salient distractors per episode, drawn uniformly
    s_min: float = 0.3                        # salience drawn from U(s_min, 1)
    sal_noise: float = 0.15                   # SD of the noise on the periphery salience channel
    p_capture: float = 0.25                   # capture chance per step per unit salience
    capture_cap: float = 0.6                  # max total capture chance per step
    efference_copy: bool = True
    step_cost: float = 0.02

    @property
    def G(self):
        return max(self.grid_sizes)

    @property
    def W(self):
        return max(self.windows)


def window_origin(center_r, center_c, w, n):
    """Top-left corner of a w x w window centred on (r, c), shifted inside an n x n grid."""
    r0 = np.clip(center_r - (w - 1) // 2, 0, n - w)
    c0 = np.clip(center_c - (w - 1) // 2, 0, n - w)
    return r0, c0


def in_window(r, c, r0, c0, w):
    return (r >= r0) & (r < r0 + w) & (c >= c0) & (c < c0 + w)


class SearchEnv:
    def __init__(self, cfg: EnvConfig = None, seed: int = 0):
        self.cfg = cfg or EnvConfig()
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ reset
    def reset(self, B, overrides=None):
        """Start B episodes. `overrides` may fix n, w, k or budget (int) for analysis runs."""
        cfg, rng, G = self.cfg, self.rng, self.cfg.G
        ov = overrides or {}
        self.B = B
        self.n = np.full(B, ov["n"]) if "n" in ov else rng.choice(cfg.grid_sizes, B)
        self.w = np.full(B, ov["w"]) if "w" in ov else rng.choice(cfg.windows, B)
        lo, hi = cfg.n_distractors
        self.k = np.full(B, ov["k"]) if "k" in ov else rng.integers(lo, hi + 1, B)
        self.k = np.minimum(self.k, self.n * self.n - 1)
        if "budget" in ov:
            self.budget = np.full(B, ov["budget"])
        else:
            frac = rng.uniform(*cfg.budget_frac, B)
            self.budget = np.maximum(cfg.min_budget, np.ceil(frac * (self.k + 1))).astype(int)

        self.color = np.full((B, G, G), -1)
        self.shape = np.full((B, G, G), -1)
        self.valid = np.zeros((B, G, G), bool)
        self.salient = np.zeros((B, G, G))
        self.target = np.zeros((B, 2), int)
        self.target_color = rng.integers(0, cfg.n_colors, B)
        self.target_shape = rng.integers(0, cfg.n_shapes, B)
        M = max(1, max(cfg.n_salient))
        self.sal_pos = np.full((B, M, 2), -1)
        self.sal_val = np.zeros((B, M))
        p_share = cfg.share_color[cfg.cue_level]

        for b in range(B):
            n, w, k = self.n[b], self.w[b], self.k[b]
            self.valid[b, :n, :n] = True
            # target position: hot quadrant (top-left) with prob prior_p, else anywhere
            h = (n + 1) // 2
            if rng.random() < cfg.prior_p:
                tr, tc = rng.integers(0, h), rng.integers(0, h)
            else:
                tr, tc = rng.integers(0, n), rng.integers(0, n)
            self.target[b] = tr, tc
            self.color[b, tr, tc] = self.target_color[b]
            self.shape[b, tr, tc] = self.target_shape[b]
            # distractors
            free = np.array([i for i in range(n * n) if i != tr * n + tc])
            cells = rng.choice(free, k, replace=False)
            for i in cells:
                r, c = divmod(int(i), n)
                if rng.random() < p_share:   # same colour, other shape (conjunction)
                    col = self.target_color[b]
                    shp = rng.choice([s for s in range(cfg.n_shapes) if s != self.target_shape[b]])
                else:                        # other colour, any shape
                    col = rng.choice([x for x in range(cfg.n_colors) if x != self.target_color[b]])
                    shp = rng.integers(0, cfg.n_shapes)
                self.color[b, r, c], self.shape[b, r, c] = col, shp
            # salient distractors (bottom-up onsets), graded salience
            m = min(int(rng.choice(cfg.n_salient)), k)
            for j, i in enumerate(rng.choice(cells, m, replace=False)):
                r, c = divmod(int(i), n)
                self.sal_pos[b, j] = r, c
                self.sal_val[b, j] = rng.uniform(cfg.s_min, 1.0)
                self.salient[b, r, c] = self.sal_val[b, j]

        self.t = 0
        self.steps = np.zeros(B, int)
        self.done = np.zeros(B, bool)
        self.found = np.zeros(B, bool)
        self.visits = np.zeros((B, G, G), int)          # actual coverage counts
        self.last_actual = np.full((B, 2), -1)          # actual window origin, last step
        self.last_window = np.zeros((B, cfg.W, cfg.W, 2 + cfg.n_colors + cfg.n_shapes))
        self.last_intended = np.zeros((B, G * G))
        self._new_sal_noise()
        return self.observe()

    def _new_sal_noise(self):
        self.sal_obs = (self.salient + self.rng.normal(0, self.cfg.sal_noise, self.salient.shape)) * self.valid

    # -------------------------------------------------------------- observe
    def periphery(self):
        cfg = self.cfg
        occ = self.color >= 0
        col = np.zeros(self.color.shape + (cfg.n_colors,))
        b, r, c = np.nonzero(occ)
        col[b, r, c, self.color[b, r, c]] = 1.0
        if cfg.cue_level == "none":
            col[:] = 0.0
        return np.concatenate(
            [self.valid[..., None], occ[..., None], self.sal_obs[..., None], col], -1
        ).astype(float)

    def cue(self):
        cfg = self.cfg
        out = np.zeros((self.B, cfg.n_colors + cfg.n_shapes))
        out[np.arange(self.B), self.target_color] = 1
        out[np.arange(self.B), cfg.n_colors + self.target_shape] = 1
        return out

    def observe(self):
        eff = self.last_intended.copy() if self.cfg.efference_copy else np.zeros_like(self.last_intended)
        return dict(cue=self.cue(), periphery=self.periphery(), fovea=self.last_window.copy(),
                    efference=eff, action_mask=self.valid.reshape(self.B, -1).copy())

    def _fovea(self, b, r0, c0, w):
        cfg = self.cfg
        patch = np.zeros((cfg.W, cfg.W, 2 + cfg.n_colors + cfg.n_shapes))
        for i in range(w):
            for j in range(w):
                patch[i, j, 0] = 1.0
                col, shp = self.color[b, r0 + i, c0 + j], self.shape[b, r0 + i, c0 + j]
                if col >= 0:
                    patch[i, j, 1] = 1.0
                    patch[i, j, 2 + col] = 1.0
                    patch[i, j, 2 + cfg.n_colors + shp] = 1.0
        return patch

    # ----------------------------------------------------------------- step
    def step(self, action, force_capture=None):
        """action: (B,) cell indices on the padded G x G grid.

        force_capture: optional (B,) bool to force (True) or forbid (False) capture,
        for the H5 test. Forced capture picks among eligible salient objects in proportion
        to salience. None = random capture.
        Returns obs, reward, done, info. Finished episodes ignore actions.
        """
        cfg, G, B = self.cfg, self.cfg.G, self.B
        action = np.asarray(action)
        ar, ac = np.divmod(action, G)
        ar = np.minimum(ar, self.n - 1)
        ac = np.minimum(ac, self.n - 1)
        ir0, ic0 = window_origin(ar, ac, self.w, self.n)

        # bottom-up capture: salient objects outside the intended window compete
        M = self.sal_pos.shape[1]
        exists = self.sal_pos[..., 0] >= 0                                  # (B, M)
        inside = in_window(self.sal_pos[..., 0], self.sal_pos[..., 1],
                           ir0[:, None], ic0[:, None], self.w[:, None])
        eligible = exists & ~inside & ~self.done[:, None]
        p = cfg.p_capture * self.sal_val * eligible
        tot = p.sum(1)
        scale = np.where(tot > cfg.capture_cap, cfg.capture_cap / np.maximum(tot, 1e-12), 1.0)
        p = p * scale[:, None]
        if force_capture is None:
            u = self.rng.random(B)
            cum = np.cumsum(p, 1)
            who = np.where(u < cum[:, -1], (u[:, None] >= cum).sum(1), -1)
        else:
            fc = np.asarray(force_capture, bool)
            q = self.sal_val * eligible
            u = self.rng.random(B) * q.sum(1)
            who = np.where(fc & (q.sum(1) > 0), (u[:, None] >= np.cumsum(q, 1)).sum(1), -1)
            who = np.minimum(who, M - 1)
        cap = who >= 0
        wi = np.maximum(who, 0)
        spos = self.sal_pos[np.arange(B), wi]
        sr0, sc0 = window_origin(spos[:, 0], spos[:, 1], self.w, self.n)
        r0 = np.where(cap, sr0, ir0)
        c0 = np.where(cap, sc0, ic0)

        active = ~self.done
        hit = active & in_window(self.target[:, 0], self.target[:, 1], r0, c0, self.w)
        reward = np.where(active, np.where(hit, 1.0, -cfg.step_cost), 0.0)

        for b in np.nonzero(active)[0]:
            w = self.w[b]
            self.visits[b, r0[b]:r0[b] + w, c0[b]:c0[b] + w] += 1
            self.last_window[b] = self._fovea(b, r0[b], c0[b], w)
        self.last_actual = np.where(active[:, None], np.stack([r0, c0], 1), self.last_actual)
        onehot = np.zeros((B, G * G))
        onehot[np.arange(B), ar * G + ac] = 1.0
        self.last_intended = np.where(active[:, None], onehot, self.last_intended)
        self._new_sal_noise()

        self.steps += active
        self.found |= hit
        self.done |= hit | (self.steps >= self.budget)
        self.t += 1
        info = dict(intended=np.stack([ir0, ic0], 1), actual=np.stack([r0, c0], 1),
                    captured=cap, capturer=who, hit=hit, active=active)
        return self.observe(), reward, self.done.copy(), info
