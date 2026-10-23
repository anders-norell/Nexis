"""Nexis v2 learning agent (sprint 2): a retinotopic priority map trained by reinforcement.

Architecture
------------
Every grid cell has its own small recurrent unit (a GRU, weights shared across cells), like a
retinotopic priority map (superior colliculus / FEF / LIP). Cells talk to each other only
through a global context: the mean of all cells' previous hidden states.

Per-cell input x_c (no gradient flows into it):
  periphery at c      valid, occupied, noisy salience, colour            (3 + C)
  cue                 what is sought (colour + shape one-hot)             (C + S)
  remapped fovea      what the fovea showed last step, placed on the cells of the *intended*
                      window via the efference copy (corollary-discharge remapping). Under
                      capture this lands on the wrong cells; the agent is not told.  (2 + C + S)
  position            row / n, col / n                                    (2)
  time                steps taken / 20                                    (1)
  schema              per-cell schema channel: [actual window last step, cells actually
                      covered so far]. All zeros in the baseline.         (2)
Plus the global context g (d).

Outputs:
  policy  one logit per cell (masked to the grid), next spotlight command
  value   critic, from the masked mean of the head activations
  1b head per-cell logit predicting where the spotlight actually was last step. Its loss weight
          is 0 in the baseline; condition 1b turns it on and feeds its prediction back into the
          schema channel.

Gradients are written by hand (NumPy only, PyTorch is unavailable) and checked against finite
differences in test_agent_rl.py.
"""
import numpy as np
from env import window_origin

SCHEMA_MODES = ("off", "fixed", "learned", "shuffled", "module", "channel")


def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def matching_signal(per, rem, C):
    """Sprint 5 matching signal per cell (B, G, G): on the cells of the intended window (where the efference
    copy remaps the fovea), +1 if what the fovea showed agrees with what the periphery shows there
    (occupied and colour; the periphery never shows shape), -1 if it does not, 0 elsewhere and at the first
    step. A -1 anywhere means the spotlight was not where the agent aimed: a comparator between expected
    and actual visual input."""
    inwin = rem[..., 0] > 0
    same = (rem[..., 1] == per[..., 1]) & np.all(rem[..., 2:2 + C] == per[..., 3:3 + C], -1)
    return np.where(inwin, np.where(same, 1.0, -1.0), 0.0)


class PriorityMapAgent:
    def __init__(self, cfg, d=16, dh=32, seed=0, schema_mode="off", dtype=np.float32, matching=False):
        assert schema_mode in SCHEMA_MODES
        self.cfg, self.d, self.dh, self.schema_mode = cfg, d, dh, schema_mode
        C, S = cfg.n_colors, cfg.n_shapes
        self.matching = matching          # sprint 5: one extra input channel, the matching signal (module only)
        self.dx = (3 + C) + (C + S) + (2 + C + S) + 2 + 1 + 2 + (1 if matching else 0)
        self.n_schema = 2
        din = self.dx + d
        rng = np.random.default_rng(seed)

        def init(*shape, fan_in):
            return rng.normal(0, 1 / np.sqrt(fan_in), shape)

        self.p = dict(
            Wz=init(din, 3 * d, fan_in=din), bz=np.zeros(3 * d),
            Ur=init(d, d, fan_in=d), Uu=init(d, d, fan_in=d), Un=init(d, d, fan_in=d),
            W1=init(self.dx + d, dh, fan_in=self.dx + d), b1=np.zeros(dh),
            w2=init(dh, fan_in=dh) * 0.1, wv=init(dh, fan_in=dh) * 0.1, bv=np.zeros(1),
            ws=init(d, fan_in=d), bs=np.zeros(1),
        )
        self.p["bz"][d:2 * d] = 1.0      # update gate biased open: remember by default
        self.dtype = dtype
        self.p = {k: v.astype(dtype) for k, v in self.p.items()}

    # ------------------------------------------------------------ features
    def features(self, env, obs, t, schema):
        """Per-cell input (B, N, dx) from the observation. `schema` is (B, G, G, 2) or None."""
        cfg = self.cfg
        B, G = env.B, cfg.G
        per = obs["periphery"]                                   # (B, G, G, 3+C)
        valid = per[..., 0] > 0
        cue = np.broadcast_to(obs["cue"][:, None, None, :], (B, G, G, obs["cue"].shape[1]))
        fov = obs["fovea"]                                       # (B, W, W, 2+C+S)
        rem = np.zeros((B, G, G, fov.shape[-1]))
        eff = obs["efference"]
        has = eff.sum(1) > 0
        if has.any():
            cell = eff.argmax(1)
            er, ec = np.divmod(cell, G)
            n = valid[:, :, 0].sum(1)                            # grid size known from periphery
            w = np.rint(np.sqrt(fov[..., 0].sum((1, 2)))).astype(int)   # window size from fovea
            er, ec = np.minimum(er, n - 1), np.minimum(ec, n - 1)
            r0, c0 = window_origin(er, ec, np.maximum(w, 1), n)
            for i in range(cfg.W):
                for j in range(cfg.W):
                    sel = has & (w > i) & (w > j)
                    b = np.nonzero(sel)[0]
                    rem[b, r0[b] + i, c0[b] + j] = fov[b, i, j]
        rr, cc = np.meshgrid(np.arange(G), np.arange(G), indexing="ij")
        n = np.maximum(valid[:, :, 0].sum(1), 1)[:, None, None]
        pos = np.stack([rr[None] / n, cc[None] / n], -1) * valid[..., None]
        tt = np.full((B, G, G, 1), t / 20.0)
        sch = np.zeros((B, G, G, self.n_schema)) if schema is None else schema
        parts = [per, cue, rem, pos, tt, sch]
        if self.matching:
            parts.append(matching_signal(per, rem, cfg.n_colors)[..., None])
        x = np.concatenate(parts, -1)
        return (x.reshape(B, G * G, self.dx).astype(self.dtype),
                valid.reshape(B, G * G).astype(self.dtype))

    # ------------------------------------------------------------- forward
    def step(self, x, m, hp):
        """One recurrent step. x (B,N,dx), m (B,N) mask, hp (B,N,d). Returns outputs and cache."""
        p, d, dx = self.p, self.d, self.dx
        B, N, _ = x.shape
        cnt = np.maximum(m.sum(1), 1.0)
        g = np.einsum("bnk,bn->bk", hp, m) / cnt[:, None]
        xf, hf = x.reshape(B * N, dx), hp.reshape(B * N, d)
        A = (xf @ p["Wz"][:dx]).reshape(B, N, 3 * d) + (g @ p["Wz"][dx:] + p["bz"])[:, None, :]
        A = A.reshape(B * N, 3 * d)
        HU = hf @ np.concatenate([p["Ur"], p["Uu"]], 1)
        r = sigmoid(A[:, :d] + HU[:, :d])
        u = sigmoid(A[:, d:2 * d] + HU[:, d:])
        rh = r * hf
        n = np.tanh(A[:, 2 * d:] + rh @ p["Un"])
        h = (1 - u) * n + u * hf
        H = np.concatenate([xf, h], 1)
        a = np.tanh(H @ p["W1"] + p["b1"])
        a3 = a.reshape(B, N, -1)
        logit = (a @ p["w2"]).reshape(B, N) + (m - 1) * 1e9
        logit = logit - logit.max(1, keepdims=True)
        logp = logit - np.log(np.exp(logit).sum(1, keepdims=True))
        pooled = np.einsum("bnk,bn->bk", a3, m) / cnt[:, None]
        V = pooled @ p["wv"] + p["bv"][0]
        slog = (h @ p["ws"]).reshape(B, N) + p["bs"][0] + (m - 1) * 1e9
        slog = slog - slog.max(1, keepdims=True)
        slogp = slog - np.log(np.exp(slog).sum(1, keepdims=True))
        cache = (x, m, hf, cnt, g, r, u, rh, n, h, H, a, pooled)
        return h.reshape(B, N, d), logp, V, slogp, cache

    # ------------------------------------------------------------ backward
    def step_backward(self, cache, dlogit, dV, dslog, dh, grads):
        """Backprop one step. dlogit/dslog are gradients w.r.t. the masked logits (B,N),
        dV (B,), dh (B,N,d) from the future. Accumulates into grads, returns d hp (B,N,d)."""
        p, d, dx = self.p, self.d, self.dx
        x, m, hf, cnt, g, r, u, rh, n, h, H, a, pooled = cache
        B, N, _ = x.shape
        dlf = dlogit.reshape(-1)
        grads["w2"] += a.T @ dlf
        grads["wv"] += pooled.T @ dV
        grads["bv"] += dV.sum()
        dpool = (dV / cnt)[:, None] * p["wv"]                       # (B, dh)
        da = dlf[:, None] * p["w2"] + (dpool[:, None, :] * m[..., None]).reshape(B * N, -1)
        dpre = da * (1 - a * a)
        grads["W1"] += H.T @ dpre
        grads["b1"] += dpre.sum(0)
        dH = dpre @ p["W1"][dx:].T
        dsf = dslog.reshape(-1)
        grads["ws"] += h.T @ dsf
        grads["bs"] += dsf.sum()
        dh = dh.reshape(B * N, d) + dH + dsf[:, None] * p["ws"]
        dn = dh * (1 - u)
        du = dh * (hf - n)
        dhp = dh * u
        dan = dn * (1 - n * n)
        grads["Un"] += rh.T @ dan
        drh = dan @ p["Un"].T
        dhp += drh * r
        dar = drh * hf * r * (1 - r)
        dau = du * u * (1 - u)
        grads["Ur"] += hf.T @ dar
        grads["Uu"] += hf.T @ dau
        dhp += dar @ p["Ur"].T + dau @ p["Uu"].T
        dA = np.concatenate([dar, dau, dan], 1)
        grads["Wz"][:dx] += x.reshape(B * N, dx).T @ dA
        dAb = dA.reshape(B, N, 3 * d).sum(1)                        # (B, 3d)
        grads["Wz"][dx:] += g.T @ dAb
        grads["bz"] += dAb.sum(0)
        dg = dAb @ p["Wz"][dx:].T                                    # (B, d)
        dhp = dhp.reshape(B, N, d) + ((dg / cnt[:, None])[:, None, :] * m[..., None])
        return dhp

    def zero_grads(self):
        return {k: np.zeros_like(v) for k, v in self.p.items()}

    def init_state(self, B):
        return np.zeros((B, self.cfg.G ** 2, self.d), self.dtype)

    # ---------------------------------------------------------- persistence
    def save(self, path, **meta):
        mod = getattr(self, "module", None)
        extra = {f"q_{k}": v for k, v in mod.p.items()} if mod is not None else {}
        np.savez(path, **{f"p_{k}": v for k, v in self.p.items()}, **extra,
                 **{f"m_{k}": np.asarray(v) for k, v in meta.items()})

    def load(self, path):
        z = np.load(path)
        for k in self.p:
            self.p[k] = z[f"p_{k}"]
        mod = getattr(self, "module", None)
        if mod is not None:
            for k in mod.p:
                mod.p[k] = z[f"q_{k}"]
        return {k[2:]: z[k] for k in z.files if k.startswith("m_")}


def make_agent(cfg, d=16, dh=32, seed=0, schema_mode="off"):
    """Policy agent; for schema mode 'module' (1b as a separate schema module, sprint 3) it carries
    a second network of the same kind whose only trained output is the 1b head. The module sees
    the same observation as the policy (no schema), predicts where the spotlight actually was last
    step, and its prediction (plus its running union) is the policy's schema channel at the same
    step, with the same timing as the fixed schema 1a. No gradient flows between the two."""
    ag = PriorityMapAgent(cfg, d=d, dh=dh, seed=seed, schema_mode=schema_mode)
    if schema_mode == "module":
        ag.module = PriorityMapAgent(cfg, d=d, dh=dh, seed=10_000 + seed, schema_mode="off")
    if schema_mode == "channel":
        from schema_channel import ChannelSchema
        ag.module = ChannelSchema(cfg, d=d, dh=dh, seed=10_000 + seed)
    return ag


def schema_input(mode, env, info, visited, learned_pred=None, donor=None):
    """Schema channel for the next step, (B, G, G, 2). Baseline: None (zeros).

    fixed     formula: actual window last step + cells actually covered so far (1a)
    learned   the agent's own 1b prediction of last step's window + its running union (1b)
    shuffled  the fixed schema of another episode in the batch, same size (Blandat)
    Returns (schema, visited).
    """
    if mode in ("off", "module", "channel"):            # module schema is built in the rollout
        return None, visited
    G = env.cfg.G
    B = env.B
    if mode == "learned":
        cur = learned_pred.reshape(B, G, G)
    else:
        cur = np.zeros((B, G, G))
        r0, c0 = info["actual"].T
        for b in np.nonzero(info["active"])[0]:
            w = env.w[b]
            cur[b, r0[b]:r0[b] + w, c0[b]:c0[b] + w] = 1.0
    vis = np.maximum(visited, cur)
    out = np.stack([cur, vis], -1)
    if mode == "shuffled":
        out = out[donor]
    return out, vis
