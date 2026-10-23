"""Sprint 5a: schema module (1b) with full input and an information cost on its inner representation.

Approved design (spec, "Sprint 5: H5", plan 5a, step 2, product owner 23 Oct):

  input     the module sees everything the policy sees (periphery, cue, fovea remapped through the
            efference copy, position, time) plus the matching signal (agent_rl.matching_signal).
            It gets no schema and no gradient from the policy.
  network   the same per-cell GRU as the policy (PriorityMapAgent), separate weights.
  channel   what the module hands on passes a noisy channel, one number per cell:
                mu = h . w_mu + b_mu,   log sigma = h . w_ls + b_ls,   z = mu + sigma * eps,  eps ~ N(0, 1)
            The information sent is the KL divergence from N(mu, sigma^2) to N(0, 1), summed over the grid
            cells, in bits. It costs beta per bit and step in training.
  decoder   the schema map is computed only from z:  logit_c = w_d * z_c + b_d  (softmax over valid cells).
            The efference copy cannot go round the bottleneck.
  target    the window the spotlight actually covered last step (as 1b in sprint 3), cross-entropy.

Alternative E (product owner, 23 Oct 14:03): with alarm=True the module also gets the global alarm
(agent_rl.alarm_signal), broadcast to every cell.

The schema the policy receives is [map, running union of maps], with the same timing as sprint 3's module.
Gradients are hand-written and checked against finite differences in test_schema_channel.py.
"""
import numpy as np
from agent_rl import PriorityMapAgent

LN2 = np.log(2.0)


class ChannelSchema:
    def __init__(self, cfg, d=16, dh=32, seed=0, dtype=np.float32, alarm=False):
        self.cfg, self.d, self.dtype = cfg, d, dtype
        self.net = PriorityMapAgent(cfg, d=d, dh=dh, seed=seed, schema_mode="off", matching=True, dtype=dtype,
                                    alarm=alarm)
        rng = np.random.default_rng(seed + 777)
        self.p = self.net.p                       # one dict: network weights + channel weights
        self.p.update(dict(
            c_wmu=(rng.normal(0, 1 / np.sqrt(d), d)).astype(dtype), c_bmu=np.zeros(1, dtype),
            c_wls=(rng.normal(0, 0.1 / np.sqrt(d), d)).astype(dtype), c_bls=np.full(1, -1.0, dtype),
            c_wd=np.ones(1, dtype), c_bd=np.zeros(1, dtype)))
        self.beta = 0.0                           # cost per bit and step (set by the trainer)

    # ------------------------------------------------------------------ forward
    def init_state(self, B):
        return self.net.init_state(B)

    def features(self, env, obs, t):
        return self.net.features(env, obs, t, None)

    def forward(self, x, m, hp, eps):
        """x (B,N,dx), m (B,N), hp (B,N,d), eps (B,N). Returns h, logp (B,N), bits (B,), cache."""
        p = self.p
        h, _, _, _, ncache = self.net.step(x, m, hp)
        mu = h @ p["c_wmu"] + p["c_bmu"][0]
        ls = h @ p["c_wls"] + p["c_bls"][0]
        ls = np.clip(ls, -12.0, 4.0)
        sig = np.exp(ls)
        z = mu + sig * eps
        logit = p["c_wd"][0] * z + p["c_bd"][0] + (m - 1) * 1e9
        logit = logit - logit.max(1, keepdims=True)
        logp = logit - np.log(np.exp(logit).sum(1, keepdims=True))
        kl = 0.5 * (mu * mu + sig * sig - 1.0 - 2.0 * ls) * m           # nats per cell
        bits = kl.sum(1) / LN2
        cache = (ncache, h, mu, ls, sig, eps, z, m)
        return h, logp, bits, cache

    # ----------------------------------------------------------------- backward
    def backward(self, cache, logp, q, ok, scale, dh_future, grads):
        """One step. q (B,N) target distribution, ok (B,) rows with a target, scale = 1 / number of
        rows with targets in the whole rollout. Loss = CE + beta * bits, both summed over rows and
        multiplied by scale. Returns d hp for the previous step."""
        p = self.p
        ncache, h, mu, ls, sig, eps, z, m = cache
        B, N = mu.shape
        okf = ok.astype(self.dtype)[:, None]
        pr = np.exp(logp)
        dlogit = (pr - q) * okf * scale                                  # d CE / d logit
        grads["c_wd"] += np.array([(dlogit * z).sum()], self.dtype)
        grads["c_bd"] += np.array([dlogit.sum()], self.dtype)
        dz = dlogit * p["c_wd"][0]
        cb = self.beta / LN2 * scale * okf * m                           # d (beta*bits) / d KL_cell
        dmu = dz + cb * mu
        clipped = (ls <= -12.0) | (ls >= 4.0)
        dls = (dz * sig * eps + cb * (sig * sig - 1.0)) * (~clipped)
        grads["c_wmu"] += np.einsum("bnk,bn->k", h, dmu).astype(self.dtype)
        grads["c_bmu"] += np.array([dmu.sum()], self.dtype)
        grads["c_wls"] += np.einsum("bnk,bn->k", h, dls).astype(self.dtype)
        grads["c_bls"] += np.array([dls.sum()], self.dtype)
        dh = dh_future + dmu[..., None] * p["c_wmu"] + dls[..., None] * p["c_wls"]
        zero = np.zeros((B, N), self.dtype)
        return self.net.step_backward(ncache, zero, np.zeros(B, self.dtype), zero, dh.astype(self.dtype), grads)

    def zero_grads(self):
        return {k: np.zeros_like(v) for k, v in self.p.items()}


def channel_grads(module, mrecs):
    """BPTT over a rollout. mrecs: per step dict(idx, cache, logp, target or None, bits).
    Returns grads, mean CE, mean bits (over rows with targets)."""
    grads = module.zero_grads()
    if not mrecs:
        return grads, 0.0, 0.0
    G2, d = module.cfg.G ** 2, module.d
    Bfull = max(int(r["idx"].max()) for r in mrecs) + 1
    n_tot = sum(int((r["target"].sum(1) > 0).sum()) for r in mrecs if r["target"] is not None)
    n_tot = max(n_tot, 1)
    scale = 1.0 / n_tot
    dh = np.zeros((Bfull, G2, d), module.dtype)
    ce, bits = 0.0, 0.0
    for rc in reversed(mrecs):
        idx = rc["idx"]
        nb = len(idx)
        if rc["target"] is not None:
            tgt = rc["target"]
            z = tgt.sum(1, keepdims=True)
            ok = z[:, 0] > 0
            q = np.where(z > 0, tgt / np.maximum(z, 1e-12), 0.0)
            ce += -(q * np.where(q > 0, rc["logp"], 0)).sum(1)[ok].sum()
            bits += rc["bits"][ok].sum()
        else:
            ok = np.zeros(nb, bool)
            q = np.zeros((nb, G2))
        dhs = module.backward(rc["cache"], rc["logp"], q.astype(module.dtype), ok, scale, dh[idx], grads)
        dh = np.zeros_like(dh)
        dh[idx] = dhs
    return grads, ce / n_tot, bits / n_tot
