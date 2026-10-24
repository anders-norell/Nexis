"""Sprint 6 (1c): a schema module trained only by the policy's gradient through the channel.

Approved plan (spec, "Sprint 6: 1c ... - plan") with the product owner's conditions of 24 Oct 23:45, which
take precedence where they differ:

  module    the same per-cell GRU as the policy (PriorityMapAgent), separate weights, and the same noisy
            channel as 1b: one number per cell,
                mu = h . w_mu + b_mu,  log sigma = h . w_ls + b_ls,  z = (mu + sigma * eps) * valid
            No decoder, no target, no loss of its own. z is the policy's schema input (first schema column;
            the second column stays 0, so the policy has exactly the baseline's parameters).
  training  the policy's A2C loss (actor + entropy + critic) is backpropagated through z into the module
            (reparameterisation: dz/dmu = 1, dz/dlog sigma = sigma * eps). The information cost is a direct
            penalty beta * bits on the module, a capacity limit with no information about content; beta is
            regulated towards a target in bits per step as in 5a. A cost subtracted from the reward would
            give the module no gradient (a term that does not depend on the action adds nothing to the policy
            gradient in expectation), so it is not done that way.
  timing    at step t the module sees the observation of step t and sends z_t, which the policy reads at the
            same step (as 1b's module). At t = 0 the channel is closed (z = 0, no bits), as in 1b.
  alarm     "without" (alarm=False): the module's input is exactly the tensor the policy gets
            (PriorityMapAgent.features with no schema), nothing derived. "with" (alarm=True): exactly
            1b-E's input (matching signal + global alarm, as ChannelSchema(alarm=True)).
  capacity  shuffle=True (matched capacity baseline, condition (b)): the same module and channel, but from
            the start its input is another episode's (the donor's) observation stream. Same parameters,
            same channel width, same cost; the channel carries nothing about the own episode.

Gradients are hand-written and checked against finite differences in test_1c.py.
"""
import numpy as np
from agent_rl import PriorityMapAgent

LN2 = np.log(2.0)
LS_MIN, LS_MAX = -12.0, 4.0


class Policy1c(PriorityMapAgent):
    """The baseline policy (schema mode 'off' features, two schema columns) with a backward pass that also
    returns the gradient with respect to the first schema column of its input."""

    def __init__(self, cfg, d=16, dh=32, seed=0, dtype=np.float32):
        super().__init__(cfg, d=d, dh=dh, seed=seed, schema_mode="off", dtype=dtype)
        self.zcol = self.dx - self.n_schema          # column of the channel value in x

    def step_backward_x(self, cache, dlogit, dV, dslog, dh, grads):
        """As PriorityMapAgent.step_backward, and also d loss / d x[..., zcol] (B, N)."""
        p, d, dx = self.p, self.d, self.dx
        x, m, hf, cnt, g, r, u, rh, n, h, H, a, pooled = cache
        B, N, _ = x.shape
        dlf = dlogit.reshape(-1)
        grads["w2"] += a.T @ dlf
        grads["wv"] += pooled.T @ dV
        grads["bv"] += dV.sum()
        dpool = (dV / cnt)[:, None] * p["wv"]
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
        dAb = dA.reshape(B, N, 3 * d).sum(1)
        grads["Wz"][dx:] += g.T @ dAb
        grads["bz"] += dAb.sum(0)
        dg = dAb @ p["Wz"][dx:].T
        dhp = dhp.reshape(B, N, d) + ((dg / cnt[:, None])[:, None, :] * m[..., None])
        k = self.zcol
        dz = (dA @ p["Wz"][k] + dpre @ p["W1"][k]).reshape(B, N)
        return dhp, dz


class Module1c:
    """Schema module of 1c: network + noisy channel, no decoder and no target."""

    def __init__(self, cfg, d=16, dh=32, seed=0, alarm=False, shuffle=False, dtype=np.float32):
        self.cfg, self.d, self.dtype = cfg, d, dtype
        self.alarm, self.shuffle = alarm, shuffle
        self.net = PriorityMapAgent(cfg, d=d, dh=dh, seed=seed, schema_mode="off", matching=alarm, alarm=alarm,
                                    dtype=dtype)
        rng = np.random.default_rng(seed + 777)
        self.p = self.net.p                        # one dict: network + channel weights
        self.p.update(dict(
            c_wmu=(rng.normal(0, 1 / np.sqrt(d), d)).astype(dtype), c_bmu=np.zeros(1, dtype),
            c_wls=(rng.normal(0, 0.1 / np.sqrt(d), d)).astype(dtype), c_bls=np.full(1, -1.0, dtype)))
        self.beta = 0.0

    def init_state(self, B):
        return self.net.init_state(B)

    def features(self, env, obs, t):
        """'without': PriorityMapAgent.features with no schema, the policy's own tensor with the two schema
        columns at 0. 'with': the same plus matching signal and alarm, i.e. 1b-E's input."""
        return self.net.features(env, obs, t, None)

    def forward(self, x, m, hp, eps, open_=True):
        """x (B,N,dx), m (B,N), hp (B,N,d), eps (B,N). Returns h, z (B,N), bits (B,), cache.
        With open_=False (t = 0) the channel sends nothing: z = 0, bits = 0, no gradient through it."""
        p = self.p
        h, _, _, _, ncache = self.net.step(x, m, hp)
        mu = h @ p["c_wmu"] + p["c_bmu"][0]
        ls_raw = h @ p["c_wls"] + p["c_bls"][0]
        ls = np.clip(ls_raw, LS_MIN, LS_MAX)
        sig = np.exp(ls)
        if open_:
            z = (mu + sig * eps) * m
            bits = (0.5 * (mu * mu + sig * sig - 1.0 - 2.0 * ls) * m).sum(1) / LN2
        else:
            z = np.zeros_like(mu)
            bits = np.zeros(mu.shape[0], mu.dtype)
        cache = (ncache, h, mu, ls_raw, ls, sig, eps, m, open_)
        return h, z, bits, cache

    def backward(self, cache, dz, cost_scale, dh_future, grads):
        """dz (B,N): gradient of the policy's loss w.r.t. z. Adds the cost beta * bits * cost_scale per row.
        Returns d hp for the previous step."""
        p = self.p
        ncache, h, mu, ls_raw, ls, sig, eps, m, open_ = cache
        B, N = mu.shape
        if open_:
            cb = self.beta / LN2 * cost_scale * m
            dmu = dz * m + cb * mu
            clipped = (ls_raw < LS_MIN) | (ls_raw > LS_MAX)
            dls = (dz * m * sig * eps + cb * (sig * sig - 1.0)) * (~clipped)
        else:
            dmu = np.zeros((B, N), self.dtype)
            dls = np.zeros((B, N), self.dtype)
        grads["c_wmu"] += np.einsum("bnk,bn->k", h, dmu).astype(self.dtype)
        grads["c_bmu"] += np.array([dmu.sum()], self.dtype)
        grads["c_wls"] += np.einsum("bnk,bn->k", h, dls).astype(self.dtype)
        grads["c_bls"] += np.array([dls.sum()], self.dtype)
        dh = dh_future + dmu[..., None] * p["c_wmu"] + dls[..., None] * p["c_wls"]
        zero = np.zeros((B, N), self.dtype)
        return self.net.step_backward(ncache, zero, np.zeros(B, self.dtype), zero, dh.astype(self.dtype), grads)

    def zero_grads(self):
        return {k: np.zeros_like(v) for k, v in self.p.items()}

    def n_params(self):
        return int(sum(v.size for v in self.p.values()))


def make_agent_1c(cfg, d=16, dh=32, seed=0, alarm=False, shuffle=False, dtype=np.float32):
    """Policy (initialised exactly as the baseline with the same seed) carrying the 1c module."""
    ag = Policy1c(cfg, d=d, dh=dh, seed=seed, dtype=dtype)
    ag.module = Module1c(cfg, d=d, dh=dh, seed=10_000 + seed, alarm=alarm, shuffle=shuffle, dtype=dtype)
    ag.variant = "1c_cap" if shuffle else ("1c_e" if alarm else "1c")
    return ag


def schema_from_z(z, B, G):
    """Policy schema input (B,G,G,2): channel in column 0, column 1 zero."""
    s = np.zeros((B, G, G, 2))
    s[..., 0] = z.reshape(B, G, G)
    return s


def module_grads_1c(module, mrecs, dzs, cost_scale):
    """BPTT through the module over a rollout. mrecs[t]: dict(idx, cache, bits, open); dzs[t]: (len(idx), N)
    gradient of the policy loss w.r.t. z_t (zero for steps where it was not used). Returns grads."""
    grads = module.zero_grads()
    if not mrecs:
        return grads
    G2, d = module.cfg.G ** 2, module.d
    Bfull = max(int(r["idx"].max()) for r in mrecs) + 1
    dh = np.zeros((Bfull, G2, d), module.dtype)
    for rc, dz in zip(reversed(mrecs), reversed(dzs)):
        idx = rc["idx"]
        dhs = module.backward(rc["cache"], dz.astype(module.dtype), cost_scale, dh[idx], grads)
        dh = np.zeros_like(dh)
        dh[idx] = dhs
    return grads
