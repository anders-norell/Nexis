"""Sprint 6 (1c): the role-blind probe, its null model and the measures (product owner's conditions 24 Oct 23:45).

Read-out
--------
The agent is frozen. At every step t of a bank episode that is still running at t + 1, the candidate windows
for where the spotlight actually was at step t are: the intended window and the window of every salient object
eligible to capture (outside the intended window), duplicates merged. Steps with fewer than two candidates are
skipped. The channel is read at t + 1 (the module has then seen the glimpse of step t, as 1b's map was read in
5a) and at t (the step before): "t and t - 1" in the conditions.

The probe gives every candidate the same score function, computed only from the channel's values on that
candidate's cells: mean, max and min of z at t + 1 and at t, and the window mean minus the grid mean at both
times (8 features, standardised with the training set's mean and SD over all candidate rows). A small MLP
(8 -> 16 tanh -> 1) maps them to a score; a softmax over the event's candidates gives the probe's
distribution. It is never told which candidate is the intended window, and the candidate order plays no role
(tested). Trained by cross-entropy on the actual window, on its own bank (probe bank, seeds 7 000 000 + i),
with equally many steps with and without capture (the larger group subsampled at random, fixed seed).

H5 events (as in 5a): a capture at step t that does not find the target, episode still running at t + 1,
the fovea excludes the intended window by occupied and colour.

Null model (conditions 24 Oct): the restricted ideal observer with the probe's prior, uniform over the
candidates, weakened by mixing: in each of K draws, with probability q the glimpse is replaced by the
restricted content of a candidate drawn from the capture prior (the environment's capture probabilities with
the true salience, intended = no capture), as decided 24 Oct. With a uniform prior the 5a mixing of salience
values has no effect and is left out. q is matched so that the null's mean mass on the actual window equals
the probe's (reserve order of 5a not needed: both live on the same candidates; if no q matches, the run is
not assessable). Common random numbers per event (episode seed, step).

Measures
--------
  (c) blind:   mass on the actual window at H5 events minus the uniform 1 / K, > 2 SE (bootstrap over episodes)
  "om" locked: mass on the intended window ("no capture") at H5 events, probe minus null at matched q
  "var" locked, descriptive: among capture places (events with another salient candidate), the share on the
               actual window, sum pA / sum (pA + pO), probe and null
Blind mode reports only quantities about the actual window: mass, accuracy, (c), whether q can be matched
and the bar (the null's mass on the intended window at the matched q). The probe's mass on the intended
window or on the other places at capture steps is computed only by measure(), which refuses to run without
results/h5_1c_lock.json, written at gate 6a.
"""
import hashlib
import json
import os
import numpy as np
from env import window_origin, in_window
from bank import BankEnv, streams
from h5 import restricted, excludes_intended_restricted
from h5_eval import QS, solve_q, clip_q, interp, prior

HERE = os.path.dirname(os.path.abspath(__file__))
LOCK = os.path.join(HERE, "results", "h5_1c_lock.json")
PROBE_BANK = 7_000_000
N_PROBE = 4000
K_NULL = 400
N_BOOT = 1000
NF = 8
BLOCK = 500


# ---------------------------------------------------------------------------------------- channel runs
def _step_channel(agent, kind, env, obs, t, idx, hm, arng, donor):
    """The channel z_t (B, N) of the agent at step t (0 outside idx), and the policy's schema input."""
    G = agent.cfg.G
    B = env.B
    mod = agent.module
    eps = np.stack([arng[b].normal(0, 1, G * G) for b in idx]) if len(idx) else np.zeros((0, G * G))
    z = np.zeros((B, G * G))
    if kind == "1c":
        xm, mm = mod.features(env, obs, t)
        src = donor[idx] if mod.shuffle else idx
        hms, zz, bits, _ = mod.forward(xm[src], mm[src], hm[idx], eps, open_=t > 0)
        hm[idx] = hms
        z[idx] = zz
        z = z * obs["action_mask"]
        return z, bits, None
    # 1b (ChannelSchema): z from the cache; the policy gets [map, union] as in h5_bank
    xm, mm = mod.features(env, obs, t)
    hms, slm, bits, cache = mod.forward(xm[idx], mm[idx], hm[idx], eps)
    hm[idx] = hms
    if t > 0:
        z[idx] = cache[6] * mm[idx]
    else:
        bits = np.zeros(len(idx))
    return z, bits, slm


def candidates_at(env, b, ir0, ic0):
    """Unique candidate origins (intended first), capture prior over them (merged), list of salient j per
    candidate."""
    cfg, w = env.cfg, int(env.w[b])
    origins, js = [(int(ir0), int(ic0))], [[]]
    sal = []
    for j in range(env.sal_pos.shape[1]):
        r, c = env.sal_pos[b, j]
        if r < 0 or in_window(r, c, ir0, ic0, w):
            continue
        sr0, sc0 = window_origin(np.array([r]), np.array([c]), w, env.n[b])
        o = (int(sr0[0]), int(sc0[0]))
        sal.append(float(env.sal_val[b, j]))
        if o in origins:
            js[origins.index(o)].append(len(sal) - 1)
        else:
            origins.append(o)
            js.append([len(sal) - 1])
    pr_obj = prior(np.array(sal), cfg.p_capture, cfg.capture_cap) if sal else np.ones(1)
    pr = np.array([pr_obj[0]] + [sum(pr_obj[1 + i] for i in jj) for jj in js[1:]])
    return origins, pr


def window_features(Z1, Z0, origins, w, n):
    """(K, NF) features per candidate from the channel at t + 1 (Z1) and t (Z0), both (G, G)."""
    m1, m0 = Z1[:n, :n].mean(), Z0[:n, :n].mean()
    out = np.zeros((len(origins), NF))
    for k, (r0, c0) in enumerate(origins):
        a, b = Z1[r0:r0 + w, c0:c0 + w], Z0[r0:r0 + w, c0:c0 + w]
        out[k] = [a.mean(), a.max(), a.min(), b.mean(), b.max(), b.min(), a.mean() - m1, b.mean() - m0]
    return out


def collect(agent, kind, seeds, block=BLOCK, test_shuffle=False, with_events=True):
    """Run the frozen agent on bank episodes. kind: '1c', '1b' or 'off'. Returns found, bits split by
    whether the previous command was captured, and the probe's events."""
    out = dict(found=[], bits_sum=np.zeros(2), bits_n=np.zeros(2), events=[])
    for s in range(0, len(seeds), block):
        _block(agent, kind, seeds[s:s + block], out, test_shuffle, with_events)
    out["found"] = np.concatenate(out["found"])
    return out


def _block(agent, kind, seeds, out, test_shuffle, with_events):
    cfg = agent.cfg
    G = cfg.G
    B = len(seeds)
    st = [streams(s) for s in seeds]
    env = BankEnv(cfg)
    obs = env.reset_bank([x[0] for x in st])
    arng = [x[1] for x in st]
    h = agent.init_state(B)
    mod = getattr(agent, "module", None)
    hm = mod.init_state(B) if kind != "off" else None
    donor = (np.arange(B) + 1) % B                  # bank convention: next episode in the block, cyclic
    union = np.zeros((B, G, G))
    zprev = np.zeros((B, G * G))
    prev_cap = np.zeros(B, bool)
    pending = {}
    t = 0
    while not env.done.all():
        idx = np.nonzero(~env.done)[0]
        schema = None
        if kind != "off":
            z, bits, slm = _step_channel(agent, kind, env, obs, t, idx, hm, arng, donor)
            if t > 0:
                pc = prev_cap[idx]
                for k in (0, 1):
                    out["bits_sum"][k] += float(bits[pc == bool(k)].sum())
                    out["bits_n"][k] += int((pc == bool(k)).sum())
            if kind == "1c":
                zin = z[donor] * obs["action_mask"] if test_shuffle else z
                schema = np.zeros((B, G, G, 2))
                schema[..., 0] = zin.reshape(B, G, G)
            else:
                cur = np.zeros((B, G * G))
                if t > 0:
                    cur[idx] = np.exp(slm)
                cur = cur.reshape(B, G, G)
                union = np.maximum(union, cur)
                schema = np.stack([cur, union], -1)
            for b, ev in list(pending.items()):
                n = ev["n"]
                ev["F"] = window_features(z[b].reshape(G, G), zprev[b].reshape(G, G), ev["origins"], ev["w"], n)
                out["events"].append(ev)
            pending = {}
            zprev = z
        x, m = agent.features(env, obs, t, schema)
        hs, logp, _, _, _ = agent.step(x[idx], m[idx], h[idx])
        h[idx] = hs
        a = np.zeros(B, int)
        for j, b in enumerate(idx):
            u = arng[b].random(logp.shape[1])
            a[b] = int((logp[j] - np.log(-np.log(u))).argmax())
        obs, r, done, info = env.step(a)
        prev_cap = info["captured"] & info["active"]
        if kind != "off" and with_events:
            ir0, ic0 = info["intended"].T
            ar0, ac0 = info["actual"].T
            for b in np.nonzero(info["active"] & ~env.done)[0]:
                origins, pr = candidates_at(env, b, ir0[b], ic0[b])
                if len(origins) < 2:
                    continue
                w = int(env.w[b])
                act = origins.index((int(ar0[b]), int(ac0[b])))
                cap = bool(info["captured"][b])
                content = [restricted(env._fovea(b, o[0], o[1], w), cfg.n_colors) for o in origins]
                L = np.array([[np.array_equal(ci, ck) for ck in content] for ci in content], float)
                pending[b] = dict(seed=int(seeds[b]), t=t, w=w, n=int(env.n[b]), origins=origins, prior=pr,
                                  actual=act, captured=cap, L=L,
                                  h5=cap and excludes_intended_restricted(env, b, ir0[b], ic0[b]),
                                  has_other=cap and len(origins) >= 3)
        t += 1
    out["found"].append(env.found.copy())


# ------------------------------------------------------------------------------------------------ probe
def pack(events):
    """Padded arrays: F (E, Kmax, NF), mask (E, Kmax), actual (E,)."""
    E = len(events)
    K = max(len(e["origins"]) for e in events)
    F = np.zeros((E, K, NF))
    M = np.zeros((E, K))
    for i, e in enumerate(events):
        k = len(e["origins"])
        F[i, :k] = e["F"]
        M[i, :k] = 1
    return F, M, np.array([e["actual"] for e in events])


class Probe:
    """Role-blind scorer: the same MLP on every candidate's features, softmax over the event's candidates."""

    def __init__(self, nh=16, seed=0):
        rng = np.random.default_rng(seed)
        self.p = dict(W=rng.normal(0, 1 / np.sqrt(NF), (NF, nh)), b=np.zeros(nh),
                      v=rng.normal(0, 1 / np.sqrt(nh), nh))
        self.mu, self.sd = np.zeros(NF), np.ones(NF)

    def scores(self, F):
        Fs = (F - self.mu) / self.sd
        a = np.tanh(Fs @ self.p["W"] + self.p["b"])
        return a @ self.p["v"], (Fs, a)

    def probs(self, F, M):
        s, _ = self.scores(F)
        s = np.where(M > 0, s, -1e9)
        s = s - s.max(1, keepdims=True)
        e = np.exp(s) * M
        return e / e.sum(1, keepdims=True)

    def loss_grad(self, F, M, y, l2=1e-4):
        s, (Fs, a) = self.scores(F)
        s = np.where(M > 0, s, -1e9)
        s = s - s.max(1, keepdims=True)
        lse = np.log((np.exp(s) * M).sum(1))
        E = len(y)
        L = -(s[np.arange(E), y] - lse).mean() + l2 * sum((v ** 2).sum() for v in self.p.values())
        P = np.exp(s - lse[:, None]) * M
        ds = P.copy()
        ds[np.arange(E), y] -= 1
        ds /= E
        g = dict(v=np.einsum("ekh,ek->h", a, ds))
        da = ds[..., None] * self.p["v"] * (1 - a * a)
        g["W"] = np.einsum("ekf,ekh->fh", Fs, da)
        g["b"] = da.sum((0, 1))
        for k in g:
            g[k] = g[k] + 2 * l2 * self.p[k]
        return L, g

    def fit(self, events, iters=1500, lr=0.01, seed=0):
        F, M, y = pack(events)
        rows = F[M > 0]
        self.mu, self.sd = rows.mean(0), rows.std(0) + 1e-6
        m = {k: np.zeros_like(v) for k, v in self.p.items()}
        v2 = {k: np.zeros_like(v) for k, v in self.p.items()}
        for i in range(1, iters + 1):
            L, g = self.loss_grad(F, M, y)
            for k in self.p:
                m[k] = 0.9 * m[k] + 0.1 * g[k]
                v2[k] = 0.999 * v2[k] + 0.001 * g[k] ** 2
                self.p[k] -= lr * (m[k] / (1 - 0.9 ** i)) / (np.sqrt(v2[k] / (1 - 0.999 ** i)) + 1e-8)
        return float(L)


def balanced(events, seed=0):
    """Equally many training steps with and without capture (larger group subsampled at random)."""
    cap = [e for e in events if e["captured"]]
    nocap = [e for e in events if not e["captured"]]
    k = min(len(cap), len(nocap))
    rng = np.random.default_rng(seed)
    pick = lambda lst: [lst[i] for i in sorted(rng.permutation(len(lst))[:k])]
    return pick(cap) + pick(nocap)


def train_probe(agent, kind, seeds=None, seed=0):
    seeds = seeds or [PROBE_BANK + i for i in range(N_PROBE)]
    tr = collect(agent, kind, seeds)
    ev = balanced(tr["events"], seed)
    pb = Probe(seed=seed)
    L = pb.fit(ev, seed=seed)
    return pb, dict(n_train=len(ev), train_loss=L)


# ------------------------------------------------------------------------------------------- null model
def null_curves(ev, K=K_NULL):
    """Null's mean mass on (intended, actual, others) at each q in QS for one event, common random numbers."""
    rng = np.random.default_rng([ev["seed"], ev["t"], 61])
    pr = ev["prior"] / ev["prior"].sum()
    Ug = rng.random(K)
    srcdraw = np.minimum((rng.random(K)[:, None] >= np.cumsum(pr)[None, :]).sum(1), len(pr) - 1)
    L = ev["L"]
    a = ev["actual"]
    out = np.zeros((len(QS), 3))
    for i, q in enumerate(QS):
        src = np.where(Ug < q, srcdraw, a)
        post = L[src]                                  # uniform prior x likelihood
        post = post / post.sum(1, keepdims=True)
        avg = post.mean(0)
        out[i] = [avg[0], avg[a], 1.0 - avg[0] - avg[a]]
    return out


# ----------------------------------------------------------------------------------------- statistics
def _boot(episodes, stat, n_boot=N_BOOT, seed=0):
    uniq, inv = np.unique(episodes, return_inverse=True)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        cnt = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))
        vals.append(stat(cnt[inv].astype(float)))
    return np.array(vals)


def test_arrays(probe, events):
    """Probe distribution and the blind-safe quantities. P is kept in memory only."""
    F, M, y = pack(events)
    P = probe.probs(F, M)
    K = M.sum(1)
    return P, K, y


def blind_summary(probe, res, seed=0):
    """Everything that may be seen before the lock: about the actual window only."""
    ev = res["events"]
    P, K, y = test_arrays(probe, ev)
    E = len(ev)
    pA = P[np.arange(E), y]
    cap = np.array([e["captured"] for e in ev])
    h5 = np.array([e["h5"] for e in ev])
    epi = np.array([e["seed"] for e in ev])
    argA = P.argmax(1) == y
    out = dict(n_events=E, n_capture=int(cap.sum()), n_h5=int(h5.sum()),
               acc_nocapture=float(argA[~cap].mean()) if (~cap).any() else None,
               acc_capture=float(argA[cap].mean()) if cap.any() else None,
               mass_actual_capture=float(pA[cap].mean()) if cap.any() else None)
    if not h5.any():
        return out
    pAh, Kh, eh = pA[h5], K[h5], epi[h5]
    diff = pAh - 1.0 / Kh
    bd = _boot(eh, lambda w: (diff * w).sum() / w.sum(), seed=seed)
    se = float(bd.std(ddof=1))
    out["c"] = dict(mass_actual=float(pAh.mean()), uniform=float((1 / Kh).mean()), diff=float(diff.mean()),
                    se=se, z=float(diff.mean() / se) if se > 0 else float("nan"), passed=bool(diff.mean() > 2 * se),
                    n_episodes=int(len(np.unique(eh))))
    curves = np.array([null_curves(e) for e, k in zip(ev, h5) if k])          # (n_h5, nq, 3)
    nA = curves[..., 1].mean(0)
    nI = curves[..., 0].mean(0)
    q = solve_q(nA, float(pAh.mean()))
    out["feasibility"] = dict(probe_mass_actual=float(pAh.mean()), null_q0=float(nA[0]), null_q1=float(nA[-1]),
                              q=q, possible=q is not None,
                              bar_null_mass_intended=interp(nI, q) if q is not None else None)
    return out


def bits_summary(res):
    s, n = res["bits_sum"], res["bits_n"]
    return dict(bits_per_step=float(s.sum() / max(n.sum(), 1)), bits_after_nocapture=float(s[0] / max(n[0], 1)),
                bits_after_capture=float(s[1] / max(n[1], 1)), n_steps=[int(n[0]), int(n[1])])


def found_summary(res):
    f = res["found"]
    return dict(found=float(f.mean()), found_se=float(f.std() / np.sqrt(len(f))), n=int(len(f)))


# ------------------------------------------------------------------------------------- locked measures
def lock_hashes():
    files = ("probe_1c.py", "schema_1c.py", "train_1c.py", "env.py", "bank.py", "h5.py", "h5_eval.py", "agent_rl.py")
    return {f: hashlib.sha256(open(os.path.join(HERE, f), "rb").read()).hexdigest() for f in files}


def write_lock(approval, extra=None):
    assert not os.path.exists(LOCK), "already locked"
    json.dump(dict(approval=approval, sha256=lock_hashes(), **(extra or {})), open(LOCK, "w"), indent=1,
              ensure_ascii=False)


def _check_lock():
    if not os.path.exists(LOCK):
        raise RuntimeError("H5 for 1c is not locked: measure() refuses to look at where the probe's errors go")
    lk = json.load(open(LOCK))
    now = lock_hashes()
    bad = [f for f, h in lk["sha256"].items() if f in now and now[f] != h]
    if bad:
        raise RuntimeError(f"code changed after the lock: {bad}")
    return lk


def measure(probe, res, seed=0):
    """'om' (judged) and 'var' (descriptive). Only after the lock (gate 6a)."""
    _check_lock()
    ev = res["events"]
    P, K, y = test_arrays(probe, ev)
    E = len(ev)
    h5 = np.array([e["h5"] for e in ev])
    idx = np.nonzero(h5)[0]
    pA = P[idx, y[idx]]
    pI = P[idx, 0]
    pO = 1.0 - pA - pI
    other = np.array([ev[i]["has_other"] for i in idx])
    epi = np.array([ev[i]["seed"] for i in idx])
    curves = np.array([null_curves(ev[i]) for i in idx])

    def once(w):
        nI = (curves[..., 0] * w[:, None]).sum(0) / w.sum()
        nA = (curves[..., 1] * w[:, None]).sum(0) / w.sum()
        tA = (pA * w).sum() / w.sum()
        q = solve_q(nA, tA)
        clipped = q is None
        q = clip_q(nA, tA) if clipped else q
        om = (pI * w).sum() / w.sum() - interp(nI, q)
        wo = w * other
        nAo = (curves[..., 1] * wo[:, None]).sum(0)
        nOo = (curves[..., 2] * wo[:, None]).sum(0)
        var_p = (pA * wo).sum() / max(((pA + pO) * wo).sum(), 1e-12)
        var_n = interp(nAo, q) / max(interp(nAo, q) + interp(nOo, q), 1e-12)
        return om, var_p, var_n, q, clipped

    om, vp, vn, q0, cl0 = once(np.ones(len(idx)))
    uniq, inv = np.unique(epi, return_inverse=True)
    rng = np.random.default_rng(seed)
    b_om, b_var, clipped = [], [], 0
    for _ in range(N_BOOT):
        cnt = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))
        o, a, b, _, c = once(cnt[inv].astype(float))
        b_om.append(o); b_var.append(a - b); clipped += c
    se_om, se_var = float(np.std(b_om, ddof=1)), float(np.std(b_var, ddof=1))
    # 5a's rule: q must be matchable on the point estimate, else the run is not assessable (e.g. a probe that is
    # more accurate on the actual window than the unweakened observer: the null then never errs towards I)
    return dict(n_h5=int(len(idx)), n_episodes=int(len(uniq)), q=q0, point_clipped=cl0, assessable=not cl0,
                boot_clipped_frac=clipped / N_BOOT,
                om=dict(probe_mass_intended=float(pI.mean()), null_mass_intended=float(pI.mean() - om), diff=float(om), se=se_om, z=float(om / se_om) if se_om > 0 else float("nan"),
                        passed=bool((not cl0) and om > 2 * se_om)),
                var=dict(n=int(other.sum()), probe_share_actual=float(vp), null_share_actual=float(vn),
                         diff=float(vp - vn), se=se_var))
