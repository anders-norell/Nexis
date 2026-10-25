"""Sprint 6 (1c): the role-blind probe, its null model and the measures (product owner's conditions 24 Oct 23:45).

Read-out
--------
The agent is frozen. At every step t of a bank episode that is still running at t + 1, the candidate windows
for where the spotlight actually was at step t are: the intended window and the window of every salient object
eligible to capture (outside the intended window), duplicates merged. Steps with fewer than two candidates are
skipped. The channel is read at t + 1 (the module has then seen the glimpse of step t, as 1b's map was read in
5a) and at t (the step before): "t and t - 1" in the conditions.

The probe gives every candidate the same score function, computed only from the channel's values on that
candidate's cells. Primary (decision 25 Oct): mean, max, min of z at t + 1 and window mean minus grid mean at
t + 1 (4 features; z at t is what the policy steered by, so it would give the intention partly for free).
Sensitivity: the same at t + 1 and at t (8 features, standardised with the training set's mean and SD over all candidate rows). A small MLP
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
  "om" locked: mass on the intended window ("no capture") at H5 events, probe minus null at matched q;
               judged through the role-swap control (decision 25 Oct, see measure())
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
COLS_PRIMARY = [0, 1, 2, 6]   # decision 25 Oct 02:40: primary read-out only from the channel at t + 1
COLS_SENS = list(range(NF))   # sensitivity analysis: t + 1 and t
BLOCK = 500


# ---------------------------------------------------------------------------------------- channel runs
def _step_channel(agent, kind, env, obs, t, idx, hm, arng, donor, mod=None):
    """The channel z_t (B, N) at step t (0 outside idx). mod: a module other than the agent's (the untrained shadow
    module, decision 25 Oct after the pre-lock review); its noise is drawn from arng too (the shadow's own streams)."""
    G = agent.cfg.G
    B = env.B
    mod = agent.module if mod is None else mod
    eps = np.stack([arng[b].normal(0, 1, G * G) for b in idx]) if len(idx) else np.zeros((0, G * G))
    z = np.zeros((B, G * G))
    if kind == "1c":
        xm, mm = mod.features(env, obs, t)
        src = donor[idx] if mod.shuffle else idx
        hms, zz, bits, _ = mod.forward(xm[src], mm[src], hm[idx], eps, open_=t > 0, cm=mm[idx])
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


def _overlap(o, p, w):
    return abs(o[0] - p[0]) < w and abs(o[1] - p[1]) < w


def window_features(Z1, Z0, origins, w, n):
    """(K, NF) features per candidate from the channel at t + 1 (Z1) and t (Z0), both (G, G)."""
    m1, m0 = Z1[:n, :n].mean(), Z0[:n, :n].mean()
    out = np.zeros((len(origins), NF))
    for k, (r0, c0) in enumerate(origins):
        a, b = Z1[r0:r0 + w, c0:c0 + w], Z0[r0:r0 + w, c0:c0 + w]
        out[k] = [a.mean(), a.max(), a.min(), b.mean(), b.max(), b.min(), a.mean() - m1, b.mean() - m0]
    return out


def collect(agent, kind, seeds, block=BLOCK, test_shuffle=False, with_events=True, shadow=None):
    """Run the frozen agent on bank episodes. kind: '1c', '1b' or 'off'. Returns found, bits split by
    whether the previous command was captured, and the probe's events."""
    out = dict(found=[], bits_sum=np.zeros(2), bits_n=np.zeros(2), events=[])
    for s in range(0, len(seeds), block):
        _block(agent, kind, seeds[s:s + block], out, test_shuffle, with_events, shadow)
    out["found"] = np.concatenate(out["found"])
    return out


def _block(agent, kind, seeds, out, test_shuffle, with_events, shadow=None):
    """shadow: an untrained module of the same architecture (and seed) run alongside on the same episodes. It sees
    the same inputs as the agent's module, its channel is never fed to the policy, and its noise comes from separate
    streams (episode seed, 99), so the episodes are exactly those of the agent alone. Events then also carry the
    shadow channel's features (F_sh): a paired reference for what an untrained channel passes on."""
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
    if shadow is not None:
        srng = [np.random.default_rng([int(s_), 99]) for s_ in seeds]
        hsh = shadow.init_state(B)
        zsh_prev = np.zeros((B, G * G))
    donor = (np.arange(B) + 1) % B                  # bank convention: next episode in the block, cyclic
    union = np.zeros((B, G, G))
    zprev = np.zeros((B, G * G))
    prev_cap = np.zeros(B, bool)
    pending = {}
    recent = {}                                     # b -> event read out at this step, waiting for command t + 1
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
            if shadow is not None:
                zsh, _, _ = _step_channel(agent, kind, env, obs, t, idx, hsh, srng, donor, mod=shadow)
            for b, ev in list(pending.items()):
                n = ev["n"]
                ev["F"] = window_features(z[b].reshape(G, G), zprev[b].reshape(G, G), ev["origins"], ev["w"], n)
                if shadow is not None:
                    ev["F_sh"] = window_features(zsh[b].reshape(G, G), zsh_prev[b].reshape(G, G), ev["origins"],
                                                 ev["w"], n)
                out["events"].append(ev)
                recent[b] = ev
            pending = {}
            zprev = z
            if shadow is not None:
                zsh_prev = zsh
        x, m = agent.features(env, obs, t, schema)
        hs, logp, _, _, _ = agent.step(x[idx], m[idx], h[idx])
        h[idx] = hs
        a = np.zeros(B, int)
        for j, b in enumerate(idx):
            u = arng[b].random(logp.shape[1])
            a[b] = int((logp[j] - np.log(-np.log(u))).argmax())
        obs, r, done, info = env.step(a)
        prev_cap = info["captured"] & info["active"]
        for b, ev in recent.items():                # does command t + 1 aim at the intended window of t again?
            r0, c0 = ev["origins"][0]
            nr, nc = info["intended"][b]
            ev["next_overlap"] = bool(abs(int(nr) - r0) < ev["w"] and abs(int(nc) - c0) < ev["w"])
        recent = {}
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
                # decoy for the role-swap control: another wrong candidate (a salient object's window, not the actual
                # one) whose content differs from the glimpse, as the intended window's does at H5 events. With the
                # probe trained on capture steps only this is calibrated (test_probe_1c.py)
                dec = [j for j in range(1, len(origins)) if j != act and L[act, j] == 0] if cap else []
                drng = np.random.default_rng([int(seeds[b]), t, 71])
                pending[b] = dict(seed=int(seeds[b]), t=t, w=w, n=int(env.n[b]), origins=origins, prior=pr,
                                  actual=act, captured=cap, L=L,
                                  h5=cap and excludes_intended_restricted(env, b, ir0[b], ic0[b]),
                                  has_other=cap and len(origins) >= 3,
                                  decoy=int(dec[drng.integers(len(dec))]) if dec else None)
        t += 1
    out["found"].append(env.found.copy())


def _window_cover(G, r0, c0, w):
    m = np.zeros((G, G), int)
    m[r0:r0 + w, c0:c0 + w] = 1
    return m


# ------------------------------------------------------------------------------------------------ probe
def pack(events, fk="F"):
    """Padded arrays: F (E, Kmax, NF), mask (E, Kmax), actual (E,). fk: 'F' (the agent's channel) or 'F_sh'."""
    E = len(events)
    K = max(len(e["origins"]) for e in events)
    F = np.zeros((E, K, NF))
    M = np.zeros((E, K))
    for i, e in enumerate(events):
        k = len(e["origins"])
        F[i, :k] = e[fk]
        M[i, :k] = 1
    return F, M, np.array([e["actual"] for e in events])


class Probe:
    """Role-blind scorer: the same MLP on every candidate's features, softmax over the event's candidates."""

    def __init__(self, nh=16, seed=0, cols=None):
        rng = np.random.default_rng(seed)
        self.cols = list(COLS_PRIMARY if cols is None else cols)
        nf = len(self.cols)
        self.p = dict(W=rng.normal(0, 1 / np.sqrt(nf), (nf, nh)), b=np.zeros(nh),
                      v=rng.normal(0, 1 / np.sqrt(nh), nh))
        self.mu, self.sd = np.zeros(nf), np.ones(nf)

    def scores(self, F):
        Fs = (F[..., self.cols] - self.mu) / self.sd
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

    def fit(self, events, iters=1500, lr=0.01, seed=0, fk="F"):
        F, M, y = pack(events, fk)
        rows = F[M > 0][:, self.cols]
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


def training_set(events):
    """Probe training set (pre-lock review 25 Oct, calibration): the capture steps only. With equally many steps
    without capture (the first design), the probe learns 'the actual window is the non-salient one' from the steps
    without capture, where the actual window is the intended one; any channel that tells salient from non-salient
    windows then passes the role-swap control (+54 SE in the calibration). Trained on capture steps, where the
    actual window is always a salient object's, no channel without role information passes."""
    return [e for e in events if e["captured"]]


def balanced(events, seed=0):
    """Equally many training steps with and without capture (larger group subsampled at random)."""
    cap = [e for e in events if e["captured"]]
    nocap = [e for e in events if not e["captured"]]
    k = min(len(cap), len(nocap))
    rng = np.random.default_rng(seed)
    pick = lambda lst: [lst[i] for i in sorted(rng.permutation(len(lst))[:k])]
    return pick(cap) + pick(nocap)


def train_probe(agent, kind, seeds=None, seed=0, cols=None):
    """The primary probe (channel at t + 1 only) unless cols says otherwise."""
    pbs, info = train_probes(agent, kind, seeds, seed, {"p": cols})
    return pbs["p"], info["p"]


def train_probes(agent, kind, seeds=None, seed=0, which=None, shadow=None):
    """Several probes on one probe-bank run. which: name -> cols (default primary and sensitivity). With a shadow
    module, also 'shadow_primary': the primary probe trained the same way on the shadow channel (F_sh)."""
    which = which or {"primary": COLS_PRIMARY, "sens": COLS_SENS}
    seeds = seeds or [PROBE_BANK + i for i in range(N_PROBE)]
    tr = collect(agent, kind, seeds, shadow=shadow)
    ev = training_set(tr["events"])
    jobs = [(name, cols, "F") for name, cols in which.items()]
    if shadow is not None:
        jobs.append(("shadow_primary", COLS_PRIMARY, "F_sh"))
    pbs, info = {}, {}
    for name, cols, fk in jobs:
        pb = Probe(seed=seed, cols=cols)
        info[name] = dict(n_train=len(ev), train_loss=pb.fit(ev, seed=seed, fk=fk), cols=pb.cols, features=fk)
        pbs[name] = pb
    return pbs, info


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
    """Everything that may be seen before the lock: about the actual window only. Review point 4: with two
    candidates the mass on the intended window is 1 - the mass on the actual window, so before the lock every
    capture-step quantity (accuracy, mass on actual, (c), feasibility of q) is computed only over events with
    at least three candidates. The full (c) over all H5 events is computed by measure() after the lock."""
    ev = res["events"]
    P, K, y = test_arrays(probe, ev)
    E = len(ev)
    pA = P[np.arange(E), y]
    cap = np.array([e["captured"] for e in ev])
    h5 = np.array([e["h5"] for e in ev])
    k3 = K >= 3
    epi = np.array([e["seed"] for e in ev])
    argA = P.argmax(1) == y
    cap3, h53 = cap & k3, h5 & k3
    out = dict(n_events=E, n_capture=int(cap.sum()), n_h5=int(h5.sum()), n_h5_k2=int((h5 & ~k3).sum()),
               n_h5_k3=int(h53.sum()), blind_subset="capture steps: K >= 3 only",
               acc_nocapture=float(argA[~cap].mean()) if (~cap).any() else None,
               acc_capture_k3=float(argA[cap3].mean()) if cap3.any() else None,
               mass_actual_capture_k3=float(pA[cap3].mean()) if cap3.any() else None)
    if not h53.any():
        return out
    pAh, Kh, eh = pA[h53], K[h53], epi[h53]
    diff = pAh - 1.0 / Kh
    bd = _boot(eh, lambda w: (diff * w).sum() / w.sum(), seed=seed)
    se = float(bd.std(ddof=1))
    out["c_k3"] = dict(mass_actual=float(pAh.mean()), uniform=float((1 / Kh).mean()), diff=float(diff.mean()),
                       se=se, z=float(diff.mean() / se) if se > 0 else float("nan"),
                       passed=bool(diff.mean() > 2 * se), n_episodes=int(len(np.unique(eh))))
    curves = np.array([null_curves(e) for e, k in zip(ev, h53) if k])
    nA = curves[..., 1].mean(0)
    nI = curves[..., 0].mean(0)
    q = solve_q(nA, float(pAh.mean()))
    out["feasibility_k3"] = dict(probe_mass_actual=float(pAh.mean()), null_q0=float(nA[0]), null_q1=float(nA[-1]),
                                 q=q, possible=q is not None,
                                 bar_null_mass_intended=interp(nI, q) if q is not None else None)
    return out


def c_all(probe, res, seed=0):
    """(c) over all H5 events (K = 2 included). Only after the lock (review point 4)."""
    _check_lock()
    ev = res["events"]
    P, K, y = test_arrays(probe, ev)
    h5 = np.array([e["h5"] for e in ev])
    pA = P[np.arange(len(ev)), y][h5]
    diff = pA - 1.0 / K[h5]
    epi = np.array([e["seed"] for e in ev])[h5]
    se = float(_boot(epi, lambda w: (diff * w).sum() / w.sum(), seed=seed).std(ddof=1))
    return dict(mass_actual=float(pA.mean()), uniform=float((1 / K[h5]).mean()), diff=float(diff.mean()), se=se,
                z=float(diff.mean() / se) if se > 0 else float("nan"), passed=bool(diff.mean() > 2 * se))


def bits_summary(res):
    s, n = res["bits_sum"], res["bits_n"]
    return dict(bits_per_step=float(s.sum() / max(n.sum(), 1)), bits_after_nocapture=float(s[0] / max(n[0], 1)),
                bits_after_capture=float(s[1] / max(n[1], 1)), n_steps=[int(n[0]), int(n[1])])


def found_summary(res):
    f = res["found"]
    return dict(found=float(f.mean()), found_se=float(f.std() / np.sqrt(len(f))), n=int(len(f)))


# ------------------------------------------------------------------------------------- locked measures
def lock_hashes():
    files = ("probe_1c.py", "schema_1c.py", "train_1c.py", "env.py", "bank.py", "h5.py", "h5_eval.py", "agent_rl.py",
             "train_rl.py", "schema_channel.py", "eval_main_1c.py", "pilot_1c.py", "run_sprint6.sh")
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


def measure(probe, res, seed=0, fk="F"):
    """'om' (judged through the role-swap control) and 'var' (descriptive). Only after the lock (gate 6a).

    Decisions 25 Oct (product owner, on Claude's recommendations; pre-lock review 25 Oct):
      om          probe's mass on the intended window minus the null's, at H5 events, q matched on mass on the
                  actual window over all H5 events. Reported, not the criterion.
      control     at H5 events with a decoy: the probe's mass on the intended window minus on the decoy, another
                  wrong candidate (a salient object's window) whose content differs from the glimpse; probe trained
                  on capture steps only (training_set). Calibration (test_probe_1c.py, 25 Oct): no channel without
                  role information passes (noise, marker, wrong candidates, salience, steering signal at t, passive
                  intended-window input: z between -57 and +0.7); AST-like channels +18 to +20 SE. A channel that
                  codes salience or passes the intended window's input on comes out negative (conservative).
                  measure_pair gives the trained minus the untrained shadow channel (reported).
      split       the control separately for events where command t + 1 aims at the intended window again and where
                  it does not (descriptive; a yes is read as AST only if the no-repeat group points the same way).
    fk: 'F' the agent's channel, 'F_sh' the shadow's (with a probe trained on F_sh)."""
    _check_lock()
    ev = [e for e in res["events"] if e["h5"]]
    if fk == "F_sh":
        assert all("F_sh" in e for e in ev), "no shadow features: collect with shadow="
    F, M, y = pack(ev, fk)
    P = probe.probs(F, M)
    hasd = np.array([e["decoy"] is not None for e in ev])
    pD = np.array([P[i, e["decoy"]] if e["decoy"] is not None else 0.0 for i, e in enumerate(ev)])
    n = len(ev)
    pA = P[np.arange(n), y]
    pI = P[:, 0]
    pO = 1.0 - pA - pI
    ctl = P[:, 0] - pD
    other = np.array([e["has_other"] for e in ev])
    rep_ = np.array([e.get("next_overlap", False) for e in ev])
    epi = np.array([e["seed"] for e in ev])
    curves = np.array([null_curves(e) for e in ev])

    def wavg(arr, w):
        return (arr * w).sum() / max(w.sum(), 1e-12)

    def once(w):
        nA = (curves[..., 1] * w[:, None]).sum(0) / w.sum()
        tA = wavg(pA, w)
        q = solve_q(nA, tA)
        clipped = q is None
        q = clip_q(nA, tA) if clipped else q
        nIq = np.array([interp(c, q) for c in curves[..., 0]])
        res_ = dict(q=q, clipped=clipped, om=wavg(pI - nIq, w))
        for name, g in (("all", np.ones(n, bool)), ("repeat", rep_), ("norepeat", ~rep_)):
            res_[f"ctrl_{name}"] = wavg(ctl, w * hasd * g)
            res_[f"om_{name}"] = wavg(pI - nIq, w * g)
        wo = w * other
        nAo = (curves[..., 1] * wo[:, None]).sum(0)
        nOo = (curves[..., 2] * wo[:, None]).sum(0)
        res_["var_p"] = (pA * wo).sum() / max(((pA + pO) * wo).sum(), 1e-12)
        res_["var_n"] = interp(nAo, q) / max(interp(nAo, q) + interp(nOo, q), 1e-12)
        return res_

    pt = once(np.ones(n))
    uniq, inv = np.unique(epi, return_inverse=True)
    rng = np.random.default_rng(seed)
    boots, clipped = [], 0
    for _ in range(N_BOOT):
        cnt = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))
        b = once(cnt[inv].astype(float))
        boots.append(b); clipped += b["clipped"]
    se = lambda k: float(np.std([b[k] for b in boots], ddof=1))
    stat = lambda k: dict(value=float(pt[k]), se=se(k), z=float(pt[k] / se(k)) if se(k) > 0 else float("nan"))
    ctrl = stat("ctrl_all")
    return dict(probe_cols=probe.cols, features=fk, n_h5=int(n), n_with_decoy=int(hasd.sum()),
                n_repeat=int((hasd & rep_).sum()), n_norepeat=int((hasd & ~rep_).sum()),
                n_episodes=int(len(uniq)), q=pt["q"], assessable=not pt["clipped"], boot_clipped_frac=clipped / N_BOOT,
                om=dict(probe_mass_intended=float(pI.mean()), **stat("om")),
                control=dict(**ctrl, passed=bool(ctrl["value"] > 2 * ctrl["se"])),
                split=dict(repeat=dict(om=stat("om_repeat"), control=stat("ctrl_repeat")),
                           norepeat=dict(om=stat("om_norepeat"), control=stat("ctrl_norepeat"))),
                var=dict(n=int(other.sum()), probe_share_actual=float(pt["var_p"]),
                         null_share_actual=float(pt["var_n"]), diff=float(pt["var_p"] - pt["var_n"]),
                         se=float(np.std([b["var_p"] - b["var_n"] for b in boots], ddof=1))))


def measure_pair(probe, probe_sh, res, seed=0):
    """The judged 'om' (pre-lock review, finding 2): the trained channel's control minus the untrained shadow
    channel's, on the same H5 events and decoys, paired per event, bootstrap over episodes; also by next-command
    group. Only after the lock."""
    _check_lock()
    ev = [e for e in res["events"] if e["h5"] and e["decoy"] is not None]
    dj = np.array([e["decoy"] for e in ev])
    Pt = probe.probs(*pack(ev, "F")[:2])
    Ps = probe_sh.probs(*pack(ev, "F_sh")[:2])
    pDt, pDs = Pt[np.arange(len(ev)), dj], Ps[np.arange(len(ev)), dj]
    d = (Pt[:, 0] - pDt) - (Ps[:, 0] - pDs)
    rep_ = np.array([e.get("next_overlap", False) for e in ev])
    epi = np.array([e["seed"] for e in ev])
    out = dict(n=int(len(ev)), n_episodes=int(len(np.unique(epi))))
    for name, g in (("all", np.ones(len(ev), bool)), ("repeat", rep_), ("norepeat", ~rep_)):
        if not g.any():
            out[name] = None
            continue
        val = float(d[g].mean())
        bs = _boot(epi[g], lambda w: (d[g] * w).sum() / w.sum(), seed=seed)
        se = float(bs.std(ddof=1))
        out[name] = dict(value=val, se=se, z=val / se if se > 0 else float("nan"), n=int(g.sum()),
                         trained=float((Pt[g, 0] - pDt[g]).mean()), untrained=float((Ps[g, 0] - pDs[g]).mean()))
    out["passed"] = bool(out["all"] is not None and out["all"]["value"] > 2 * out["all"]["se"])
    return out
