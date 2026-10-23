"""H5 measurement and its null model, the ideal observer (sprint 3).

H5: when the spotlight is captured, does the learned schema (1b) report the *intended* place
more often than other wrong places, even when the information says otherwise?

Measurement rules (spec, locked in sprint 3):
- only captures where the fovea excludes the intended place: the glimpse was shown (not masked)
  and differs from what the intended window would have shown;
- the schema is read the step after the capture (the 1b head at step t+1 predicts where the
  spotlight actually was at step t, after the agent has seen the captured glimpse);
- captures that find the target are excluded (the episode ends);
- the comparison is with an ideal observer that has the same inputs, and with the other
  salient places, not with chance.

Ideal observer. Candidates for where the spotlight landed at step t: the intended window, or
the window of each salient object that was eligible to capture (outside the intended window).
Prior: the environment's own capture rule, p_j = p_capture * s_j (scaled to capture_cap), no
capture with 1 - sum p_j. The observer is given the true salience values, which is more than
the agent has (it sees them with noise), so it is if anything a stronger observer. It knows
that the agent never aims at a salient object only through the intended window, which it gets
from the efference copy like the agent. Likelihood: 1 if the candidate window would show
exactly the glimpse seen, else 0; a masked glimpse fits every candidate.
"""
import numpy as np
from env import window_origin, in_window


def candidates(env, b, ir0, ic0):
    """Candidate window origins for episode b at the step just taken, with prior weights.
    Returns list of (origin, prior, kind) with kind 'intended' or salient index j."""
    cfg, w = env.cfg, env.w[b]
    out = []
    ps = []
    for j in range(env.sal_pos.shape[1]):
        r, c = env.sal_pos[b, j]
        if r < 0 or in_window(r, c, ir0, ic0, w):
            continue
        ps.append((j, cfg.p_capture * env.sal_val[b, j]))
    tot = sum(p for _, p in ps)
    scale = cfg.capture_cap / tot if tot > cfg.capture_cap else 1.0
    out.append(((ir0, ic0), 1.0 - tot * scale, "intended"))
    for j, p in ps:
        r, c = env.sal_pos[b, j]
        sr0, sc0 = window_origin(np.array([r]), np.array([c]), w, env.n[b])
        out.append(((int(sr0[0]), int(sc0[0])), p * scale, j))
    return out


def observer_posterior(env, b, ir0, ic0, glimpse, revealed):
    """Posterior over candidate windows given the glimpse. Returns list of (origin, post, kind)."""
    w = env.w[b]
    cands = candidates(env, b, ir0, ic0)
    post = []
    for (o, pr, kind) in cands:
        lik = 1.0 if (not revealed or np.array_equal(env._fovea(b, o[0], o[1], w), glimpse)) else 0.0
        post.append(pr * lik)
    z = sum(post)
    return [(o, p / z if z > 0 else 0.0, kind) for (o, _, kind), p in zip(cands, post)]


def excludes_intended(env, b, ir0, ic0):
    """True if the glimpse just shown rules out the intended window."""
    return bool(env.last_revealed[b]) and not np.array_equal(
        env._fovea(b, ir0, ic0, env.w[b]), env.last_window[b])


def window_mask(G, r0, c0, w):
    m = np.zeros((G, G))
    m[r0:r0 + w, c0:c0 + w] = 1.0
    return m


def run_h5(agent, env, n_ep, rng, B=500):
    """Run the agent with its own schema mode and collect H5 events.

    Each event: an eligible capture at step t; the 1b head's prediction at step t+1 is split
    into mass on the intended window, the actual window, the other candidate windows and the
    rest; the observer's posterior mass on the same regions; and which region the head's
    most probable cell falls in.
    """
    from agent_rl import schema_input
    G = env.cfg.G
    events = []
    for _ in range(n_ep // B):
        obs = env.reset(B)
        h = agent.init_state(B)
        mod = getattr(agent, "module", None)
        hm = mod.init_state(B) if mod is not None else None
        union = np.zeros((B, G, G))
        visited = np.zeros((B, G, G))
        schema = None
        donor = rng.permutation(B)
        pending = {}                                    # b -> event waiting for step t+1 read-out
        t = 0
        while not env.done.all():
            idx = np.nonzero(~env.done)[0]
            if mod is not None:                          # separate schema module: its 1b head is the schema
                xm, mm = mod.features(env, obs, t, None)
                hms, _, _, slm, _ = mod.step(xm[idx], mm[idx], hm[idx])
                hm[idx] = hms
                cur = np.zeros((B, G * G))
                if t > 0:
                    cur[idx] = np.exp(slm)
                union = np.maximum(union, cur.reshape(B, G, G))
                if agent.schema_mode == "module":
                    schema = np.stack([cur.reshape(B, G, G), union], -1)
            x, m = agent.features(env, obs, t, schema)
            hs, logp, Vs, slogp, _ = agent.step(x[idx], m[idx], h[idx])
            h[idx] = hs
            pred = np.zeros((B, G * G)); pred[idx] = np.exp(slogp)
            if mod is not None:
                pred = cur
            for b, ev in list(pending.items()):          # read the head the step after the capture
                p = pred[b].reshape(G, G)
                regions = ev.pop("_masks")
                ev["head"] = {k: float((p * mk).sum()) for k, mk in regions.items()}
                cell = np.unravel_index(p.argmax(), p.shape)
                ev["head_argmax"] = next((k for k in ("intended", "actual", "other")
                                          if regions[k][cell] > 0), "rest")
                events.append(ev)
            pending = {}
            a_s = (logp - np.log(-np.log(rng.random(logp.shape)))).argmax(1)
            a = np.zeros(B, int); a[idx] = a_s
            obs, r, done, info = env.step(a)
            for b in np.nonzero(info["captured"] & info["active"] & ~info["hit"] & ~env.done)[0]:
                ir0, ic0 = info["intended"][b]
                ar0, ac0 = info["actual"][b]
                if not excludes_intended(env, b, ir0, ic0):
                    continue
                w = env.w[b]
                post = observer_posterior(env, b, ir0, ic0, env.last_window[b], True)
                mi = window_mask(G, ir0, ic0, w)
                ma = window_mask(G, ar0, ac0, w)
                mo = np.zeros((G, G))
                for (o, _, kind) in post:
                    if kind != "intended" and tuple(o) != (ar0, ac0):
                        mo = np.maximum(mo, window_mask(G, o[0], o[1], w))
                ma_only = ma * (1 - mi)
                mo_only = mo * (1 - mi) * (1 - ma)
                rest = np.ones((G, G)) - np.maximum(np.maximum(mi, ma), mo)
                obs_mass = dict(intended=sum(p for o, p, k in post if k == "intended"),
                                actual=sum(p for o, p, k in post if tuple(o) == (ar0, ac0) and k != "intended"))
                obs_mass["other"] = 1.0 - obs_mass["intended"] - obs_mass["actual"]
                pending[b] = dict(t=t, w=int(w), n_cand=len(post), observer=obs_mass,
                                  _masks=dict(intended=mi, actual=ma_only, other=mo_only, rest=rest))
            if agent.schema_mode != "module":
                schema, visited = schema_input(agent.schema_mode, env, info, visited, pred, donor)
            t += 1
    return events


def summarise_h5(events):
    if not events:
        return dict(n=0)
    keys = ("intended", "actual", "other", "rest")
    head = {k: float(np.mean([e["head"][k] for e in events])) for k in keys}
    argmax = {k: float(np.mean([e["head_argmax"] == k for e in events])) for k in keys}
    obs = {k: float(np.mean([e["observer"][k] for e in events])) for k in ("intended", "actual", "other")}
    with_other = [e for e in events if e["n_cand"] > 2]
    return dict(n=len(events), head_mass=head, head_argmax=argmax, observer_mass=obs,
                n_with_other_salient=len(with_other))


# ---------------------------------------------------------------------------------------------------
# Sprint 5a (approved 23 Oct): restricted observer and the mixing null model (condition a).
# "The fovea excludes the intended place" is decided with what the agent can compare against the
# periphery: occupied and colour, not shape (the periphery never shows shape). The observer gets the
# same restriction.

def restricted(glimpse, C):
    """Keep in-window flag, occupied and colour channels of a fovea patch."""
    return glimpse[..., :2 + C]


def excludes_intended_restricted(env, b, ir0, ic0):
    C = env.cfg.n_colors
    return bool(env.last_revealed[b]) and not np.array_equal(
        restricted(env._fovea(b, ir0, ic0, env.w[b]), C), restricted(env.last_window[b], C))


def observer_restricted(env, b, ir0, ic0, glimpse, sal=None):
    """Posterior over candidate windows with the restricted likelihood. sal: optional replacement
    salience values (M,) for the prior. Returns list of (origin, post, kind)."""
    cfg, w, C = env.cfg, env.w[b], env.cfg.n_colors
    sv = env.sal_val[b] if sal is None else sal
    cands, ps = [], []
    for j in range(env.sal_pos.shape[1]):
        r, c = env.sal_pos[b, j]
        if r < 0 or in_window(r, c, ir0, ic0, w):
            continue
        ps.append((j, cfg.p_capture * sv[j]))
    tot = sum(p for _, p in ps)
    scale = cfg.capture_cap / tot if tot > cfg.capture_cap else 1.0
    cands.append(((ir0, ic0), 1.0 - tot * scale, "intended"))
    for j, p in ps:
        r, c = env.sal_pos[b, j]
        sr0, sc0 = window_origin(np.array([r]), np.array([c]), w, env.n[b])
        cands.append(((int(sr0[0]), int(sc0[0])), p * scale, j))
    g = restricted(glimpse, C)
    post = [pr * (1.0 if np.array_equal(restricted(env._fovea(b, o[0], o[1], w), C), g) else 0.0)
            for (o, pr, _) in cands]
    z = sum(post)
    return [(o, (p / z if z > 0 else 0.0), kind) for (o, _, kind), p in zip(cands, post)]


def observer_mixed(env, b, ir0, ic0, glimpse, q, rng, K=200):
    """Null model (condition a): the restricted ideal observer weakened by mixing. In each of K draws,
    with probability q the glimpse is replaced by the (restricted) content of a candidate window drawn
    at random, and each salience value independently with probability q by a fresh draw from the
    environment's own distribution U(s_min, 1). The answer is the posterior averaged over the draws.
    Unlike damping (a power of the likelihood), an excluded place can get mass when q > 0."""
    cfg = env.cfg
    base = observer_restricted(env, b, ir0, ic0, glimpse)
    acc = np.zeros(len(base))
    if q <= 0:
        return base
    for _ in range(K):
        sal = env.sal_val[b].copy()
        swap = rng.random(sal.shape) < q
        sal[swap] = rng.uniform(cfg.s_min, 1.0, swap.sum())
        g = glimpse
        if rng.random() < q:
            o = base[int(rng.integers(len(base)))][0]
            g = env._fovea(b, o[0], o[1], env.w[b])
        post = observer_restricted(env, b, ir0, ic0, g, sal)
        acc += np.array([p for _, p, _ in post])
    acc /= K
    return [(o, float(p), kind) for (o, _, kind), p in zip(base, acc)]
