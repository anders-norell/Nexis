"""H5 evaluation for 1b (gate 5a, product owner's decisions 24 Oct 2026).

What is computed, per run (one seed, one level):

Events. H5 events as in h5_bank (capture at step t that does not find the target, episode still running,
the fovea excludes the intended window by occupied and colour; the map is read at step t + 1), restricted
to events with at least one OTHER salient candidate: an eligible salient object whose window is not the
actual window and has cells outside the intended and actual windows. Otherwise there is only one wrong
place and the comparison is empty (review point 2).

Regions (cells, counted the same way for schema and null, review point 2):
  I = intended window;  A = actual window minus I;  O = other candidate windows minus I and A.
The schema's map is renormalised over I + A + O; its mass outside the candidates ("rest") is reported
descriptively.

Measure. Share of error mass on the intended window: sum over events of I' divided by the sum of
(I' + O'), where ' means renormalised over the candidates (ratio of sums within the seed). The same for
the null model.

Null model (condition a, decisions 24 Oct). The restricted ideal observer weakened by mixing: in each of
K draws, with probability q the glimpse is replaced by the restricted content of a candidate window drawn
from the capture prior (the environment's own capture probabilities with the true salience: "after the
capture's prior probabilities"), and each salience value independently with probability q by a fresh draw
from U(s_min, 1). The null's answer is the posterior averaged over the draws. Its window posterior is
turned into a cell map by spreading each candidate's mass evenly over the window's cells, so it lives on
the same cells as the schema. Common random numbers per event (seeded by episode seed and step) make the
null's curves smooth in q.

Matching (decisions 24 Oct, reserve order fixed in advance):
  1. "mass": q such that the null's mean renormalised mass on A equals the schema's.
  2. "argmax" if 1 is impossible: q such that the null's rate of most probable cell in A equals the
     schema's (the null's ties are split evenly over the tied cells).
  3. otherwise the level is not assessable.
The same method is used for both seeds of a level: mass if mass is possible in every seed, else argmax if
argmax is possible in every seed, else not assessable. "Possible" = the schema's value lies within the
range the null's curve covers on q in [0, 1] (grid of 41 points, linear interpolation, first crossing).

Statistics. Bootstrap over episodes (events cluster within episodes), 1 000 resamples, q matched again in
each resample (clipped to the end of the grid when a resample falls outside the curve; the clipped fraction
is reported). SE = SD of the bootstrap differences. Judged per seed: schema share > null share by more than
2 SE.

Blindness. feasibility() uses only quantities about the actual window (renormalised mass on A, most
probable cell in A), never I or O of the schema. Schema I and O are stored only when the bank is run with
blind=False, which h5_5a.py allows only after the lock file exists.
"""
import numpy as np
from env import window_origin, in_window
from h5 import restricted

QS = np.linspace(0.0, 1.0, 41)
# Proposed after the code review (24 Oct, not yet approved by the product owner):
DROP_SHARED = True        # cells shared by the intended window and another candidate window belong to no region
EQUALIZE_DIFFUSE = True   # the null gets the schema's diffuse (uniform) share, per event, before renormalising
PLATEAU = 0.10            # argmax reserve: if the curve is within PLATEAU_TOL of the target over a q-span wider
PLATEAU_TOL = 0.01        # than this, q is not identified and the reserve counts as impossible
K_NULL = 400
N_BOOT = 1000


# ------------------------------------------------------------------------------------------- events
def event_detail(env, b, ir0, ic0, ar0, ac0, who):
    """Everything the null model and the region split need for one capture event."""
    cfg, w, C, G = env.cfg, int(env.w[b]), env.cfg.n_colors, env.cfg.G
    origins, sal, objs = [(int(ir0), int(ic0))], [], [None]
    src = None
    for j in range(env.sal_pos.shape[1]):
        r, c = env.sal_pos[b, j]
        if r < 0 or in_window(r, c, ir0, ic0, w):
            continue
        sr0, sc0 = window_origin(np.array([r]), np.array([c]), w, env.n[b])
        if j == who:
            src = len(origins)
        origins.append((int(sr0[0]), int(sc0[0])))
        sal.append(float(env.sal_val[b, j]))
        objs.append((int(r), int(c)))
    assert src is not None, "capturer must be an eligible candidate"
    nc = len(origins)
    content = [restricted(env._fovea(b, o[0], o[1], w), C) for o in origins]
    L = np.array([[np.array_equal(content[i], content[k]) for k in range(nc)] for i in range(nc)], float)
    assert np.array_equal(restricted(env.last_window[b], C), content[src]), "glimpse = capturer's window"
    masks = np.zeros((nc, G, G), bool)
    for k, (r0, c0) in enumerate(origins):
        masks[k, r0:r0 + w, c0:c0 + w] = True
    Iw = masks[0]
    Aw = np.zeros((G, G), bool)
    Aw[ar0:ar0 + w, ac0:ac0 + w] = True
    Ow = np.zeros((G, G), bool)
    for k in range(1, nc):
        if origins[k] != (int(ar0), int(ac0)):
            Ow |= masks[k]
    shared = Iw & (Aw | Ow) if DROP_SHARED else np.zeros((G, G), bool)
    I = Iw & ~shared
    A = Aw & ~Iw
    O = Ow & ~Iw & ~A
    frac = np.stack([(masks & R).sum((1, 2)) for R in (I, A, O)], 1) / float(w * w)
    n = int(env.n[b])
    return dict(origins=origins, objs=objs, sal=np.array(sal), src=src, L=L, masks=masks, I=I, A=A, O=O, frac=frac,
                w=w, n=n, n_valid=n * n, counts=np.array([I.sum(), A.sum(), O.sum()], float),
                p_capture=cfg.p_capture, cap=cfg.capture_cap, s_min=cfg.s_min, has_other=bool(O.any()))


def prior(sal, p_capture, cap):
    """Capture prior over candidates (intended first), for sal of shape (..., ns)."""
    p = p_capture * sal
    tot = p.sum(-1, keepdims=True)
    scale = np.where(tot > cap, cap / np.maximum(tot, 1e-12), 1.0)
    p = p * scale
    return np.concatenate([1.0 - p.sum(-1, keepdims=True), p], -1)


def schema_regions(p, det):
    """Schema's map p (G, G) on the event's regions. Returns blind-safe and full fields."""
    I, A, O = (float(p[det[k]].sum()) for k in ("I", "A", "O"))
    tot = I + A + O
    cand = det["I"] | det["A"] | det["O"]
    cell = np.unravel_index(int(np.argmax(np.where(cand, p, -1.0))), p.shape)   # over candidate cells only
    region = "A" if det["A"][cell] else "I" if det["I"][cell] else "O"
    blind = dict(sA=A / tot if tot > 0 else 0.0, s_cand=tot, s_rest=1.0 - tot, s_argA=float(region == "A"))
    full = dict(sI=I / tot if tot > 0 else 0.0, sO=O / tot if tot > 0 else 0.0, s_argregion=region)
    return blind, full


def null_cellmap(det, seed, q, K=K_NULL, return_post=False):
    """The null's cell map at one q (for synthetic tests): window posterior spread evenly over window cells."""
    rng = np.random.default_rng(seed)
    ns = len(det["sal"])
    s0 = det["sal"]
    p0 = prior(s0, det["p_capture"], det["cap"])
    Ug = rng.random(K)
    srcdraw = np.minimum((rng.random(K)[:, None] >= np.cumsum(p0)[None, :]).sum(1), len(p0) - 1)
    Us = rng.random((K, ns))
    snew = rng.uniform(det["s_min"], 1.0, (K, ns))
    src = np.where(Ug < q, srcdraw, det["src"])
    sal = np.where(Us < q, snew, s0[None, :])
    post = prior(sal, det["p_capture"], det["cap"]) * det["L"][src]
    post /= post.sum(1, keepdims=True)
    cm = np.tensordot(post.mean(0), det["masks"].astype(float), 1) / det["w"] ** 2
    return (cm, post.mean(0)) if return_post else cm


def null_curve(det, seed, qs=QS, K=K_NULL):
    """Null model at each q: its cell-map mass on the regions (nq, 3) as (I, A, O), not renormalised (shared
    cells may hold some), and the rate of most probable candidate cell in A (nq,), ties split evenly. Common
    random numbers across q."""
    rng = np.random.default_rng(seed)
    ns = len(det["sal"])
    s0 = det["sal"]
    p0 = prior(s0, det["p_capture"], det["cap"])
    Ug = rng.random(K)
    srcdraw = np.minimum((rng.random(K)[:, None] >= np.cumsum(p0)[None, :]).sum(1), len(p0) - 1)
    Us = rng.random((K, ns))
    snew = rng.uniform(det["s_min"], 1.0, (K, ns))
    L, frac, masks, w2 = det["L"], det["frac"], det["masks"], det["w"] ** 2
    regs = np.zeros((len(qs), 3))
    argA = np.zeros(len(qs))
    A, I, O = det["A"], det["I"], det["O"]
    for i, q in enumerate(qs):
        src = np.where(Ug < q, srcdraw, det["src"])
        sal = np.where(Us < q, snew, s0[None, :])
        post = prior(sal, det["p_capture"], det["cap"]) * L[src]
        post /= post.sum(1, keepdims=True)
        avg = post.mean(0)
        regs[i] = avg @ frac
        cm = np.tensordot(avg, masks.astype(float), 1) / w2
        cand = I | A | O
        cm = np.where(cand, cm, -1.0)
        top = cm >= cm.max() - 1e-12
        argA[i] = (top & A).sum() / top.sum()
    return regs, argA


def build(events, seed_offset=0, raw=None):
    """Arrays for the statistics from events carrying 'det' and the schema's blind (and maybe full) fields.
    Only events with at least one other salient candidate are kept."""
    ev = [e for e in events if e["det"]["has_other"]]
    n = len(ev)
    out = dict(n=n, episode=np.array([e["seed"] for e in ev]),
               sA=np.array([e["sA"] for e in ev]), s_argA=np.array([e["s_argA"] for e in ev]),
               s_rest=np.array([e["s_rest"] for e in ev]))
    if n and "sI" in ev[0]:
        out["sI"] = np.array([e["sI"] for e in ev])
        out["sO"] = np.array([e["sO"] for e in ev])
    if raw is None:                           # the null's curves do not depend on the schema: reusable
        nr = np.zeros((n, len(QS), 3))
        na = np.zeros((n, len(QS)))
        for i, e in enumerate(ev):
            nr[i], na[i] = null_curve(e["det"], [e["seed"], e["t"], 5 + seed_offset])
    else:
        nr, na = raw
    out["_raw"] = (nr, na)
    # Diffuse share (review finding 1): null map = (1 - lam) * null + lam * uniform over the valid cells, lam per
    # event so that the null's mass outside the candidates equals the schema's (blind-safe). Then renormalise
    # over the candidate regions, exactly as the schema.
    cnt = np.array([e["det"]["counts"] for e in ev]).reshape(n, 3)
    nval = np.array([e["det"]["n_valid"] for e in ev], float)
    cand_share = cnt.sum(1) / nval
    r_null = 1.0 - nr.sum(2)                                   # null mass on shared cells, (n, nq)
    if EQUALIZE_DIFFUSE:
        lam = (out["s_rest"][:, None] - r_null) / np.maximum((1.0 - cand_share)[:, None] - r_null, 1e-12)
        lam = np.clip(lam, 0.0, 1.0)
    else:
        lam = np.zeros_like(r_null)
    out["lam"] = lam
    mixed = (1 - lam)[:, :, None] * nr + lam[:, :, None] * (cnt / nval[:, None])[:, None, :]
    mixed = mixed / np.maximum(mixed.sum(2, keepdims=True), 1e-12)
    out["nI"], out["nA"], out["nO"], out["n_argA"] = mixed[..., 0], mixed[..., 1], mixed[..., 2], na
    return out


# --------------------------------------------------------------------------------------- matching
def solve_q(curve, target, plateau=False):
    """First q on the grid where curve crosses target (linear interpolation). None if outside range, or (with
    plateau=True, used for the argmax reserve) if the curve stays within PLATEAU_TOL of the target over a q-span
    wider than PLATEAU, where q is not identified."""
    if target > curve.max() + 1e-12 or target < curve.min() - 1e-12:
        return None
    if plateau:
        near = QS[np.abs(curve - target) <= PLATEAU_TOL]
        if len(near) and near.max() - near.min() > PLATEAU:
            return None
    d = curve - target
    if abs(d[0]) < 1e-12:
        return float(QS[0])
    for i in range(len(QS) - 1):
        if d[i] == 0 or d[i] * d[i + 1] <= 0:
            if d[i] == d[i + 1]:
                return float(QS[i])
            f = d[i] / (d[i] - d[i + 1])
            return float(QS[i] + f * (QS[i + 1] - QS[i]))
    return None


def clip_q(curve, target):
    """q at the end of the grid nearest to target when there is no crossing."""
    return float(QS[int(np.argmin(np.abs(curve - target)))])


def interp(curve, q):
    return float(np.interp(q, QS, curve))


def _weights(arr, w):
    return (arr * (w[:, None] if arr.ndim == 2 else w)).sum(0) / w.sum()


def match_targets(d, w, method):
    """(schema value, null curve) for the chosen matching method, weighted by event weights w."""
    if method == "mass":
        return _weights(d["sA"], w), _weights(d["nA"], w)
    return _weights(d["s_argA"], w), _weights(d["n_argA"], w)


def feasibility(d):
    """Blind: which matching works for this run. Uses only quantities about the actual window."""
    w = np.ones(d["n"])
    res = dict(n_events=int(d["n"]), n_episodes=int(len(np.unique(d["episode"]))))
    for m in ("mass", "argmax"):
        t, c = match_targets(d, w, m)
        q = solve_q(c, t, plateau=m == "argmax")
        bar = None
        if q is not None:                          # the null's share on I at the matched q: the bar (blind-safe)
            nI, nO = _weights(d["nI"], w), _weights(d["nO"], w)
            bar = interp(nI, q) / max(interp(nI, q) + interp(nO, q), 1e-12)
        res[m] = dict(bar=bar, schema=float(t), null_q0=float(c[0]), null_q1=float(c[-1]),
                      null_min=float(c.min()), null_max=float(c.max()), q=q, possible=q is not None)
    res["schema_rest_mass"] = float(d["s_rest"].mean())
    return res


def level_method(feas_per_seed):
    """Reserve order, decided 24 Oct: mass in every seed, else argmax in every seed, else not assessable."""
    for m in ("mass", "argmax"):
        if all(f[m]["possible"] for f in feas_per_seed):
            return m
    return None


def share(I, O, w):
    return float((I * w).sum() / max(((I + O) * w).sum(), 1e-12))


def evaluate(d, method, n_boot=N_BOOT, seed=0):
    """Unblinded H5 for one run (needs sI, sO). Returns point estimates, bootstrap SE and pass/fail."""
    assert "sI" in d, "evaluate needs the schema's I and O (unblinded run)"
    ep = d["episode"]
    uniq, inv = np.unique(ep, return_inverse=True)

    def once(w):
        t, c = match_targets(d, w, method)
        q = solve_q(c, t, plateau=method == "argmax")
        clipped = q is None
        if clipped:
            q = clip_q(c, t)
        nI = _weights(d["nI"], w)
        nO = _weights(d["nO"], w)
        null_sh = interp(nI, q) / max(interp(nI, q) + interp(nO, q), 1e-12)
        sch_sh = share(d["sI"], d["sO"], w)
        return sch_sh, null_sh, q, clipped

    w0 = np.ones(d["n"])
    s_sh, n_sh, q0, cl0 = once(w0)
    rng = np.random.default_rng(seed)
    diffs, clipped = [], 0
    for _ in range(n_boot):
        cnt = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))
        a, b_, _, c = once(cnt[inv].astype(float))
        diffs.append(a - b_)
        clipped += c
    se = float(np.std(diffs, ddof=1))
    diff = s_sh - n_sh
    return dict(method=method, n_events=int(d["n"]), n_episodes=int(len(uniq)), q=q0, point_clipped=cl0,
                schema_share=s_sh, null_share=n_sh, diff=diff, se=se, z=diff / se if se > 0 else float("nan"),
                passed=bool(diff > 2 * se), boot_clipped_frac=clipped / n_boot,
                schema_rest_mass=float(d["s_rest"].mean()))
