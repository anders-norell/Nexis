"""Sprint 5a step 1: diagnostic of the H2 asymmetry on sprint 4's agents (descriptive, no criterion).

Approved plan (spec, "Sprint 5: H5", plan 5a, step 1). All comparisons use only cells with an object of
the target's colour (never the target itself), and not salient objects (technical correction before the
full run: a salient object is never the target and the agent never aims at it, so capture-only cells,
which are mostly the salient object, would have shown no effect for that reason alone). Three explanations, each with a test and a prediction
fixed before running:

  E1  own memory of aim and glimpse (E2 merged into it). (a)-cells split into cells the agent aimed at in
      steps 1-3 and cells covered only through capture. E1 predicts a larger effect for capture-only cells.
  E3  plausibility. Variant a3: only the cells first covered by step 3's actual window are shown as not
      covered (as after a single capture). Per-cell effect on those cells, a3 vs the full (a), split by
      whether step 3 was a capture. E3 predicts a larger per-cell effect under a3 in both groups.
  E4  the asymmetry is in the measure. (i) (a) split by the number of truly unvisited target-colour
      cells left (competitors). E4 predicts the (a) effect shrinks with more competitors. (ii) normalised
      effects: share of the gap to full trust that each lie closes,
        (a): (A_with - A_without) / (U_without - A_without)
        (b): (B_without - B_with) / (B_without - A_without)
      where U = truly unvisited target-colour cells (not the target, not (b)-cells), A = covered
      target-colour cells. E4 predicts the asymmetry shrinks in the normalised measure.
  Plus: (a) and (b) on the final checkpoints, and (b) per command and in episodes with all three commands.

Per-cell measure: number of the next three commands (4th-6th) whose intended window covers the cell.
Paired per bank episode (with lie minus without), SE over episodes; ratios get bootstrap SE (1 000).

The runner reproduces bank._run_block exactly (same streams, same cell choice, same h2 draws); this is
checked against bank.run_bank before anything else is computed.

  python3 diag_h2.py            writes results/sprint5_diag_h2.json
"""
import json
import os
import sys
from multiprocessing import Pool
import numpy as np
from bank import BankEnv, streams, bank_seeds, run_bank, _window_mask, H2_T, H2_SPAN, BLOCK
from eval_mvp import load, RES

SEEDS = (10, 11, 12)
NBOOT = 1000


def run_block(agent, seeds, mode, intervention):
    """As bank._run_block, plus intervention 'a3' and per-cell records."""
    cfg = agent.cfg
    G = cfg.G
    B = len(seeds)
    st = [streams(s) for s in seeds]
    env = BankEnv(cfg)
    obs = env.reset_bank([x[0] for x in st])
    arng = [x[1] for x in st]
    hrng = [x[2] for x in st]
    h = agent.init_state(B)
    donor = (np.arange(B) + 1) % B if B > 1 else np.zeros(1, int)
    src = donor if mode == "shuffled" else np.arange(B)
    vis = np.zeros((B, G, G))
    post = np.zeros((B, G, G))
    cur = np.zeros((B, G, G))
    A = np.zeros((B, G, G), bool)
    Bm = np.zeros((B, G, G), bool)
    S3 = np.zeros((B, G, G), bool)          # cells first covered by step 3 (t = 2)
    cap3 = np.zeros(B, bool)
    intended_union = np.zeros((B, G, G), bool)
    vis_before3 = np.zeros((B, G, G))
    elig = np.zeros(B, bool)
    n_cand_b = np.zeros(B, int)
    lie = np.zeros(B, bool)
    aimc = np.zeros((B, G, G), int)          # per-cell aim count, commands 4-6
    ncmd = np.zeros(B, int)
    aim_a = np.zeros(B, int)
    aim_b = np.zeros(B, int)
    tcol = env.color == env.target_color[:, None, None]
    tmask = np.zeros((B, G, G), bool)
    tmask[np.arange(B), env.target[:, 0], env.target[:, 1]] = True
    L = None                                  # cells whose mark is removed for a / a3
    t = 0
    while not env.done.all():
        if t == H2_T:
            running = ~env.done
            for b in range(B):
                cand = np.argwhere(tcol[b] & ~tmask[b] & (vis[b] == 0) & env.valid[b])
                n_cand_b[b] = len(cand)
                g = hrng[b]
                pick = g.permutation(len(cand))[:3] if len(cand) else []
                for j in pick:
                    Bm[b, cand[j][0], cand[j][1]] = True
                A[b] = vis[b] > 0
                n, w = env.n[b], env.w[b]
                r0, c0 = env.last_actual[b]
                opts = [(r, c) for r in range(n - w + 1) for c in range(n - w + 1)
                        if not (_window_mask(G, r, c, w) & _window_mask(G, r0, c0, w)).any()]
                if opts:
                    g.integers(len(opts))             # same draw as bank (for the (c) window)
            S3[:] = (vis > 0) & (vis_before3 == 0)
            elig = running.copy()
            L = A if intervention == "a" else S3
            if intervention in ("a", "a3"):
                lie = elig.copy()
            elif intervention == "b":
                lie = elig & (n_cand_b > 0)
        if mode == "off" or t == 0:
            schema = None
        else:
            sv = vis[src].copy()
            sc = cur[src].copy()
            if intervention in ("a", "a3") and t >= H2_T:
                sv[lie] = np.where(L[lie], post[src][lie], sv[lie])
            elif intervention == "b" and t >= H2_T:
                sv[lie] = np.where(Bm[lie], 1.0, sv[lie])
            schema = np.stack([sc, sv], -1)
        idx = np.nonzero(~env.done)[0]
        x, m = agent.features(env, obs, t, schema)
        hs, logp, _, _, _ = agent.step(x[idx], m[idx], h[idx])
        h[idx] = hs
        a = np.zeros(B, int)
        for j, b in enumerate(idx):
            u = arng[b].random(logp.shape[1])
            a[b] = int((logp[j] - np.log(-np.log(u))).argmax())
        obs, r, done, info = env.step(a)
        ir0, ic0 = info["intended"].T
        r0, c0 = info["actual"].T
        newcur = np.zeros((B, G, G))
        for b in np.nonzero(info["active"])[0]:
            w = env.w[b]
            iw = _window_mask(G, ir0[b], ic0[b], w)
            if t < H2_T:
                intended_union[b] |= iw
            if H2_T <= t < H2_T + H2_SPAN:
                aimc[b] += iw
                ncmd[b] += 1
                aim_a[b] += bool((iw & A[b]).any())
                aim_b[b] += bool((iw & Bm[b]).any())
            aw = _window_mask(G, r0[b], c0[b], w)
            newcur[b] = aw
            if t >= H2_T:
                post[b] = np.maximum(post[b], aw)
        if t == H2_T - 1:
            cap3 = info["captured"] & info["active"]
            vis_before3 = vis.copy()
        cur = newcur
        vis = np.maximum(vis, cur)
        t += 1
    # cells worth a visit: target colour, not the target, not a salient object (a salient object is never
    # the target and the periphery shows its salience, so the agent never aims at it; added after a
    # 500-episode check showed capture-only cells were almost all salient objects, see spec)
    tc = tcol & ~tmask & env.valid & ~(env.salient > 0)
    U = tc & ~A & ~Bm
    Bm_tc = Bm & tc
    return dict(elig=elig, n_cand_b=n_cand_b, ncmd=ncmd, aim_a=aim_a, aim_b=aim_b, aimc=aimc,
                A_aim=A & tc & intended_union, A_cap=A & tc & ~intended_union, A_tc=A & tc,
                B=Bm_tc, U=U, S3=S3 & tc, cap3=cap3, found=env.found.copy())


def run_all(agent, mode, intervention, seeds):
    out = [run_block(agent, seeds[s:s + BLOCK], mode, intervention) for s in range(0, len(seeds), BLOCK)]
    return {k: np.concatenate([o[k] for o in out], 0) for k in out[0]}


def cell_rate(r, mask):
    """Per-episode mean aim count over the cells in mask (nan where the episode has none)."""
    n = mask.sum((1, 2))
    s = (r["aimc"] * mask).sum((1, 2))
    return np.where(n > 0, s / np.maximum(n, 1), np.nan)


def paired(x_with, x_without, sel):
    d = (x_with - x_without)[sel & ~np.isnan(x_with) & ~np.isnan(x_without)]
    n = len(d)
    return dict(n=int(n), mean=float(d.mean()) if n else float("nan"),
                se=float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan"))


def ratio_boot(num, den, seed=0):
    ok = ~np.isnan(num) & ~np.isnan(den)
    num, den = num[ok], den[ok]
    est = float(num.mean() / den.mean()) if len(num) and den.mean() != 0 else float("nan")
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(NBOOT):
        i = rng.integers(0, len(num), len(num))
        if den[i].mean() != 0:
            vals.append(num[i].mean() / den[i].mean())
    return dict(n=int(len(num)), est=est, se=float(np.std(vals)) if vals else float("nan"))


def analyse(base, ra, rb, ra3):
    e = base["elig"]
    out = {}
    # E1: aimed vs capture-only covered target-colour cells, under the full (a)
    out["E1"] = {g: paired(cell_rate(ra, base[g]), cell_rate(base, base[g]), e) for g in ("A_aim", "A_cap")}
    # E3: step-3 cells, a3 vs full a, split by capture at step 3
    out["E3"] = {}
    for name, sel in (("cap3", e & base["cap3"]), ("nocap3", e & ~base["cap3"])):
        out["E3"][name] = dict(a3=paired(cell_rate(ra3, base["S3"]), cell_rate(base, base["S3"]), sel),
                               a_full=paired(cell_rate(ra, base["S3"]), cell_rate(base, base["S3"]), sel))
    # E4 (i): (a) on covered target-colour cells by number of competitors
    nU = base["U"].sum((1, 2))
    out["E4_competitors"] = {}
    for lo, hi in ((0, 2), (3, 5), (6, 99)):
        sel = e & (nU >= lo) & (nU <= hi)
        out["E4_competitors"][f"{lo}-{hi}"] = paired(cell_rate(ra, base["A_tc"]), cell_rate(base, base["A_tc"]), sel)
    # E4 (ii): normalised share of the gap closed
    A0, U0 = cell_rate(base, base["A_tc"]), cell_rate(base, base["U"])
    A1 = cell_rate(ra, base["A_tc"])
    B0, B1 = cell_rate(base, base["B"]), cell_rate(rb, base["B"])
    sel_a = e & ~np.isnan(A0) & ~np.isnan(U0) & ~np.isnan(A1)
    sel_b = e & ~np.isnan(B0) & ~np.isnan(B1) & ~np.isnan(A0)
    out["E4_norm"] = dict(
        a=ratio_boot(np.where(sel_a, A1 - A0, np.nan), np.where(sel_a, U0 - A0, np.nan), 1),
        b=ratio_boot(np.where(sel_b, B0 - B1, np.nan), np.where(sel_b, B0 - A0, np.nan), 2),
        raw=dict(A_without=float(np.nanmean(A0[sel_a])), A_with=float(np.nanmean(A1[sel_a])),
                 U_without=float(np.nanmean(U0[sel_a])), B_without=float(np.nanmean(B0[sel_b])),
                 B_with=float(np.nanmean(B1[sel_b])), A_without_b=float(np.nanmean(A0[sel_b]))))
    # (b) per command, and in episodes where all three commands were given in both runs
    eb = e & (base["n_cand_b"] > 0)
    pc = lambda r: np.where(r["ncmd"] > 0, r["aim_b"] / np.maximum(r["ncmd"], 1), np.nan)
    out["b_per_command"] = paired(pc(rb), pc(base), eb)
    full3 = eb & (base["ncmd"] == 3) & (rb["ncmd"] == 3)
    out["b_full3"] = paired(rb["aim_b"].astype(float), base["aim_b"].astype(float), full3)
    out["locked_check"] = dict(a=paired(ra["aim_a"].astype(float), base["aim_a"].astype(float), e),
                               b=paired(rb["aim_b"].astype(float), base["aim_b"].astype(float), eb))
    return out


def job(args):
    name, ckpt = args
    import eval_mvp
    path = os.path.join(RES, name + f".{ckpt}.npz")
    z = np.load(path)
    margs = json.loads(str(z["m_args"]))
    from env import EnvConfig
    from agent_rl import make_agent
    cfg = EnvConfig(reveal=margs["reveal"], p_capture=margs["p_capture"],
                    windows=tuple(int(x) for x in str(margs["windows"]).split(",")))
    ag = make_agent(cfg, d=margs["d"], dh=margs["dh"], seed=margs["seed"], schema_mode=margs["schema"])
    ag.load(path)
    seeds = bank_seeds()
    mode = ag.schema_mode
    base = run_all(ag, mode, None, seeds)
    ra = run_all(ag, mode, "a", seeds)
    rb = run_all(ag, mode, "b", seeds)
    if ckpt == "best":
        ra3 = run_all(ag, mode, "a3", seeds)
        res = analyse(base, ra, rb, ra3)
    else:
        e = base["elig"]
        eb = e & (base["n_cand_b"] > 0)
        res = dict(locked_check=dict(a=paired(ra["aim_a"].astype(float), base["aim_a"].astype(float), e),
                                     b=paired(rb["aim_b"].astype(float), base["aim_b"].astype(float), eb)))
    return f"{name}|{ckpt}", res


def check_reproduces_bank():
    """The diagnostic runner must give exactly bank.run_bank's numbers (first 500 bank episodes)."""
    for cond in ("fixed", "shuffled"):
        ag, _ = load(f"s4_{cond}_s10")
        seeds = bank_seeds(500)
        for iv in (None, "a", "b"):
            mine = run_all(ag, ag.schema_mode, iv, seeds)
            ref = run_bank(ag, seeds, intervention=iv)
            for k in ("found", "aim_a", "aim_b"):
                assert np.array_equal(mine[k], ref[k]), (cond, iv, k)
            assert np.array_equal(mine["elig"], ref["elig_a"])
    print("diagnostic runner reproduces bank.run_bank exactly", flush=True)


def main():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    check_reproduces_bank()
    jobs = [(f"s4_{c}_s{s}", ck) for c in ("fixed", "shuffled") for s in SEEDS for ck in ("best", "final")]
    out = {}
    with Pool(2) as pool:
        for key, res in pool.imap_unordered(job, jobs):
            out[key] = res
            print(key, "done", flush=True)
            json.dump(out, open(os.path.join(RES, "sprint5_diag_h2.json"), "w"), indent=1)
    json.dump(out, open(os.path.join(RES, "sprint5_diag_h2.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
