"""Sprint 5a: run schema-module agents (mode 'channel', and 'off' for the baseline) on a seed bank, and
collect what the pilot and later 5b need.

Per episode the bank streams are used as in bank.py: env stream for the environment, agent stream for the
agent's own draws (the channel's noise each step, then the policy's sampling), h2 stream for H2 (b)'s
choice of cells. Episodes are independent of the rest of their batch.

Collected (blind mode, used in the pilot): hit rate; information per step (bits, steps t > 0); accuracy
over all steps (the map's most probable cell lies in last step's actual window); and at H5 events only
whether the map's most probable cell lies in the actual window, plus the restricted observer's mass on the
actual window. Where the map's errors go (intended window or elsewhere) and the behaviour outcome are not
computed in blind mode. For the behaviour measure's dose check only "group has >= N_MIN events" is
reported, never the counts, since the counts would show how often the map points at the intended place.

H5 event (spec): a capture at step t that does not find the target, in a running episode, where the fovea
excludes the intended window by occupied and colour (h5.excludes_intended_restricted). The map is read
at step t + 1.

Intervention 'b' (H2 (b) on 1b, built in 5a, run in 5b): before the 4th command, up to three uncovered
target-colour cells (never the target), chosen from the h2 stream exactly as bank.py does, are shown as
covered in the module's running map for the rest of the episode. "Aims there" = intended window overlaps.
"""
import numpy as np
from bank import BankEnv, streams, _window_mask, H2_T, H2_SPAN, BLOCK
from h5 import excludes_intended_restricted, observer_restricted
from h5_eval import event_detail, schema_regions

N_MIN = 200          # minimum events per behaviour group (dose check), per seed and level


def run_bank_module(agent, seeds, intervention=None, blind=True, block=BLOCK, overrides=None, record=False,
                    detail=False):
    """detail=True (gate 5a, 24 Oct): each H5 event also carries 'det' (h5_eval.event_detail) and the
    schema's region fields: blind-safe ones always, I and O only when blind=False."""
    out = [_block(agent, seeds[s:s + block], intervention, blind, overrides, record, detail)
           for s in range(0, len(seeds), block)]
    grp = np.sum([o.pop("_grp") for o in out], 0)          # (right, wrong) among behaviour-eligible events
    res = {}
    for k in out[0]:
        if isinstance(out[0][k], np.ndarray):
            res[k] = np.concatenate([o[k] for o in out], 0)
        else:
            res[k] = sum((o[k] for o in out), [])
    res["dose_groups"] = dict(right_ok=bool(grp[0] >= N_MIN), wrong_ok=bool(grp[1] >= N_MIN))
    if not blind:
        res["group_counts"] = dict(right=int(grp[0]), wrong=int(grp[1]))
    return res


def _block(agent, seeds, intervention, blind, overrides, record=False, detail=False):
    cfg = agent.cfg
    G = cfg.G
    B = len(seeds)
    st = [streams(s) for s in seeds]
    env = BankEnv(cfg)
    obs = env.reset_bank([x[0] for x in st], overrides)
    arng = [x[1] for x in st]
    hrng = [x[2] for x in st]
    mod = getattr(agent, "module", None)
    channel = agent.schema_mode in ("channel", "channel_e")
    h = agent.init_state(B)
    hm = mod.init_state(B) if channel else None
    union = np.zeros((B, G, G))
    vis = np.zeros((B, G, G))
    Bm = np.zeros((B, G, G), bool)
    elig_b = np.zeros(B, bool)
    aim_b = np.zeros(B, int)
    tcol = env.color == env.target_color[:, None, None]
    tmask = np.zeros((B, G, G), bool)
    tmask[np.arange(B), env.target[:, 0], env.target[:, 1]] = True
    prev_actual = np.zeros((B, G, G))
    bits_sum = np.zeros(B)
    bits_n = np.zeros(B, int)
    acc_n = np.zeros(B, int)
    acc_ok = np.zeros(B, int)
    pending = {}
    events = []
    grp = np.zeros(2, int)
    trace = []
    t = 0
    schema = None
    while not env.done.all():
        idx = np.nonzero(~env.done)[0]
        if t == H2_T and intervention == "b":
            running = ~env.done
            for b in range(B):
                cand = np.argwhere(tcol[b] & ~tmask[b] & (vis[b] == 0) & env.valid[b])
                g = hrng[b]
                pick = g.permutation(len(cand))[:3] if len(cand) else []
                for j in pick:
                    Bm[b, cand[j][0], cand[j][1]] = True
            elig_b = running & Bm.any((1, 2))
        if channel:
            xm, mm = mod.features(env, obs, t)
            eps = np.stack([arng[b].normal(0, 1, G * G) for b in idx])
            hms, slm, bits, _ = mod.forward(xm[idx], mm[idx], hm[idx], eps)
            hm[idx] = hms
            cur = np.zeros((B, G * G))
            if t > 0:
                cur[idx] = np.exp(slm)
                bits_sum[idx] += bits
                bits_n[idx] += 1
                for j, b in enumerate(idx):
                    cell = np.unravel_index(int(np.argmax(cur[b])), (G, G))
                    acc_n[b] += 1
                    acc_ok[b] += bool(prev_actual[b][cell])
            cur = cur.reshape(B, G, G)
            union = np.maximum(union, cur)
            shown = union.copy()
            if intervention == "b" and t >= H2_T:
                lie = elig_b[:, None, None] & Bm
                shown = np.where(lie, 1.0, shown)
            schema = np.stack([cur, shown], -1)
            for b, ev in list(pending.items()):          # read the map the step after the capture
                p = cur[b]
                cell = np.unravel_index(int(np.argmax(p)), (G, G))
                ev["map_argmax_actual"] = bool(ev["_ma"][cell])
                ev["map_mass_actual"] = float((p * ev["_ma"]).sum())
                if ev["behaviour_eligible"]:          # dose check counts, kept in memory only
                    grp[0] += ev["map_argmax_actual"]
                    grp[1] += bool(ev["_mi"][cell])
                if not blind:
                    ev["group_wrong"] = bool(ev["_mi"][cell])
                if "det" in ev:
                    sb, sf = schema_regions(p, ev["det"])
                    ev.update(sb)
                    if not blind:
                        ev.update(sf)
                for k in ("_ma", "_mi"):
                    ev.pop(k)
                events.append(ev)
            pending = {}
        x, m = agent.features(env, obs, t, schema)
        hs, logp, _, _, _ = agent.step(x[idx], m[idx], h[idx])
        h[idx] = hs
        a = np.zeros(B, int)
        for j, b in enumerate(idx):
            u = arng[b].random(logp.shape[1])
            a[b] = int((logp[j] - np.log(-np.log(u))).argmax())
        if record:
            trace.append(dict(t=t, a=a.copy(), schema=None if schema is None else schema.copy()))
        obs, r, done, info = env.step(a)
        ir0, ic0 = info["intended"].T
        r0, c0 = info["actual"].T
        newcur = np.zeros((B, G, G))
        for b in np.nonzero(info["active"])[0]:
            w = env.w[b]
            iw = _window_mask(G, ir0[b], ic0[b], w)
            if intervention == "b" and H2_T <= t < H2_T + H2_SPAN:
                aim_b[b] += bool((iw & Bm[b]).any())
            newcur[b] = _window_mask(G, r0[b], c0[b], w)
        if channel:
            for b in np.nonzero(info["captured"] & info["active"] & ~info["hit"] & ~env.done)[0]:
                if not excludes_intended_restricted(env, b, ir0[b], ic0[b]):
                    continue
                w = env.w[b]
                mi = _window_mask(G, ir0[b], ic0[b], w)
                ma = _window_mask(G, r0[b], c0[b], w) & ~mi
                post = observer_restricted(env, b, ir0[b], ic0[b], env.last_window[b])
                obs_actual = sum(p for o, p, k in post if tuple(o) == (r0[b], c0[b]) and k != "intended")
                unseen_tc = tcol[b] & mi & ((vis[b] + newcur[b]) == 0)
                pending[b] = dict(seed=int(seeds[b]), t=t, w=int(w), observer_actual=float(obs_actual),
                                  behaviour_eligible=bool(unseen_tc.any()), _ma=ma, _mi=mi)
                if detail:
                    pending[b]["det"] = event_detail(env, b, ir0[b], ic0[b], r0[b], c0[b],
                                                     int(info["capturer"][b]), visited=vis[b])
        prev_actual = newcur
        vis = np.maximum(vis, newcur)
        t += 1
    return dict(found=env.found.copy(), steps=env.steps.copy(), bits=bits_sum / np.maximum(bits_n, 1),
                bits_n=bits_n, acc_ok=acc_ok, acc_n=acc_n, elig_b=elig_b, aim_b=aim_b, events=events, _grp=grp,
                **(dict(trace=[trace], cellsB=[Bm & elig_b[:, None, None]]) if record else {}))


def summary(res):
    ev = res["events"]
    return dict(found=float(res["found"].mean()),
                found_se=float(res["found"].std() / np.sqrt(len(res["found"]))),
                bits_per_step=float((res["bits"] * res["bits_n"]).sum() / max(res["bits_n"].sum(), 1)),
                acc_all_steps=float(res["acc_ok"].sum() / max(res["acc_n"].sum(), 1)),
                n_h5_events=len(ev),
                acc_at_h5=float(np.mean([e["map_argmax_actual"] for e in ev])) if ev else float("nan"),
                mass_actual_at_h5=float(np.mean([e["map_mass_actual"] for e in ev])) if ev else float("nan"),
                observer_actual_at_h5=float(np.mean([e["observer_actual"] for e in ev])) if ev else float("nan"))
