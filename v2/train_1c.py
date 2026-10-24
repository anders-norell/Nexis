"""Sprint 6 (1c): train policy + schema module with A2C; the module learns only through the policy's loss.

  python3 train_1c.py --seed 40 --out results/c6_e22.1_s40 --alarm --rate-target 22.1
  python3 train_1c.py --seed 40 --out results/c6_cap22.1_s40 --alarm --shuffle --rate-target 22.1

Same hyperparameters, environment seeding, checkpointing and evaluation as train_rl.py (the baseline is
trained with train_rl.py --schema off). Per log line, besides train_rl's fields: bits per step (t > 0), bits
at steps that follow a capture and at steps that do not, beta, and every --diag-every iterations the
critic's share of the gradient into the module (critic_norm, actor_norm, cos between them, total), computed
by repeating the backward pass with only the critic's (resp. actor's) loss. The capacity baseline also logs
the share of steps whose donor episode is still running.
"""
import argparse
import json
import os
import time
import numpy as np
from env import SearchEnv
from train_rl import Adam, env_config
from schema_1c import make_agent_1c, schema_from_z, module_grads_1c


def rollout(agent, env, B, rng, greedy=False, keep=True, test_shuffle=False):
    """Run B episodes. The module runs first at each step; its channel z_t is the policy's schema input at
    the same step. Capacity baseline (module.shuffle): the module's input row b is the donor episode's.
    test_shuffle (sanity check only): the policy gets another episode's z at test time."""
    G = env.cfg.G
    mod = agent.module
    obs = env.reset(B)
    h = agent.init_state(B)
    hm = mod.init_state(B)
    donor = rng.permutation(B)
    recs, mrecs = [], []
    ret = np.zeros(B)
    t = 0
    bits_sum = np.zeros(2); bits_n = np.zeros(2)          # [after no capture, after capture]
    prev_cap = np.zeros(B, bool)
    donor_running, donor_n = 0, 0
    while not env.done.all():
        idx = np.nonzero(~env.done)[0]
        xm, mm = mod.features(env, obs, t)
        src = donor[idx] if mod.shuffle else idx
        if mod.shuffle:
            donor_running += int((~env.done[src]).sum()); donor_n += len(idx)
        eps = rng.normal(0, 1, (len(idx), G * G))
        hms, z, bits, mcache = mod.forward(xm[src], mm[src], hm[idx], eps, open_=t > 0)
        hm[idx] = hms
        zf = np.zeros((B, G * G))
        zf[idx] = z
        valid = obs["action_mask"]
        zf = zf * valid
        if test_shuffle:
            zf = zf[donor] * valid
        if t > 0:
            pc = prev_cap[idx]
            for k in (0, 1):
                sel = pc == bool(k)
                bits_sum[k] += float(bits[sel].sum()); bits_n[k] += int(sel.sum())
        schema = schema_from_z(zf, B, G)
        x, m = agent.features(env, obs, t, schema)
        hs, logp, Vs, _, cache = agent.step(x[idx], m[idx], h[idx])
        h[idx] = hs
        if greedy:
            a_s = logp.argmax(1)
        else:
            a_s = (logp - np.log(-np.log(rng.random(logp.shape)))).argmax(1)
        a = np.zeros(B, int)
        a[idx] = a_s
        active = ~env.done
        obs, r, done, info = env.step(a)
        ret += r
        prev_cap = info["captured"] & info["active"]
        if keep:
            V = np.zeros(B); V[idx] = Vs
            recs.append(dict(idx=idx, cache=cache, logp=logp, a=a_s, V=V, r=r, active=active,
                             zmask=valid[idx]))
            mrecs.append(dict(idx=idx, cache=mcache, bits=bits, open=t > 0))
        t += 1
    tot = bits_sum.sum() / max(bits_n.sum(), 1)
    stats = dict(found=env.found.copy(), steps=env.steps.copy(), ret=ret, mrecs=mrecs, bits=tot,
                 bits_nocap=bits_sum[0] / max(bits_n[0], 1), bits_cap=bits_sum[1] / max(bits_n[1], 1),
                 n_open=int(bits_n.sum()),
                 donor_running=donor_running / max(donor_n, 1) if mod.shuffle else float("nan"))
    return recs, stats


def a2c_grads_1c(agent, recs, gamma, lam, ent_coef, vf_coef, actor=1.0, critic=1.0):
    """A2C gradients for the policy, and per step d loss / d z (the gradient that trains the module).
    actor / critic scale the two parts of the loss (diagnostics: one of them 0)."""
    T = len(recs)
    B = recs[0]["a"].shape[0]
    act = np.stack([r["active"] for r in recs]).astype(float)
    rew = np.stack([r["r"] for r in recs]) * act
    V = np.stack([r["V"] for r in recs]) * act
    adv = np.zeros((T, B))
    last = np.zeros(B)
    for t in reversed(range(T)):
        nv = V[t + 1] * act[t + 1] if t + 1 < T else np.zeros(B)
        delta = rew[t] + gamma * nv - V[t]
        last = delta + gamma * lam * (act[t + 1] if t + 1 < T else 0.0) * last
        adv[t] = last * act[t]
    rets = adv + V
    n_act = act.sum()
    a_mean = (adv * act).sum() / n_act
    a_std = np.sqrt((((adv - a_mean) * act) ** 2).sum() / n_act) + 1e-8
    nadv = (adv - a_mean) / a_std * act
    grads = agent.zero_grads()
    G2, d = agent.cfg.G ** 2, agent.d
    dh = np.zeros((B, G2, d), agent.dtype)
    dzs = [None] * T
    ent_sum, vloss = 0.0, 0.0
    for t in reversed(range(T)):
        rc = recs[t]
        idx = rc["idx"]
        logp = rc["logp"]
        nb = len(idx)
        p = np.exp(logp)
        onehot = np.zeros_like(p)
        onehot[np.arange(nb), rc["a"]] = 1.0
        plogp = np.where(p > 0, p * logp, 0.0)
        H = -plogp.sum(1)
        ent_sum += H.sum()
        dlogit = -nadv[t][idx][:, None] * (onehot - p)
        dlogit += ent_coef * (plogp + p * H[:, None])
        dlogit *= actor / n_act
        dV = critic * vf_coef * (V[t][idx] - rets[t][idx]) / n_act
        vloss += ((V[t][idx] - rets[t][idx]) ** 2).sum()
        dt = lambda z: z.astype(agent.dtype)
        dhs, dz = agent.step_backward_x(rc["cache"], dt(dlogit), dt(dV), dt(np.zeros_like(p)), dh[idx], grads)
        dh = np.zeros_like(dh)
        dh[idx] = dhs
        dzs[t] = dz * rc["zmask"]
    gn = np.sqrt(sum((g ** 2).sum() for g in grads.values()))
    info = dict(entropy=ent_sum / n_act, vloss=vloss / n_act, grad_norm=float(gn), n_act=float(n_act))
    return grads, dzs, info


def _norm(g):
    return float(np.sqrt(sum((v.astype(np.float64) ** 2).sum() for v in g.values())))


def critic_diagnostic(agent, recs, mrecs, gamma, lam, ent, vf):
    """Gradient into the module from the critic's loss and from the actor's (policy + entropy) loss, each
    without the information cost. Reported, not used for training."""
    mod = agent.module
    beta, mod.beta = mod.beta, 0.0
    try:
        _, dzc, info = a2c_grads_1c(agent, recs, gamma, lam, ent, vf, actor=0.0, critic=1.0)
        _, dza, _ = a2c_grads_1c(agent, recs, gamma, lam, ent, vf, actor=1.0, critic=0.0)
        gc = module_grads_1c(mod, mrecs, dzc, 1.0 / info["n_act"])
        ga = module_grads_1c(mod, mrecs, dza, 1.0 / info["n_act"])
    finally:
        mod.beta = beta
    nc, na = _norm(gc), _norm(ga)
    dot = sum(float((gc[k].astype(np.float64) * ga[k]).sum()) for k in gc)
    tot = _norm({k: gc[k] + ga[k] for k in gc})
    return dict(critic_norm=nc, actor_norm=na, cos=dot / max(nc * na, 1e-30), total_norm=tot,
                critic_share=nc / max(nc + na, 1e-30))


def evaluate(agent, n_ep, seed, B=500, greedy=False, test_shuffle=False):
    env = SearchEnv(agent.cfg, seed=seed)
    rng = np.random.default_rng(seed + 1)
    found, ret, bits = [], [], []
    for _ in range(n_ep // B):
        _, st = rollout(agent, env, B, rng, greedy=greedy, keep=False, test_shuffle=test_shuffle)
        found.append(st["found"]); ret.append(st["ret"]); bits.append(st["bits"])
    found, ret = np.concatenate(found), np.concatenate(ret)
    return dict(found=float(found.mean()), found_se=float(found.std() / np.sqrt(len(found))),
                ret=float(ret.mean()), bits=float(np.mean(bits)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--minutes", type=float, default=100)
    ap.add_argument("--max-episodes", type=int, default=10_000_000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--dh", type=int, default=32)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--ent-final", type=float, default=0.001)
    ap.add_argument("--ent-episodes", type=int, default=3_000_000)
    ap.add_argument("--vf", type=float, default=0.5)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--eval-episodes", type=int, default=2000)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--diag-every", type=int, default=100, help="critic/actor gradient diagnostic interval")
    ap.add_argument("--reveal", type=float, default=1.0)
    ap.add_argument("--p-capture", type=float, default=0.25)
    ap.add_argument("--windows", default="1,2")
    ap.add_argument("--alarm", action="store_true", help="'with': 1b-E's input (matching signal + alarm)")
    ap.add_argument("--shuffle", action="store_true", help="matched capacity baseline: donor's input")
    ap.add_argument("--rate-cost", type=float, default=0.001, help="cost per bit and step (start value)")
    ap.add_argument("--rate-target", type=float, default=0.0, help="bits per step to hold (0 = fixed cost)")
    ap.add_argument("--rate-eta", type=float, default=0.05)
    args = ap.parse_args()
    args.schema = "1c"                              # recorded in checkpoints for loaders

    cfg = env_config(args)
    agent = make_agent_1c(cfg, d=args.d, dh=args.dh, seed=args.seed, alarm=args.alarm, shuffle=args.shuffle)
    mod = agent.module
    mod.beta = args.rate_cost
    opt, mopt = Adam(agent.p, args.lr), Adam(mod.p, args.lr)
    env = SearchEnv(cfg, seed=10_000 + args.seed)
    rng = np.random.default_rng(20_000 + args.seed)
    ckpt = args.out + ".ckpt.npz"
    it, episodes, best = 0, 0, -1.0
    if os.path.exists(ckpt):
        meta = agent.load(ckpt)
        z = np.load(ckpt)
        for k in agent.p:
            opt.m[k], opt.v[k] = z[f"am_{k}"], z[f"av_{k}"]
        for k in mod.p:
            mopt.m[k], mopt.v[k] = z[f"qm_{k}"], z[f"qv_{k}"]
        opt.t = mopt.t = int(meta["opt_t"]); it = int(meta["it"]); episodes = int(meta["episodes"])
        best = float(meta["best"]); mod.beta = float(meta["beta"])
        env = SearchEnv(cfg, seed=10_000 + args.seed + 7919 * it)
        rng = np.random.default_rng(20_000 + args.seed + 7919 * it)
        print(f"resumed at it {it}, {episodes} episodes", flush=True)

    def save_ckpt():
        np.savez(ckpt, **{f"q_{k}": v for k, v in mod.p.items()}, **{f"qm_{k}": v for k, v in mopt.m.items()},
                 **{f"qv_{k}": v for k, v in mopt.v.items()}, **{f"p_{k}": v for k, v in agent.p.items()},
                 **{f"am_{k}": v for k, v in opt.m.items()}, **{f"av_{k}": v for k, v in opt.v.items()},
                 m_opt_t=opt.t, m_it=it, m_episodes=episodes, m_best=best, m_beta=mod.beta,
                 m_args=json.dumps(vars(args)))

    log = open(args.out + ".log.jsonl", "a")
    t0, acc = time.time(), []
    while time.time() - t0 < args.minutes * 60 and episodes < args.max_episodes:
        frac = min(1.0, episodes / args.ent_episodes)
        ent = args.ent + (args.ent_final - args.ent) * frac
        recs, st = rollout(agent, env, args.batch, rng)
        grads, dzs, info = a2c_grads_1c(agent, recs, args.gamma, args.lam, ent, args.vf)
        mg = module_grads_1c(mod, st["mrecs"], dzs, 1.0 / info["n_act"])
        diag = None
        if args.diag_every and (it + 1) % args.diag_every == 0:
            diag = critic_diagnostic(agent, recs, st["mrecs"], args.gamma, args.lam, ent, args.vf)
        if info["grad_norm"] > args.clip:
            for k in grads:
                grads[k] *= args.clip / info["grad_norm"]
        opt.step(agent.p, grads)
        mn = _norm(mg)
        if mn > args.clip:
            for k in mg:
                mg[k] *= args.clip / mn
        mopt.step(mod.p, mg)
        if args.rate_target > 0:
            mod.beta = float(np.clip(mod.beta * np.exp(args.rate_eta * (st["bits"] - args.rate_target)
                                                       / args.rate_target), 1e-7, 10.0))
        it += 1
        episodes += args.batch
        acc.append((st["found"].mean(), st["ret"].mean(), info["entropy"], info["vloss"], info["grad_norm"],
                    mn, st["bits"], st["bits_nocap"], st["bits_cap"], st["donor_running"]))
        if diag is not None:
            log.write(json.dumps(dict(it=it, episodes=episodes, diag=diag)) + "\n"); log.flush()
        if it % args.log_every == 0:
            a = np.mean(np.array(acc, float), 0)          # donor_running is NaN unless --shuffle
            acc = []
            rec = dict(it=it, episodes=episodes, minutes=round((time.time() - t0) / 60, 2),
                       train_found=round(a[0], 4), train_ret=round(a[1], 4), entropy=round(a[2], 3),
                       vloss=round(a[3], 4), grad_norm=round(a[4], 3), mod_grad_norm=round(a[5], 5),
                       bits=round(a[6], 3), bits_nocap=round(a[7], 3), bits_cap=round(a[8], 3),
                       beta=mod.beta, ent_coef=round(ent, 5))
            if args.shuffle:
                rec["donor_running"] = round(float(a[9]), 4)
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(rec, flush=True)
        if it % args.eval_every == 0:
            ev = evaluate(agent, args.eval_episodes, seed=500_000 + args.seed)
            rec = dict(it=it, episodes=episodes, eval=ev)
            log.write(json.dumps(rec) + "\n"); log.flush()
            print("EVAL", rec, flush=True)
            if ev["found"] > best:
                best = ev["found"]
                agent.save(args.out + ".best.npz", it=it, episodes=episodes, eval_found=best,
                           args=json.dumps(vars(args)), beta=mod.beta)
            save_ckpt()
    save_ckpt()
    agent.save(args.out + ".final.npz", it=it, episodes=episodes, args=json.dumps(vars(args)), beta=mod.beta)
    print("done", it, episodes, flush=True)


if __name__ == "__main__":
    main()
