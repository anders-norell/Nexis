"""Train the priority-map agent with A2C (GAE, entropy bonus) on the v2 search environment.

  python3 train_rl.py --seed 0 --minutes 100 --out results/s2_base_s0

Resumable: if <out>.ckpt.npz exists, training continues from it (weights, Adam state, counters).
Logs one JSON line per log interval to <out>.log.jsonl; evaluates on held-out episodes every
--eval-every iterations (eval seed differs from the training seed).
"""
import argparse
import json
import os
import time
import numpy as np
from env import EnvConfig, SearchEnv
from agent_rl import PriorityMapAgent, schema_input


class Adam:
    def __init__(self, params, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for k in params:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * grads[k] ** 2
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)


def rollout(agent, env, B, rng, greedy=False, keep=True, sl_weight=0.0, overrides=None):
    """Run B episodes to the end. Returns per-step records (for learning) and episode stats."""
    G = env.cfg.G
    obs = env.reset(B, overrides)
    h = agent.init_state(B)
    visited = np.zeros((B, G, G))
    schema, pred = None, None
    donor = rng.permutation(B)
    recs, t = [], 0
    ret = np.zeros(B)
    prev_actual = None
    while not env.done.all():
        idx = np.nonzero(~env.done)[0]                       # only running episodes are computed
        x, m = agent.features(env, obs, t, schema)
        hs, logp, Vs, slogp, cache = agent.step(x[idx], m[idx], h[idx])
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
        if keep:
            V = np.zeros(B); V[idx] = Vs
            recs.append(dict(idx=idx, cache=cache, logp=logp, a=a_s, V=V, r=r, active=active,
                             slogp=slogp, sl_target=None if prev_actual is None else prev_actual[idx]))
        cur = np.zeros((B, G, G))
        r0, c0 = info["actual"].T
        for b in np.nonzero(info["active"])[0]:
            w = env.w[b]
            cur[b, r0[b]:r0[b] + w, c0[b]:c0[b] + w] = 1.0
        prev_actual = np.where(info["active"][:, None], cur.reshape(B, -1), 0.0)
        pred = np.zeros((B, G * G)); pred[idx] = np.exp(slogp)
        schema, visited = schema_input(agent.schema_mode, env, info, visited, pred, donor)
        t += 1
    stats = dict(found=env.found.copy(), steps=env.steps.copy(), ret=ret)
    return recs, stats


def a2c_grads(agent, recs, gamma, lam, ent_coef, vf_coef, sl_coef):
    T = len(recs)
    B = recs[0]["a"].shape[0]
    act = np.stack([r["active"] for r in recs]).astype(float)       # (T, B)
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
    ent_sum, vloss, sl_loss, n_sl = 0.0, 0.0, 0.0, 0.0
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
        # loss = -A logpi(a) - c_e H ; gradients w.r.t. logits
        dlogit = -nadv[t][idx][:, None] * (onehot - p)
        dlogit += ent_coef * (plogp + p * H[:, None])
        dlogit /= n_act
        dV = vf_coef * (V[t][idx] - rets[t][idx]) / n_act
        vloss += ((V[t][idx] - rets[t][idx]) ** 2).sum()
        dslog = np.zeros_like(p)
        if sl_coef > 0 and rc["sl_target"] is not None:
            tgt = rc["sl_target"]
            z = tgt.sum(1, keepdims=True)
            ok = z[:, 0] > 0
            q = np.where(z > 0, tgt / np.maximum(z, 1e-12), 0.0)
            sp = np.exp(rc["slogp"])
            dslog = sl_coef * (sp - q) * ok[:, None] / n_act
            sl_loss += -(q * np.where(q > 0, rc["slogp"], 0)).sum(1)[ok].sum()
            n_sl += ok.sum()
        dt = lambda z: z.astype(agent.dtype)
        dhs = agent.step_backward(rc["cache"], dt(dlogit), dt(dV), dt(dslog), dh[idx], grads)
        dh = np.zeros_like(dh)
        dh[idx] = dhs
    gn = np.sqrt(sum((g ** 2).sum() for g in grads.values()))
    info = dict(entropy=ent_sum / n_act, vloss=vloss / n_act, grad_norm=float(gn),
                sl_loss=sl_loss / max(n_sl, 1))
    return grads, info


def evaluate(agent, n_ep, seed, B=500, greedy=False):
    env = SearchEnv(agent.cfg, seed=seed)
    rng = np.random.default_rng(seed + 1)
    found, ret = [], []
    for _ in range(n_ep // B):
        _, st = rollout(agent, env, B, rng, greedy=greedy, keep=False)
        found.append(st["found"]); ret.append(st["ret"])
    found, ret = np.concatenate(found), np.concatenate(ret)
    return dict(found=float(found.mean()), found_se=float(found.std() / np.sqrt(len(found))),
                ret=float(ret.mean()))


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
    ap.add_argument("--sl", type=float, default=0.0, help="1b head loss weight (0 = off)")
    ap.add_argument("--schema", default="off")
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--eval-episodes", type=int, default=2000)
    ap.add_argument("--log-every", type=int, default=20)
    args = ap.parse_args()

    cfg = EnvConfig()
    agent = PriorityMapAgent(cfg, d=args.d, dh=args.dh, seed=args.seed, schema_mode=args.schema)
    opt = Adam(agent.p, args.lr)
    env = SearchEnv(cfg, seed=10_000 + args.seed)
    rng = np.random.default_rng(20_000 + args.seed)
    ckpt = args.out + ".ckpt.npz"
    it, episodes, best = 0, 0, -1.0
    if os.path.exists(ckpt):
        meta = agent.load(ckpt)
        z = np.load(ckpt)
        for k in agent.p:
            opt.m[k], opt.v[k] = z[f"am_{k}"], z[f"av_{k}"]
        opt.t = int(meta["opt_t"]); it = int(meta["it"]); episodes = int(meta["episodes"])
        best = float(meta["best"])
        env = SearchEnv(cfg, seed=10_000 + args.seed + 7919 * it)
        rng = np.random.default_rng(20_000 + args.seed + 7919 * it)
        print(f"resumed at it {it}, {episodes} episodes", flush=True)

    def save_ckpt():
        np.savez(ckpt, **{f"p_{k}": v for k, v in agent.p.items()},
                 **{f"am_{k}": v for k, v in opt.m.items()}, **{f"av_{k}": v for k, v in opt.v.items()},
                 m_opt_t=opt.t, m_it=it, m_episodes=episodes, m_best=best,
                 m_args=json.dumps(vars(args)))

    log = open(args.out + ".log.jsonl", "a")
    t0, acc = time.time(), []
    while time.time() - t0 < args.minutes * 60 and episodes < args.max_episodes:
        frac = min(1.0, episodes / args.ent_episodes)
        ent = args.ent + (args.ent_final - args.ent) * frac
        recs, st = rollout(agent, env, args.batch, rng, sl_weight=args.sl)
        grads, info = a2c_grads(agent, recs, args.gamma, args.lam, ent, args.vf, args.sl)
        if info["grad_norm"] > args.clip:
            for k in grads:
                grads[k] *= args.clip / info["grad_norm"]
        opt.step(agent.p, grads)
        it += 1
        episodes += args.batch
        acc.append((st["found"].mean(), st["ret"].mean(), info["entropy"], info["vloss"],
                    info["grad_norm"], info["sl_loss"]))
        if it % args.log_every == 0:
            a = np.mean(acc, 0); acc = []
            rec = dict(it=it, episodes=episodes, minutes=round((time.time() - t0) / 60, 2),
                       train_found=round(a[0], 4), train_ret=round(a[1], 4), entropy=round(a[2], 3),
                       vloss=round(a[3], 4), grad_norm=round(a[4], 3), sl_loss=round(a[5], 3),
                       ent_coef=round(ent, 5))
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(rec, flush=True)
        if it % args.eval_every == 0:
            ev = evaluate(agent, args.eval_episodes, seed=500_000 + args.seed)
            rec = dict(it=it, episodes=episodes, eval=ev)
            log.write(json.dumps(rec) + "\n"); log.flush()
            print("EVAL", rec, flush=True)
            if ev["found"] > best:
                best = ev["found"]
                agent.save(args.out + ".best.npz", it=it, episodes=episodes, eval_found=best)
            save_ckpt()
    save_ckpt()
    agent.save(args.out + ".final.npz", it=it, episodes=episodes)
    print("done", it, episodes, flush=True)


if __name__ == "__main__":
    main()
