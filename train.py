"""Train an agent and log learning curves to results/<name>.json.

Sprint 1: --flash-to-ctrl 1 (controller sees the flash), no schema.
Sprint 2: condition A = --schema 16, condition C = --schema 0 --n-h 74
          (parameter-matched); condition B = A evaluated with the schema ablated.
"""
import argparse, json, time, os
import numpy as np
from autodiff import Adam
from env import sample_batch
from model import Agent


def evaluate(agent, seed, n=2000, eval_seed=10_000):
    rng = np.random.default_rng(eval_seed + seed)
    b = sample_batch(n, rng, p_captured=0.5)
    o = agent.forward(b)
    final_ok = o["guess"][:, -1] == b["target_pixel"][:, -1]
    guess_patch = (o["guess"][:, -1] // 8 // 2) * 4 + (o["guess"][:, -1] % 8) // 2
    patch_ok = guess_patch == b["target_patch"][:, -1]
    tp = b["target_patch"]
    att_on_target = np.take_along_axis(o["att"], tp[..., None], -1)[..., 0]  # (B,T)
    cap = b["captured"]
    res = dict(
        acc=float(final_ok.mean()),
        acc_ordinary=float(final_ok[~cap].mean()),
        acc_captured=float(final_ok[cap].mean()),
        patch_acc=float(patch_ok.mean()),
        att_on_target=float(att_on_target[:, 2:].mean()),
        loss=float(o["loss"].v),
    )
    if agent.k:
        # does the schema's prediction point where attention actually peaks?
        res["schema_top1"] = float((o["schema_pred"].argmax(-1) == o["att"].argmax(-1)).mean())
    return res


def schema_mean_state(agent, seed, n=2000):
    b = sample_batch(n, np.random.default_rng(20_000 + seed), p_captured=0.5)
    return agent.forward(b)["schema_state"].mean(axis=(0, 1))


def train(name, seed=0, attention="learned", iters=6000, batch=128, lr=3e-3, n_h=64,
          gain=1.0, flash_to_ctrl=True, schema=0, v2=False, trace_to="none"):
    agent = Agent(seed=seed, n_h=n_h, attention=attention, gain=gain,
                  flash_to_ctrl=flash_to_ctrl, schema_width=schema, v2=v2, trace_to=trace_to)
    opt = Adam(agent.params(), lr=lr)
    rng = np.random.default_rng(seed)
    log = dict(name=name, seed=seed, attention=attention, gain=gain, n_h=n_h,
               flash_to_ctrl=flash_to_ctrl, schema=schema, v2=v2, trace_to=trace_to,
               n_params=agent.n_params(), curve=[])
    t0 = time.time()
    for it in range(1, iters + 1):
        b = sample_batch(batch, rng, p_captured=0.5)
        o = agent.forward(b)
        o["loss"].backprop()
        opt.step()
        if it % 250 == 0 or it == 1:
            ev = evaluate(agent, seed, n=1000)
            ev["iter"] = it
            ev["sec"] = round(time.time() - t0, 1)
            log["curve"].append(ev)
            print(name, ev, flush=True)
    # final tests on held-out episodes (different seed stream from the curve)
    log["final"] = evaluate(agent, seed, n=4000, eval_seed=30_000)
    if schema:
        agent.schema_mean = schema_mean_state(agent, seed)
        agent.ablate = True
        log["final_ablated"] = evaluate(agent, seed, n=4000, eval_seed=30_000)
        agent.ablate = False
        np.save(f"results/{name}_schema_mean.npy", agent.schema_mean)
    os.makedirs("results", exist_ok=True)
    with open(f"results/{name}.json", "w") as f:
        json.dump(log, f, indent=1)
    np.savez(f"results/{name}_params.npz", *[p.v for p in agent.params()])
    return agent, log


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--attention", default="learned")
    ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--gain", type=float, default=1.0)
    ap.add_argument("--n-h", type=int, default=64)
    ap.add_argument("--flash-to-ctrl", type=int, default=1)
    ap.add_argument("--schema", type=int, default=0)
    ap.add_argument("--v2", type=int, default=0)
    ap.add_argument("--trace-to", default="none")
    a = ap.parse_args()
    train(a.name, a.seed, a.attention, a.iters, gain=a.gain, n_h=a.n_h,
          flash_to_ctrl=bool(a.flash_to_ctrl), schema=a.schema, v2=bool(a.v2), trace_to=a.trace_to)
