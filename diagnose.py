"""Sprint 1 checks on a trained agent: is attention used, and does a flash capture it?"""
import json, sys
import numpy as np
from env import sample_batch
from model import Agent


def load(name):
    log = json.load(open(f"results/{name}.json"))
    ag = Agent(seed=log["seed"], attention=log["attention"], gain=log.get("gain", 1.0),
               n_h=log.get("n_h", 64), flash_to_ctrl=log.get("flash_to_ctrl", True),
               schema_width=log.get("schema", 0), v2=log.get("v2", False),
               trace_to=log.get("trace_to", "none"))
    if ag.k:
        ag.schema_mean = np.load(f"results/{name}_schema_mean.npy")
    d = np.load(f"results/{name}_params.npz")
    keys = sorted(d.files, key=lambda s: int(s.split("_")[1]))
    for p, k in zip(ag.params(), keys):
        p.v[...] = d[k]
    return ag


def capture_profile(ag, n=4000, seed=99):
    b = sample_batch(n, np.random.default_rng(seed), p_captured=1.0)
    a = ag.forward(b)["att"]
    fs, fp, tp, i = b["flash_start"], b["flash_patch"], b["target_patch"], np.arange(n)
    rel = range(-1, 3)  # steps relative to flash onset (flash ends <= step 10)
    on_flash = [float(a[i, fs + k, fp].mean()) for k in rel]
    on_target = [float(a[i, fs + k, tp[i, fs + k]].mean()) for k in rel]
    return dict(rel_steps=list(rel), on_flash=on_flash, on_target=on_target)


if __name__ == "__main__":
    for name in sys.argv[1:]:
        prof = capture_profile(load(name))
        print(name)
        print("  step rel. to flash :", prof["rel_steps"])
        print("  attention on flash :", np.round(prof["on_flash"], 2))
        print("  attention on target:", np.round(prof["on_target"], 2))
