"""Sprint 5a pilot: schema accuracy split by capture / no capture and window size, 1 000 pilot-bank episodes.
Shows how often the map is wrong, not where it points (blind to H5).  python3 pilot_5a_accsplit.py <run>"""
import sys, numpy as np
sys.path.insert(0, "/home/claude/nexis/v2")
from bank import BankEnv, streams, bank_seeds, _window_mask
from pilot_5a import load, RES
import os
name = sys.argv[1]
ag, a, meta = load(os.path.join(RES, name + ".best.npz"))
mod = ag.module; G = ag.cfg.G
seeds = bank_seeds(1000, 4_000_000)
B = len(seeds); st = [streams(s) for s in seeds]
env = BankEnv(ag.cfg); obs = env.reset_bank([x[0] for x in st]); arng = [x[1] for x in st]
h = ag.init_state(B); hm = mod.init_state(B); union = np.zeros((B, G, G))
prev = np.zeros((B, G, G)); prevcap = np.zeros(B, bool); prevw = np.zeros(B, int)
cnt = {k: [0, 0] for k in ("nocap_w1", "nocap_w2", "cap")}
sharp = []
t = 0
while not env.done.all():
    idx = np.nonzero(~env.done)[0]
    xm, mm = mod.features(env, obs, t)
    eps = np.stack([arng[b].normal(0, 1, G * G) for b in idx])
    hms, slm, bits, _ = mod.forward(xm[idx], mm[idx], hm[idx], eps); hm[idx] = hms
    cur = np.zeros((B, G * G))
    if t > 0:
        cur[idx] = np.exp(slm)
        for b in idx:
            cell = np.unravel_index(int(np.argmax(cur[b])), (G, G))
            k = "cap" if prevcap[b] else f"nocap_w{prevw[b]}"
            cnt[k][0] += 1; cnt[k][1] += bool(prev[b][cell])
            sharp.append(cur[b].max())
    cur = cur.reshape(B, G, G); union = np.maximum(union, cur)
    x, m = ag.features(env, obs, t, np.stack([cur, union], -1))
    hs, logp, _, _, _ = ag.step(x[idx], m[idx], h[idx]); h[idx] = hs
    act = np.zeros(B, int)
    for j, b in enumerate(idx):
        act[b] = int((logp[j] - np.log(-np.log(arng[b].random(logp.shape[1])))).argmax())
    obs, r, done, info = env.step(act)
    new = np.zeros((B, G, G))
    for b in np.nonzero(info["active"])[0]:
        new[b] = _window_mask(G, *info["actual"][b], env.w[b])
    prev = new; prevcap = info["captured"] & info["active"]; prevw = env.w.copy()
    t += 1
for k, (n, ok) in cnt.items():
    print(name, k, "steps", n, "acc", round(ok / max(n, 1), 3))
print(name, "mean max prob", round(float(np.mean(sharp)), 3))
