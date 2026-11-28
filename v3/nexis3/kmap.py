"""Sprint 4: kartritaren – ett schema som ser attentionen glest och bär sin uppskattning däremellan. [Claudes förslag]

Förregistrering: results/sprint4_prereg.md, del 2a. Ser bara: observationerna o_t (handvald andel per mening, lager 9–17,
var M:e svarstoken), frågans läge q och tokenposition. Aldrig texten eller svarsorden.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .compress import caricature

M = 4            # observation var fjärde svarstoken
EPS = 1e-4


def seq(r):
    """(H (T,4), q (d,)) för en v4-post."""
    return caricature(r["att"]).clamp(min=EPS), r["q"].float()


class KMap(nn.Module):
    def __init__(self, d_q=1536, qk=8, h=32):
        super().__init__()
        self.Q = nn.Linear(d_q, qk)
        self.G = nn.Sequential(nn.Linear(4 + qk + 2, h), nn.GELU(), nn.Linear(h, 4))
        self.K = nn.Sequential(nn.Linear(8, h), nn.GELU(), nn.Linear(h, 1))
        self.register_buffer("qmu", torch.zeros(d_q))
        self.register_buffer("qsd", torch.ones(d_q))

    def run(self, H, q, observed=None):
        """Logfördelningar S (T,4). observed: bool-mask (T,); standard t % M == 0."""
        T = H.shape[0]
        if observed is None:
            observed = torch.arange(T) % M == 0
        qz = self.Q((q - self.qmu) / self.qsd)
        out, s, since = [], None, 0.0
        for t in range(T):
            if s is None:
                pred = torch.log(torch.full((4,), 0.25))
            else:
                x = torch.cat([s, qz, torch.tensor([t / 20.0, since / M])])
                pred = torch.log_softmax(s + self.G(x), -1)
            if observed[t]:
                lo = torch.log(H[t])
                k = torch.sigmoid(self.K(torch.cat([pred, lo])))
                s = torch.log_softmax(pred + k * (lo - pred), -1)
                since = 0.0
            else:
                s = pred
                since += 1.0
            out.append(s)
        return torch.stack(out)


def baselines(H, observed, pos_mean, const):
    """Tre jämförelser per token: senaste observation, drift per position, konstant."""
    T = H.shape[0]
    last, hold = None, []
    for t in range(T):
        if observed[t]:
            last = H[t]
        hold.append(last if last is not None else const)
    P = torch.stack([pos_mean[min(t, len(pos_mean) - 1)] for t in range(T)])
    return {"hold": torch.stack(hold), "drift": P, "constant": const.expand(T, 4)}


def fit(model, train, val, epochs=40, lr=3e-3, seed=0, patience=6):
    torch.manual_seed(seed)
    qs = torch.stack([q for _, q in train])
    model.qmu.copy_(qs.mean(0)); model.qsd.copy_(qs.std(0).clamp(min=1e-4))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, state, bad = float("inf"), None, 0
    g = torch.Generator().manual_seed(seed)

    def loss_of(data):
        tot = 0.0
        for H, q in data:
            S = model.run(H, q)
            tot = tot + (H * (torch.log(H) - S)).sum(-1).mean()
        return tot / len(data)

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(train), generator=g).tolist()
        for i in range(0, len(perm), 32):
            l = loss_of([train[j] for j in perm[i:i + 32]])
            opt.zero_grad(); l.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(loss_of(val))
        print(f"epok {ep}: val KL {vl:.4f}", flush=True)
        if vl < best - 1e-5:
            best, state, bad = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(state); model.eval()
    return model, best
