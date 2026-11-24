"""Relationsschema (sprint 3 steg 2b) [Claudes förslag].

Symmetrikrav: schemat ska vara okänsligt för vilket ämne texten handlar om. Därför ser det inte tillståndsvektorerna
själva, bara skalära relationer mellan dem (per lager i STATE_LAYERS):
  cq   cosinus mellan meningens tillstånd och sista prompttokens tillstånd (frågans läge)
  bq   inlärd lågrangslikhet (W·seg)·(W·last)/sqrt(k), samma W för alla meningar och texter (k=0 stänger av)
  co   största och medel-cosinus mot de andra meningarna i texten
  cq, bq också centrerade inom texten (relativ likhet)
  meningens position (fyra värden)
Ett litet nät poängsätter varje mening ur dessa tal; softmax över meningarna. Tränas med KL(A || S).
"""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


def _cos(a, b):
    return F.cosine_similarity(a, b, dim=-1)


class RelSchema(nn.Module):
    def __init__(self, n_layers: int, d: int, k: int = 16, h: int = 32, dropout: float = 0.1):
        super().__init__()
        self.k = k
        self.W = nn.Parameter(torch.randn(n_layers, d, k) / math.sqrt(d)) if k > 0 else None
        nf = n_layers * (2 + 2) + 4 + (n_layers * 2 if k > 0 else 0)
        self.mlp = nn.Sequential(nn.Linear(nf, h), nn.GELU(), nn.Dropout(dropout), nn.Linear(h, h), nn.GELU(),
                                 nn.Linear(h, 1))
        self.register_buffer("mu", torch.zeros(n_layers, d))
        self.register_buffer("sd", torch.ones(n_layers, d))

    def features(self, seg, last):
        # seg (B, S, L, d), last (B, L, d)
        B, S, L, _ = seg.shape
        s = (seg.float() - self.mu) / self.sd
        q = ((last.float() - self.mu) / self.sd)[:, None].expand(-1, S, -1, -1)
        cq = _cos(s, q)                                             # (B, S, L)
        so = _cos(s[:, :, None], s[:, None, :])                    # (B, S, S, L)
        eye = torch.eye(S, dtype=torch.bool)[None, :, :, None]
        mx = so.masked_fill(eye, -2).amax(2)
        mn = so.masked_fill(eye, 0).sum(2) / (S - 1)
        f = [cq, cq - cq.mean(1, keepdim=True), mx, mn]
        if self.k > 0:
            ws = torch.einsum("bsld,ldk->bslk", s, self.W)
            wq = torch.einsum("bsld,ldk->bslk", q, self.W)
            bq = (ws * wq).sum(-1) / math.sqrt(self.k)
            f += [bq, bq - bq.mean(1, keepdim=True)]
        pos = torch.eye(4)[None].expand(B, -1, -1)[:, :S]
        return torch.cat(f + [pos], -1)

    def logits(self, seg, last):
        return self.mlp(self.features(seg, last)).squeeze(-1)

    def forward(self, seg, last):
        return torch.softmax(self.logits(seg, last), -1)


def train_rel(train, val, seed=0, epochs=400, lr=2e-3, wd=1e-2, k=16, h=32, dropout=0.1, patience=40):
    from .schema import stack
    torch.manual_seed(seed)
    seg, last, A = stack(train)
    vseg, vlast, vA = stack(val)
    m = RelSchema(seg.shape[2], seg.shape[3], k, h, dropout)
    allf = torch.cat([seg.reshape(-1, seg.shape[2], seg.shape[3]).float(), last.float()])
    m.mu.copy_(allf.mean(0))
    m.sd.copy_(allf.std(0).clamp(min=1e-3))
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    best, best_state, bad = float("inf"), None, 0
    g = torch.Generator().manual_seed(seed)
    kl = lambda a, lp: (a * (torch.log(a.clamp(min=1e-9)) - lp)).sum(-1).mean()
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(len(A), generator=g)
        for i in range(0, len(A), 64):
            b = perm[i:i + 64]
            loss = kl(A[b], torch.log_softmax(m.logits(seg[b], last[b]), -1))
            opt.zero_grad()
            loss.backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            vloss = float(kl(vA, torch.log_softmax(m.logits(vseg, vlast), -1)))
        if vloss < best - 1e-5:
            best, best_state, bad = vloss, {kk: v.clone() for kk, v in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    m.load_state_dict(best_state)
    m.eval()
    return m, best
