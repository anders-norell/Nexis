"""Uppmärksamhetsschemat S (sprint 2) [Claudes förslag].

Indata: modellens dolda tillstånd innan svaret skrivs (från `extract.py`):
  seg  (S, Ls, d)  medel per mening i lagren STATE_LAYERS
  last (Ls, d)     sista prompttoken
Utdata: fördelning över meningarna (förenklad bild av A: bara "vilken mening och hur starkt").
Varje mening poängsätts med samma lilla nät utifrån meningens tillstånd och sista tokens tillstånd;
poängen softmaxas över meningarna. Tränas med KL(A || S).
"""
from __future__ import annotations

import torch
from torch import nn


class Schema(nn.Module):
    def __init__(self, n_layers: int, d: int, h: int = 64, dropout: float = 0.1):
        super().__init__()
        self.ps = nn.Linear(n_layers * d, h)
        self.pq = nn.Linear(n_layers * d, h)
        self.mlp = nn.Sequential(nn.GELU(), nn.Dropout(dropout), nn.Linear(3 * h, h), nn.GELU(), nn.Linear(h, 1))
        self.register_buffer("mu", torch.zeros(n_layers * d))
        self.register_buffer("sd", torch.ones(n_layers * d))

    def logits(self, seg, last):
        # seg (B, S, Ls, d), last (B, Ls, d)
        B, S = seg.shape[:2]
        s = (seg.reshape(B, S, -1).float() - self.mu) / self.sd
        q = (last.reshape(B, -1).float() - self.mu) / self.sd
        ps, pq = self.ps(s), self.pq(q)[:, None, :].expand(-1, S, -1)
        return self.mlp(torch.cat([ps, pq, ps * pq], -1)).squeeze(-1)  # (B, S)

    def forward(self, seg, last):
        return torch.softmax(self.logits(seg, last), -1)


def load_records(files):
    recs = []
    for f in sorted(files):
        recs += torch.load(f, weights_only=False)["records"]
    return recs


def stack(recs):
    return (torch.stack([r["seg"] for r in recs]), torch.stack([r["last"] for r in recs]),
            torch.stack([r["A"] for r in recs]).float())


def train_schema(train, val, seed=0, epochs=300, lr=1e-3, wd=1e-2, h=64, dropout=0.1, patience=30, log=None):
    torch.manual_seed(seed)
    seg, last, A = stack(train)
    vseg, vlast, vA = stack(val)
    m = Schema(seg.shape[2], seg.shape[3], h, dropout)
    allf = torch.cat([seg.reshape(-1, seg.shape[2] * seg.shape[3]).float(), last.reshape(len(last), -1).float()])
    m.mu.copy_(allf.mean(0))
    m.sd.copy_(allf.std(0).clamp(min=1e-3))
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    best, best_state, bad = float("inf"), None, 0
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(len(A), generator=g)
        for i in range(0, len(A), 64):
            b = perm[i:i + 64]
            lp = torch.log_softmax(m.logits(seg[b], last[b]), -1)
            loss = (A[b] * (torch.log(A[b].clamp(min=1e-9)) - lp)).sum(-1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            vlp = torch.log_softmax(m.logits(vseg, vlast), -1)
            vloss = float((vA * (torch.log(vA.clamp(min=1e-9)) - vlp)).sum(-1).mean())
        if log and ep % 20 == 0:
            log(f"epok {ep}: val KL {vloss:.4f}")
        if vloss < best - 1e-5:
            best, best_state, bad = vloss, {k: v.clone() for k, v in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    m.load_state_dict(best_state)
    m.eval()
    return m, best
