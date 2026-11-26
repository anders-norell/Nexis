"""Sprint 3b: schemat som komprimering av attention (projektet claude/nexis_v3_sprint3b_plan.md) [Claudes förslag].

P: A_t (lager × huvud × textdel, per svarstoken) → S_t (k tal). P ser bara attentionen, aldrig texten eller tillstånden.
G: (S_t, q) → förutsägelse av S_{t+1}; q = frågans läge (lager 13), nedprojicerat till 8 tal.
Y-huvud: [medel_t S_t, S_0] → om svaret blir rätt.
Kommuteringsfel ε normeras med variansen hos S (annars kan P bara krympa skalan): ε = E‖S_{t+1} − Ĝ‖² / E‖S − S̄‖².
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

LAYERS_A = list(range(9, 18))        # lagren i den låsta A-definitionen (handvald karikatyr)


def load(paths):
    recs = []
    for p in paths:
        recs += torch.load(p, weights_only=False)["records"]
    return recs


def caricature(att):
    """Handvald karikatyr per token: andel attention per mening i lager 9–17 (4 tal, summa 1)."""
    s = att[:, LAYERS_A, :, :4].float().mean(dim=(1, 2))
    return s / s.sum(-1, keepdim=True)


class Batch:
    """Tokenserier i en platt tensor med textindex."""
    def __init__(self, recs):
        self.recs = recs
        self.X = torch.cat([r["att"].float().reshape(r["att"].shape[0], -1) for r in recs])
        self.H = torch.cat([caricature(r["att"]) for r in recs])
        self.text = torch.cat([torch.full((r["att"].shape[0],), i) for i, r in enumerate(recs)])
        self.pos = torch.cat([torch.arange(r["att"].shape[0]) for r in recs])
        self.q = torch.stack([r["q"].float() for r in recs])
        self.y = torch.tensor([float(r["task_correct"]) for r in recs])
        nxt = torch.arange(len(self.text)) + 1
        ok = nxt < len(self.text)
        ok[ok.clone()] &= self.text[nxt[ok]] == self.text[ok.nonzero().squeeze(-1)]
        self.cur = ok.nonzero().squeeze(-1)         # tokenindex t med en efterföljare t+1 i samma text
        self.n = len(recs)


def pool(S, b: Batch):
    """[medel över svarets token, första token] per text."""
    n = b.n
    sums = torch.zeros(n, S.shape[1]).index_add_(0, b.text, S)
    cnt = torch.zeros(n).index_add_(0, b.text, torch.ones(len(b.text))).clamp(min=1)
    first = S[b.pos == 0]
    return torch.cat([sums / cnt[:, None], first], -1)


class Compressor(nn.Module):
    def __init__(self, d_in, k=4, h=64, d_q=1536, qk=8):
        super().__init__()
        self.P = nn.Sequential(nn.Linear(d_in, h), nn.GELU(), nn.Linear(h, k))
        self.Q = nn.Linear(d_q, qk)
        self.G = nn.Sequential(nn.Linear(k + qk, h), nn.GELU(), nn.Linear(h, k))
        self.Y = nn.Sequential(nn.Linear(2 * k, 16), nn.GELU(), nn.Linear(16, 1))
        self.register_buffer("mu", torch.zeros(d_in))
        self.register_buffer("sd", torch.ones(d_in))
        self.register_buffer("qmu", torch.zeros(d_q))
        self.register_buffer("qsd", torch.ones(d_q))

    def S(self, X):
        return self.P((X - self.mu) / self.sd)

    def g(self, S, qz):
        return S + self.G(torch.cat([S, qz], -1))           # G förutsäger förändringen; G = S är specialfallet 0

    def qz(self, q):
        return self.Q((q - self.qmu) / self.qsd)


def eps_terms(S, b: Batch, Shat=None):
    """Per övergång: ‖S_{t+1} − Ŝ‖² och nämnaren (variansen hos S). Ŝ = S_t om inget annat ges (G = identitet)."""
    nxt = S[b.cur + 1]
    pred = S[b.cur] if Shat is None else Shat
    num = ((nxt - pred) ** 2).sum(-1)
    den = ((S - S.mean(0)) ** 2).sum(-1).mean()
    return num, den


def losses(m: Compressor, b: Batch, lam):
    S = m.S(b.X)
    qz = m.qz(b.q)[b.text[b.cur]]
    num, den = eps_terms(S, b, m.g(S[b.cur], qz))
    eps = num.mean() / den.clamp(min=1e-6)
    yl = F.binary_cross_entropy_with_logits(m.Y(pool(S, b)).squeeze(-1), b.y)
    return eps + lam * yl, eps, yl


def fit(m, b_tr, b_va, lam, epochs=600, lr=3e-3, wd=1e-3, seed=0, patience=60):
    torch.manual_seed(seed)
    m.mu.copy_(b_tr.X.mean(0)); m.sd.copy_(b_tr.X.std(0).clamp(min=1e-4))
    m.qmu.copy_(b_tr.q.mean(0)); m.qsd.copy_(b_tr.q.std(0).clamp(min=1e-4))
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    best, state, bad = float("inf"), None, 0
    for ep in range(epochs):
        m.train()
        loss, _, _ = losses(m, b_tr, lam)
        opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            vl = float(losses(m, b_va, lam)[0])
        if vl < best - 1e-5:
            best, state, bad = vl, {k: v.clone() for k, v in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    m.load_state_dict(state); m.eval()
    return m


class YHead(nn.Module):
    """Samma lilla Y-huvud för S och för den handvalda karikatyren (tränas på fryst indata)."""
    def __init__(self, k):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * k, 16), nn.GELU(), nn.Linear(16, 1))
        self.register_buffer("mu", torch.zeros(2 * k))
        self.register_buffer("sd", torch.ones(2 * k))

    def forward(self, Z):
        return self.net((Z - self.mu) / self.sd).squeeze(-1)


def fit_yhead(Z_tr, y_tr, Z_va, y_va, seed=0, epochs=2000, lr=3e-3, wd=1e-2, patience=100):
    torch.manual_seed(seed)
    h = YHead(Z_tr.shape[1] // 2)
    h.mu.copy_(Z_tr.mean(0)); h.sd.copy_(Z_tr.std(0).clamp(min=1e-4))
    opt = torch.optim.AdamW(h.parameters(), lr=lr, weight_decay=wd)
    best, state, bad = float("inf"), None, 0
    for _ in range(epochs):
        h.train()
        l = F.binary_cross_entropy_with_logits(h(Z_tr), y_tr)
        opt.zero_grad(); l.backward(); opt.step()
        h.eval()
        with torch.no_grad():
            vl = float(F.binary_cross_entropy_with_logits(h(Z_va), y_va))
        if vl < best - 1e-6:
            best, state, bad = vl, {k: v.clone() for k, v in h.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    h.load_state_dict(state); h.eval()
    return h, best


def spearman_dist(Xa, Xb, gen=None, max_n=600):
    """Spearman mellan parvisa euklidiska avstånd i två representationer av samma token."""
    n = len(Xa)
    if n > max_n:
        idx = torch.randperm(n, generator=gen)[:max_n]
        Xa, Xb = Xa[idx], Xb[idx]
    iu = torch.triu_indices(len(Xa), len(Xa), 1)
    a = torch.cdist(Xa, Xa)[iu[0], iu[1]]
    b = torch.cdist(Xb, Xb)[iu[0], iu[1]]
    rk = lambda x: x.argsort().argsort().float()
    return float(torch.corrcoef(torch.stack([rk(a), rk(b)]))[0, 1])
