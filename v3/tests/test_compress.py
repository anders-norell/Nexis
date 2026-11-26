"""Komprimeringsschemat: P ser bara attentionen; ε är skalinvariant; konstant P ger ingen Y-nytta; G = S är specialfall."""
import torch

from nexis3.compress import Batch, Compressor, caricature, eps_terms, fit_yhead, pool


def fake(n=40, T=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    recs = []
    for i in range(n):
        a = torch.rand(T, 28, 12, 7, generator=g)
        a = a / a.sum(-1, keepdim=True)
        recs.append({"att": a.half(), "q": torch.randn(1536, generator=g).half(), "task_correct": bool(i % 3),
                     "target": i % 4, "lure": (i + 1) % 4, "topics": ["x"] * 4})
    return recs


def test_batch_transitions_stay_within_text():
    b = Batch(fake(5, 3))
    assert len(b.cur) == 5 * 2 and all(b.text[t] == b.text[t + 1] for t in b.cur.tolist())


def test_caricature_sums_to_one():
    a = fake(1)[0]["att"]
    assert torch.allclose(caricature(a).sum(-1), torch.ones(4))


def test_eps_scale_invariant():
    b = Batch(fake(10))
    S = torch.randn(len(b.text), 4)
    n1, d1 = eps_terms(S, b)
    n2, d2 = eps_terms(10 * S, b)
    assert torch.allclose(n1.mean() / d1, n2.mean() / d2, rtol=1e-4)


def test_P_reads_only_attention():
    m = Compressor(28 * 12 * 7, 4)
    import inspect
    assert list(inspect.signature(m.S).parameters) == ["X"]
    b = Batch(fake(3))
    b2 = Batch(fake(3)); b2.q = torch.randn_like(b2.q)   # annan fråga, samma attention
    b2.X = b.X
    assert torch.allclose(m.S(b.X), m.S(b2.X))


def test_constant_S_gives_no_y_gain():
    b = Batch(fake(60))
    Z = pool(torch.zeros(len(b.text), 4), b)
    h, vl = fit_yhead(Z, b.y, Z, b.y)
    p = b.y.mean()
    null = float(-(b.y * p.log() + (1 - b.y) * (1 - p).log()).mean())
    assert vl >= null - 1e-3
