"""Sprint 3b steg 2–4: komprimeringsschema P, G och Y, låsning och en körning på testtexterna.

    python train_compress.py train --data <dir>   # bara tok_train; val = sex undanhållna träningsämnen (frö 31)
    python train_compress.py lock "<Anders svar>" # skriver results/sprint3b_lock.json (sha256 av modeller och kod)
    python train_compress.py test --data <dir>    # en gång; vägrar utan lås eller om något ändrats
Kriterier: results/sprint3b_prereg.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from nexis3.compress import Batch, Compressor, eps_terms, fit, fit_yhead, load, pool, spearman_dist  # noqa: E402
from nexis3.topics_v3 import TRAIN_TOPICS  # noqa: E402

LAMS = [0.1, 1.0, 10.0]
KS = [4, 2, 8]
MODELS = Path("models")
LOCK = Path("results/sprint3b_lock.json")
CODE = ["train_compress.py", "nexis3/compress.py", "extract_tokens.py"]


def base(t):
    return t.replace("-lure", "")


def split(recs):
    tt = sorted(TRAIN_TOPICS)
    random.Random(31).shuffle(tt)
    vt = set(tt[:6])
    tr = [r for r in recs if not any(base(t) in vt for t in r["topics"])]
    va = [r for r in recs if base(r["topics"][r["target"]]) in vt]
    return tr, va


def paired(a, b):
    d = torch.tensor(a) - torch.tensor(b)
    se = float(d.std() / len(d) ** 0.5)
    return {"diff": float(d.mean()), "se": se, "z": float(d.mean()) / se if se > 0 else 0.0, "n": len(d)}


@torch.no_grad()
def per_text_eps(m, b):
    S = m.S(b.X)
    qz = m.qz(b.q)[b.text[b.cur]]
    nG, den = eps_terms(S, b, m.g(S[b.cur], qz))
    nI, _ = eps_terms(S, b)
    tx = b.text[b.cur]
    eg, ei = [], []
    for i in range(b.n):
        mk = tx == i
        if mk.any():
            eg.append(float(nG[mk].mean() / den)); ei.append(float(nI[mk].mean() / den))
    return eg, ei, float(nG.mean() / den), float(nI.mean() / den)


def yfeat(m, b):
    with torch.no_grad():
        return pool(m.S(b.X), b), pool(b.H, b)


def ylosses(head, Z, y):
    with torch.no_grad():
        return F.binary_cross_entropy_with_logits(head(Z), y, reduction="none").tolist()


def evaluate(m, b_tr, b_va, b_ev):
    """K1/K2 på b_ev; Y-huvuden tränas på b_tr med tidigt stopp på b_va (samma för S och karikatyren)."""
    eg, ei, EG, EI = per_text_eps(m, b_ev)
    Zs_tr, Zh_tr = yfeat(m, b_tr)
    Zs_va, Zh_va = yfeat(m, b_va)
    Zs, Zh = yfeat(m, b_ev)
    hs, _ = fit_yhead(Zs_tr, b_tr.y, Zs_va, b_va.y)
    hh, _ = fit_yhead(Zh_tr, b_tr.y, Zh_va, b_va.y)
    ls, lh = ylosses(hs, Zs, b_ev.y), ylosses(hh, Zh, b_ev.y)
    p = b_ev.y.mean().clamp(1e-3, 1 - 1e-3)
    null = float(-(b_ev.y * p.log() + (1 - b_ev.y) * (1 - p).log()).mean())
    return {"eps_G": EG, "eps_identity": EI, "K1": paired(ei, eg),
            "Ylogloss_S": sum(ls) / len(ls), "Ylogloss_caricature": sum(lh) / len(lh), "Ylogloss_constant": null,
            "K2": paired(lh, ls), "n_texts": b_ev.n, "n_transitions": len(b_ev.cur), "p_correct": float(b_ev.y.mean())}


@torch.no_grad()
def k3(m, b, B=200, seed=0, k=None):
    g = torch.Generator().manual_seed(seed)
    S = m.S(b.X)
    Xz = (b.X - m.mu) / m.sd
    R = torch.randn(Xz.shape[1], S.shape[1], generator=torch.Generator().manual_seed(12345))
    Rp = Xz @ R
    diffs = []
    for _ in range(B):
        ts = torch.randint(0, b.n, (b.n,), generator=g)
        idx = torch.cat([(b.text == t).nonzero().squeeze(-1) for t in ts.tolist()])
        sub = idx[torch.randperm(len(idx), generator=g)[:400]]
        diffs.append(spearman_dist(Xz[sub], S[sub]) - spearman_dist(Xz[sub], Rp[sub]))
    d = torch.tensor(diffs)
    full = torch.arange(len(b.text))
    return {"rho_S": spearman_dist(Xz, S, g, 800), "rho_random": spearman_dist(Xz, Rp, g, 800),
            "diff_boot_mean": float(d.mean()), "diff_boot_sd": float(d.std()), "B": B}


def describe(m, b):
    """Vad S:s tal betyder: korrelation med meningsandelarna (karikatyren) och med andel på lockmeningen."""
    with torch.no_grad():
        S = m.S(b.X)
    lure = torch.cat([b.H[b.text == i][:, b.recs[i]["lure"]] for i in range(b.n)])
    tgt = torch.cat([b.H[b.text == i][:, b.recs[i]["target"]] for i in range(b.n)])
    C = lambda u, v: float(torch.corrcoef(torch.stack([u, v]))[0, 1])
    return {"corr_S_with_target_share": [C(S[:, j], tgt) for j in range(S.shape[1])],
            "corr_S_with_lure_share": [C(S[:, j], lure) for j in range(S.shape[1])],
            "corr_S_with_sentence_shares": [[C(S[:, j], b.H[:, s]) for s in range(4)] for j in range(S.shape[1])]}


def ylength_controls(m, b_tr, b_va, b_ev):
    """Beskrivande: Y ur bara första token (före svaret) och med svarslängd som extra indata till båda."""
    Ln = lambda b: torch.tensor([[float(min(r["n_answer_tokens"], 12)), float(r["n_answer_tokens"] == 2)] for r in b.recs])
    out = {}
    feats = {"S": lambda b: pool(m.S(b.X), b).detach(), "caricature": lambda b: pool(b.H, b),
             "S_first_token": lambda b: torch.cat([m.S(b.X)[b.pos == 0]] * 2, 1).detach(),
             "caricature_first_token": lambda b: torch.cat([b.H[b.pos == 0]] * 2, 1)}
    for name, f in feats.items():
        for extra in (False, True):
            Z = lambda b: torch.cat([f(b), Ln(b)], 1) if extra else f(b)
            h, _ = fit_yhead(Z(b_tr), b_tr.y, Z(b_va), b_va.y)
            out[name + ("+length" if extra else "")] = sum(ylosses(h, Z(b_ev), b_ev.y)) / b_ev.n
    h, _ = fit_yhead(Ln(b_tr), b_tr.y, Ln(b_va), b_va.y)
    out["length_only"] = sum(ylosses(h, Ln(b_ev), b_ev.y)) / b_ev.n
    return out


def cmd_train(a):
    recs = load(sorted(Path(a.data).glob("tok_train_*.pt")))
    tr, va = split(recs)
    b_tr, b_va = Batch(tr), Batch(va)
    print(f"träning {b_tr.n} texter / {len(b_tr.text)} token, validering {b_va.n} / {len(b_va.text)}")
    d_in, report = b_tr.X.shape[1], {"grid": {}}
    for lam in LAMS:
        m = fit(Compressor(d_in, 4), b_tr, b_va, lam)
        ev = evaluate(m, b_tr, b_va, b_va)
        report["grid"][str(lam)] = ev
        print(f"λ={lam}: val ε_G {ev['eps_G']:.3f} ε_id {ev['eps_identity']:.3f} | Y S {ev['Ylogloss_S']:.3f} "
              f"karikatyr {ev['Ylogloss_caricature']:.3f} konstant {ev['Ylogloss_constant']:.3f}", flush=True)
    ok = [l for l in LAMS if report["grid"][str(l)]["eps_G"] < report["grid"][str(l)]["eps_identity"]]
    lam = (min(ok, key=lambda l: report["grid"][str(l)]["Ylogloss_S"]) if ok
           else min(LAMS, key=lambda l: report["grid"][str(l)]["eps_G"]))
    report["lambda"] = lam
    MODELS.mkdir(exist_ok=True)
    for k in KS:
        m = fit(Compressor(d_in, k), b_tr, b_va, lam)
        torch.save({"state_dict": m.state_dict(), "k": k, "d_in": d_in, "lambda": lam}, MODELS / f"compress_k{k}.pt")
        report[f"val_k{k}"] = evaluate(m, b_tr, b_va, b_va)
        report[f"val_k{k}"]["K3"] = k3(m, b_va, B=50)
        report[f"val_k{k}"]["describe"] = describe(m, b_va)
    Path("results/sprint3b_train.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    v = report["val_k4"]
    print(f"vald λ={lam}. k=4 val: K1 z {v['K1']['z']:.1f}, K2 z {v['K2']['z']:.1f}, K3 {v['K3']['diff_boot_mean']:.3f}"
          f" ± {v['K3']['diff_boot_sd']:.3f}")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def cmd_lock(a):
    assert not LOCK.exists(), "redan låst"
    files = [str(MODELS / f"compress_k{k}.pt") for k in KS] + CODE + ["results/sprint3b_prereg.md"]
    LOCK.write_text(json.dumps({"answer": a.answer, "sha256": {f: sha(f) for f in files}}, indent=1, ensure_ascii=False),
                    encoding="utf-8")
    print("låst:", LOCK)


def load_model(k):
    ck = torch.load(MODELS / f"compress_k{k}.pt", weights_only=False)
    m = Compressor(ck["d_in"], ck["k"])
    m.load_state_dict(ck["state_dict"]); m.eval()
    return m


def cmd_test(a):
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    for f, h in lock["sha256"].items():
        assert sha(f) == h, f"{f} ändrad efter låsning"
    out = Path("results/sprint3b_test.json")
    assert not out.exists(), "testet är redan kört"
    d = Path(a.data)
    tr, va = split(load(sorted(d.glob("tok_train_*.pt"))))
    b_tr, b_va = Batch(tr), Batch(va)
    b_te = Batch(load(sorted(d.glob("tok_test_*.pt"))))
    res = {}
    for k in KS:
        m = load_model(k)
        ev = evaluate(m, b_tr, b_va, b_te)
        ev["K3"] = k3(m, b_te)
        ev["describe"] = describe(m, b_te)
        ev["Y_controls"] = ylength_controls(m, b_tr, b_va, b_te)
        res[f"k{k}"] = ev
    r4 = res["k4"]
    res["K1_pass"] = r4["K1"]["z"] > 2
    res["K2_pass"] = r4["K2"]["z"] > 2
    res["K3_pass"] = r4["K3"]["diff_boot_mean"] > 2 * r4["K3"]["diff_boot_sd"]
    res["gate3_pass"] = res["K1_pass"] and res["K2_pass"] and res["K3_pass"]
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ["K1_pass", "K2_pass", "K3_pass", "gate3_pass"]}))
    print("k4:", json.dumps({k: r4[k] for k in ["eps_G", "eps_identity", "K1", "Ylogloss_S", "Ylogloss_caricature",
                                                "Ylogloss_constant", "K2", "p_correct"]}, indent=1))
    print("K3:", r4["K3"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    for c in ["train", "test"]:
        s = sp.add_parser(c); s.add_argument("--data", required=True)
    s = sp.add_parser("lock"); s.add_argument("answer")
    a = ap.parse_args()
    {"train": cmd_train, "lock": cmd_lock, "test": cmd_test}[a.cmd](a)
