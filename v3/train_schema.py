"""Sprint 2: träna schemat S på träningsdata och bedöm G2a på testdata (en gång).

    python train_schema.py --data <katalog med train_*.pt och test_*.pt>
Hyperparametrar väljs bara på en valideringsdel av träningsdatan (sista 200 texterna).
Resultat: results/sprint2_schema.json, vikter results/schema_s.pt.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from nexis3.schema import load_records, stack, train_schema  # noqa: E402

GRID = [dict(h=32, wd=1e-2, dropout=0.1), dict(h=64, wd=1e-2, dropout=0.1), dict(h=64, wd=1e-1, dropout=0.3),
        dict(h=128, wd=1e-1, dropout=0.3)]


def paired(a, b):
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    m = sum(d) / n
    se = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1) / n)
    return {"diff": m, "se": se, "z": m / se if se > 0 else float("inf"), "n": n}


def evaluate(m, recs):
    seg, last, A = stack(recs)
    with torch.no_grad():
        S = m(seg, last)
    a_arg = A.argmax(-1).tolist()
    s_arg = S.argmax(-1).tolist()
    hit_s = [int(x == y) for x, y in zip(s_arg, a_arg)]
    hit_rec = [int(y == 3) for y in a_arg]
    hit_ovl = [int(r["overlap_choice"] == y) for r, y in zip(recs, a_arg)]
    kl = float((A * (torch.log(A.clamp(min=1e-9)) - torch.log(S.clamp(min=1e-9)))).sum(-1).mean())
    corr = float(torch.corrcoef(torch.stack([S.flatten(), A.flatten()]))[0, 1])
    return {
        "n": len(recs), "S_argmax_eq_A_argmax": sum(hit_s) / len(hit_s),
        "recency_eq_A_argmax": sum(hit_rec) / len(hit_rec), "overlap_eq_A_argmax": sum(hit_ovl) / len(hit_ovl),
        "vs_recency": paired(hit_s, hit_rec), "vs_overlap": paired(hit_s, hit_ovl),
        "KL_A_S": kl, "corr_S_A": corr,
        "S_argmax_eq_target": sum(int(s == r["target"]) for s, r in zip(s_arg, recs)) / len(recs),
        "A_argmax_eq_target": sum(int(a == r["target"]) for a, r in zip(a_arg, recs)) / len(recs),
        "S_argmax_eq_A_argmax_when_A_not_target": (
            lambda xs: (sum(xs) / len(xs), len(xs)) if xs else (None, 0))(
            [h for h, a, r in zip(hit_s, a_arg, recs) if a != r["target"]]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    d = Path(a.data)
    train_all = load_records(d.glob("train_*.pt"))
    test = load_records(d.glob("test_*.pt"))
    tr, va = train_all[:-200], train_all[-200:]
    print(f"träning {len(tr)}, validering {len(va)}, test {len(test)}")
    best = None
    for hp in GRID:
        m, vl = train_schema(tr, va, **hp)
        print(f"{hp}: val KL {vl:.4f}, val argmax-träff {evaluate(m, va)['S_argmax_eq_A_argmax']:.3f}")
        if best is None or vl < best[1]:
            best = (hp, vl)
    hp = best[0]
    m, vl = train_schema(tr, va, **hp)
    res = {"hyperparams": hp, "val_KL": vl, "val": evaluate(m, va), "test": evaluate(m, test),
           "prereg": "results/sprint2_prereg.md (G2a)"}
    t = res["test"]
    res["G2a_pass"] = t["vs_recency"]["z"] > 2 and t["vs_overlap"]["z"] > 2
    Path(a.out).mkdir(exist_ok=True)
    torch.save({"state_dict": m.state_dict(), "hyperparams": hp, "n_layers": m.ps.in_features}, Path(a.out) / "schema_s.pt")
    Path(a.out, "sprint2_schema.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res["test"], indent=1, ensure_ascii=False))
    print("G2a klarat:", res["G2a_pass"])


if __name__ == "__main__":
    main()
