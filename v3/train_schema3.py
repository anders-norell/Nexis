"""Sprint 3 steg 2: träna schemat S på träningsämnen (train3) och pröva på undanhållna testämnen (test3), en gång.

    python train_schema3.py --data <katalog med train3_*.pt och test3_*.pt>
Kriterium (results/sprint3_prereg.md): argmax S = argmax A på test, parat mot nollmodellen "alltid målmeningen",
> 2 SE. Hyperparametrar väljs bara på valideringsdelen (sista 200 träningstexterna), samma rutnät som sprint 2.
Resultat: results/sprint3_schema.json, vikter models/schema_s3.pt.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from nexis3.schema import load_records, stack, train_schema  # noqa: E402
from train_schema import GRID, evaluate, paired  # noqa: E402


def evaluate3(m, recs):
    r = evaluate(m, recs)
    seg, last, A = stack(recs)
    with torch.no_grad():
        S = m(seg, last)
    a_arg = A.argmax(-1).tolist()
    s_arg = S.argmax(-1).tolist()
    hit_s = [int(x == y) for x, y in zip(s_arg, a_arg)]
    hit_tgt = [int(rec["target"] == y) for rec, y in zip(recs, a_arg)]
    r["target_eq_A_argmax"] = sum(hit_tgt) / len(hit_tgt)
    r["vs_target"] = paired(hit_s, hit_tgt)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--models", default="models")
    a = ap.parse_args()
    d = Path(a.data)
    train_all = load_records(sorted(d.glob("train3_*.pt")))
    test = load_records(sorted(d.glob("test3_*.pt")))
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
    res = {"hyperparams": hp, "val_KL": vl, "val": evaluate3(m, va), "test": evaluate3(m, test),
           "prereg": "results/sprint3_prereg.md (steg 2)"}
    res["step2_pass"] = res["test"]["vs_target"]["z"] > 2
    Path(a.out).mkdir(exist_ok=True)
    Path(a.models).mkdir(exist_ok=True)
    torch.save({"state_dict": m.state_dict(), "hyperparams": hp, "n_layers": m.ps.in_features},
               Path(a.models) / "schema_s3.pt")
    Path(a.out, "sprint3_schema.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res["test"], indent=1, ensure_ascii=False))
    print("Steg 2 klarat:", res["step2_pass"])


if __name__ == "__main__":
    main()
