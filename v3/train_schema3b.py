"""Sprint 3 steg 2b/2c: relationsschema (ämnesinvariant) [Claudes förslag]. Förregistrering: results/sprint3_prereg.md,
tillägg 2b/2c.

    python train_schema3b.py train --data <dir>      # träna på train3, välj hyperparametrar på ämnesvalidering;
                                                     # beskrivande körning på test3 (redan sedd) + karikatyrmått
    python train_schema3b.py test  --data <dir>      # en gång: låsta kriterier på test3b (nya texter, symmetripar)
Validering: sex träningsämnen hålls undan (frö 31). Träning = texter utan något av dem, validering = texter vars
målämne är ett av dem. Hyperparametrar väljs bara på valideringens KL.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from nexis3.schema import load_records, stack  # noqa: E402
from nexis3.schema_rel import RelSchema, train_rel  # noqa: E402
from nexis3.topics_v3 import TRAIN_TOPICS  # noqa: E402
from train_schema import paired  # noqa: E402
from train_schema3 import evaluate3  # noqa: E402

GRID = [dict(k=0, h=32, dropout=0.1), dict(k=8, h=32, dropout=0.1), dict(k=16, h=32, dropout=0.1),
        dict(k=32, h=64, dropout=0.2)]
MODEL = Path("models/schema_s3b.pt")


def base(t):
    return t.replace("-lure", "")


def split_train(recs):
    tt = sorted(TRAIN_TOPICS)
    random.Random(31).shuffle(tt)
    vt = set(tt[:6])
    tr = [r for r in recs if not any(base(t) in vt for t in r["topics"])]
    va = [r for r in recs if base(r["topics"][r["target"]]) in vt]
    return tr, va, sorted(vt)


def caricature(S, A):
    """Karikatyrmått (beskrivande): Spearman mellan parvisa avstånd (L1) i A och i S, samt andel av de 10 % största
    A-avstånden som också hör till de 10 % största S-avstånden."""
    dA = torch.cdist(A, A, p=1)
    dS = torch.cdist(S, S, p=1)
    iu = torch.triu_indices(len(A), len(A), 1)
    a, s = dA[iu[0], iu[1]], dS[iu[0], iu[1]]
    rk = lambda x: x.argsort().argsort().float()
    rho = float(torch.corrcoef(torch.stack([rk(a), rk(s)]))[0, 1])
    qa, qs = a.quantile(0.9), s.quantile(0.9)
    top = float(((a >= qa) & (s >= qs)).sum() / (a >= qa).sum())
    return {"spearman_dist": rho, "top10_overlap": top}


def predict(m, recs):
    seg, last, A = stack(recs)
    with torch.no_grad():
        return m(seg, last), A


def cmd_train(a):
    d = Path(a.data)
    recs = load_records(sorted(d.glob("train3_*.pt")))
    tr, va, vt = split_train(recs)
    print(f"träning {len(tr)}, validering {len(va)} (ämnen {vt})")
    best = None
    for hp in GRID:
        m, vl = train_rel(tr, va, **hp)
        print(f"{hp}: val KL {vl:.4f}, val argmax-träff {evaluate3(m, va)['S_argmax_eq_A_argmax']:.3f}", flush=True)
        if best is None or vl < best[1]:
            best = (hp, vl)
    hp = best[0]
    m, vl = train_rel(tr, va, **hp)
    MODEL.parent.mkdir(exist_ok=True)
    torch.save({"state_dict": m.state_dict(), "hyperparams": hp, "n_layers": 3, "d": 1536}, MODEL)
    test3 = load_records(sorted(d.glob("test3_*.pt")))
    S, A = predict(m, test3)
    res = {"hyperparams": hp, "val_KL": vl, "val_topics": vt, "n_train": len(tr), "n_val": len(va),
           "val": evaluate3(m, va), "test3_descriptive": evaluate3(m, test3),
           "test3_caricature": caricature(S, A),
           "note": "test3 var redan sedd (steg 2); beskrivande, ingen bedömning.",
           "model_sha256": hashlib.sha256(MODEL.read_bytes()).hexdigest()}
    # jämförelse: det gamla schemat på samma texter
    from nexis3.schema import Schema
    ck = torch.load("models/schema_s3.pt", weights_only=False)
    old = Schema(3, 1536, ck["hyperparams"]["h"], ck["hyperparams"]["dropout"])
    old.load_state_dict(ck["state_dict"])
    old.eval()
    So, _ = predict(old, test3)
    res["test3_caricature_old_schema"] = caricature(So, A)
    Path("results/sprint3_schema3b_train.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    t = res["test3_descriptive"]
    print(json.dumps({k: t[k] for k in ["S_argmax_eq_A_argmax", "target_eq_A_argmax", "vs_target", "vs_recency",
                                         "vs_overlap", "S_argmax_eq_A_argmax_when_A_not_target"]}, indent=1))
    print("karikatyr nytt:", res["test3_caricature"], "gammalt:", res["test3_caricature_old_schema"])


def cmd_test(a):
    lock = json.loads(Path("results/sprint3_schema3b_lock.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(MODEL.read_bytes()).hexdigest() == lock["model_sha256"], "schemat ändrat efter låsning"
    ck = torch.load(MODEL, weights_only=False)
    m = RelSchema(3, 1536, **ck["hyperparams"])
    m.load_state_dict(ck["state_dict"])
    m.eval()
    recs = load_records(sorted(Path(a.data).glob("test3b_*.pt")))
    orig = [r for r in recs if r["twin"] == 0]
    twin = {r["idx"]: r for r in recs if r["twin"] == 1}
    main = evaluate3(m, orig)
    S, A = predict(m, orig)
    St, At = predict(m, [twin[r["idx"]] for r in orig])
    dS = (S - St).abs().sum(-1).tolist()
    dA = (A - At).abs().sum(-1).tolist()
    sym = paired(dS, dA)
    res = {"n_pairs": len(orig), "main": main, "twins_main": evaluate3(m, list(twin.values())),
           "symmetry": {"mean_dS": sum(dS) / len(dS), "mean_dA": sum(dA) / len(dA), "dS_minus_dA": sym},
           "caricature": caricature(S, A),
           "main_pass": main["vs_target"]["z"] > 2,
           "symmetry_pass": sym["diff"] < 2 * sym["se"]}
    res["step2c_pass"] = res["main_pass"] and res["symmetry_pass"]
    Path("results/sprint3_schema3b_test.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ["symmetry", "caricature", "main_pass", "symmetry_pass", "step2c_pass"]}, indent=1))
    print("huvud:", json.dumps({k: main[k] for k in ["S_argmax_eq_A_argmax", "target_eq_A_argmax", "vs_target"]}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "test"])
    ap.add_argument("--data", required=True)
    a = ap.parse_args()
    cmd_train(a) if a.cmd == "train" else cmd_test(a)
