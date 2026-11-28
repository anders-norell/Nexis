"""Sprint 4: träna kartan på train4 och mät F2 på validering (förregistrering del 2a).

    python train_kmap.py --data <dir med train4_*.pt>
Resultat: results/sprint4_f2.json, modell models/kmap.pt.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from nexis3.compress import load  # noqa: E402
from nexis3.kmap import M, KMap, baselines, fit, seq  # noqa: E402
from nexis3.topics_v3 import TRAIN_TOPICS  # noqa: E402
from extract_v4 import content_rows  # noqa: E402


def base(t):
    return t.replace("-lure", "")


def split(recs):
    tt = sorted(TRAIN_TOPICS)
    random.Random(31).shuffle(tt)
    vt = set(tt[:6])
    tr = [r for r in recs if not any(base(t) in vt for t in r["topics"])]
    va = [r for r in recs if base(r["topics"][r["roles"]["target_a"]]) in vt]
    return tr, va


def paired(a, b):
    d = torch.tensor(a) - torch.tensor(b)
    se = float(d.std() / len(d) ** 0.5)
    return {"diff": float(d.mean()), "se": se, "z": float(d.mean()) / se if se > 0 else 0.0, "n": len(d)}


@torch.no_grad()
def evaluate(model, recs, pos_mean, const, content_only=False):
    errs = {"map": [], "hold": [], "drift": [], "constant": []}
    for r in recs:
        H, q = seq(r)
        T = H.shape[0]
        obs = torch.arange(T) % M == 0
        rows = [t for t in range(T) if not obs[t]]
        if content_only:
            k = r["part2_start"]
            c = set(content_rows(r["answer_tokens"], r["answer_a"], 0, k) +
                    content_rows(r["answer_tokens"], r["answer_b"], k, T))
            rows = [t for t in rows if t in c]
        if not rows:
            continue
        S = model.run(H, q).exp()
        B = baselines(H, obs, pos_mean, const)
        for name, P in [("map", S)] + list(B.items()):
            errs[name].append(float((P[rows] - H[rows]).abs().sum(-1).mean()))
    out = {k: sum(v) / len(v) for k, v in errs.items()}
    out["n_texts"] = len(errs["map"])
    for b in ("hold", "drift", "constant"):
        out[f"map_vs_{b}"] = paired(errs[b], errs["map"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    a = ap.parse_args()
    recs = load(sorted(Path(a.data).glob("train4_*.pt")))
    tr, va = split(recs)
    print(f"träning {len(tr)}, validering {len(va)}")
    Htr = [seq(r)[0] for r in tr]
    L = max(len(h) for h in Htr)
    pos_mean = torch.stack([torch.stack([h[t] for h in Htr if len(h) > t]).mean(0) for t in range(L)])
    const = torch.cat(Htr).mean(0)
    model, vl = fit(KMap(), [seq(r) for r in tr], [seq(r) for r in va])
    Path("models").mkdir(exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "pos_mean": pos_mean, "const": const}, "models/kmap.pt")
    res = {"val_KL": vl, "n_train": len(tr), "n_val": len(va),
           "val_all_unobserved": evaluate(model, va, pos_mean, const),
           "val_content_unobserved": evaluate(model, va, pos_mean, const, content_only=True)}
    v = res["val_all_unobserved"]
    res["F2_pass"] = all(v[f"map_vs_{b}"]["z"] > 2 for b in ("hold", "drift", "constant"))
    Path("results/sprint4_f2.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
