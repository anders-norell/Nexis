"""Sprint 4: låst extern kombinerare (granskning före låsning, omgång 2, R2-2/R2-3). [Claudes förslag]

Multinomial logistisk regression på ledtrådar utan introspektion, tränad på train4 (oobserverade innehållstoken,
korsanpassad S), facit = argmax A:
  utan S:  logit_j = b_j + c_j·del2 + w_T·[T = j]                  → referens för "inre väg" (kontrollen ska slå den)
  med S:   dessutom + w_S·log S_j                                  → rationell referens för schemamodellen
    python combiner_fit.py --data <v4_data>  → results/sprint4_combiner.json (parametrar, låses)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import report_train as rt  # noqa: E402
from nexis3.compress import load  # noqa: E402
from nexis3.kmap import seq  # noqa: E402
from nexis3.textgen import make_items_v4  # noqa: E402
from extract_v4 import SPLITS, content_rows  # noqa: E402


def feats(rows):
    S = torch.stack([x["S"] for x in rows]).float().clamp(min=1e-6)
    T = torch.tensor([x["txt"] for x in rows])
    P2 = torch.tensor([float(x["part2"]) for x in rows])
    y = torch.tensor([x["aA"] for x in rows])
    return S, T, P2, y


def logits(p, S, T, P2, with_s):
    b, c, wT, wS = torch.tensor(p["b"]), torch.tensor(p["c"]), p["wT"], p["wS"]
    z = b[None] + c[None] * P2[:, None] + wT * torch.nn.functional.one_hot(T, 4).float()
    if with_s:
        z = z + wS * S.log()
    return z


def predict(p, rows, with_s):
    S, T, P2, _ = feats(rows)
    return logits(p, S, T, P2, with_s).argmax(-1).tolist()


def fit(rows, with_s):
    S, T, P2, y = feats(rows)
    b = torch.zeros(4, requires_grad=True); c = torch.zeros(4, requires_grad=True)
    wT = torch.zeros((), requires_grad=True); wS = torch.zeros((), requires_grad=True)
    opt = torch.optim.LBFGS([b, c, wT, wS], max_iter=500)

    def closure():
        opt.zero_grad()
        z = b[None] + c[None] * P2[:, None] + wT * torch.nn.functional.one_hot(T, 4).float()
        if with_s:
            z = z + wS * S.log()
        l = torch.nn.functional.cross_entropy(z, y)
        l.backward()
        return l
    opt.step(closure)
    return {"b": b.tolist(), "c": c.tolist(), "wT": float(wT), "wS": float(wS) if with_s else 0.0}


def train_rows(data):
    recs = load(sorted(Path(data).glob("train4_*.pt")))
    s0, n, pool = SPLITS["train4"]
    items = {it.idx: it for it in make_items_v4(n, s0, pool)}
    km = {f: rt.load_kmap(f"models/kmap_fold{f}.pt") for f in (0, 1)}
    rows = []
    for r in recs:
        it = items[r["idx"]]
        H, q = seq(r)
        with torch.no_grad():
            S = km[r["idx"] % 2].run(H, q).exp()
        T, k = len(r["answer_tokens"]), r["part2_start"]
        c = sorted(set(content_rows(r["answer_tokens"], r["answer_a"], 0, k) +
                       content_rows(r["answer_tokens"], r["answer_b"], k, T)))
        for t in c:
            if t % 4:
                rows.append({"S": S[t], "txt": it.target_a if t < k else it.target_b, "part2": t >= k,
                             "aA": int(H[t].argmax()), "aS": int(S[t].argmax())})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    a = ap.parse_args()
    rows = train_rows(a.data)
    out = {"n": len(rows), "no_s": fit(rows, False), "with_s": fit(rows, True)}
    for k, ws in (("no_s", False), ("with_s", True)):
        pr = predict(out[k], rows, ws)
        out[k]["train_acc"] = sum(p == x["aA"] for p, x in zip(pr, rows)) / len(rows)
    D = [x for x in rows if x["aS"] != x["aA"]]
    out["with_s"]["train_D_R_eq_S"] = sum(p == x["aS"] for p, x in zip(predict(out["with_s"], D, True), D)) / len(D)
    out["train_S_eq_A"] = sum(x["aS"] == x["aA"] for x in rows) / len(rows)
    Path("results/sprint4_combiner.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
