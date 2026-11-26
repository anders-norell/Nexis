"""Sprint 4 steg 1: förutsättning F1 på base4 (förregistrerad i results/sprint4_prereg.md).

F1: andel texter där argmax av medel-A (lager 9–17, andel per mening) över del 1:s svarstoken = mål A OCH argmax över
del 2:s svarstoken = mål B. Krav ≥ 70 %. Bara texter med båda delarna rätt räknas i huvudmåttet; alla redovisas.
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from nexis3.compress import caricature, load  # noqa: E402


def main(d):
    recs = load(sorted(Path(d).glob("base4_*.pt")))
    rows = []
    for r in recs:
        H = caricature(r["att"])
        k = r["part2_start"]
        if k <= 0 or k >= len(H):
            rows.append(None)
            continue
        a1, a2 = int(H[:k].mean(0).argmax()), int(H[k:].mean(0).argmax())
        rows.append((a1 == r["roles"]["target_a"], a2 == r["roles"]["target_b"]))
    both = [r["correct_a"] and r["correct_b"] for r in recs]
    ok = [x for x, b in zip(rows, both) if x is not None and b]
    f1 = sum(x[0] and x[1] for x in ok) / max(1, len(ok))
    allv = [x for x in rows if x is not None]
    res = {"n": len(recs), "both_correct": sum(both) / len(recs),
           "correct_a": sum(r["correct_a"] for r in recs) / len(recs),
           "correct_b": sum(r["correct_b"] for r in recs) / len(recs),
           "has_part2": len(allv) / len(recs), "n_answer_tokens_median": sorted(r["n_answer_tokens"] for r in recs)[len(recs) // 2],
           "F1": f1, "F1_n": len(ok), "F1_pass": f1 >= 0.70,
           "F1_all_texts": sum(x[0] and x[1] for x in allv) / max(1, len(allv)),
           "examples": [r["answer"] for r in recs[:8]]}
    Path("results/sprint4_f1.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1])
