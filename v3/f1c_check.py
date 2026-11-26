"""Sprint 4 steg 1, tredje omtaget: F1c på base4c (förregistrerad i results/sprint4_prereg.md, tillägg 26 nov 16:00).

F1c räknas per del: en del bedöms om dess delsvar är rätt. Delen klaras om argmax av medel-A (lager 9–17, andel per
mening) över de token som skriver svarsordet är rätt målmening (del 1: mål A, del 2: mål B). Saknas innehållstoken
räknas delen som ej klarad. Krav: andel klarade delar ≥ 70 %.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nexis3.compress import caricature, load  # noqa: E402
from extract_v4 import content_rows  # noqa: E402


def main(d):
    recs = load(sorted(Path(d).glob("base4c_*.pt")))
    hits = {1: [], 2: []}
    for r in recs:
        H = caricature(r["att"])
        k = r["part2_start"]
        T = r["answer_tokens"]
        for part, ok, val, lo, hi, tgt in ((1, r["correct_a"], r["answer_a"], 0, k, r["roles"]["target_a"]),
                                           (2, r["correct_b"], r["answer_b"], k, len(T), r["roles"]["target_b"])):
            if not ok:
                continue
            idx = content_rows(T, val, lo, hi)
            hits[part].append(bool(idx) and int(H[idx].mean(0).argmax()) == tgt)
    allh = hits[1] + hits[2]
    f1c = sum(allh) / max(1, len(allh))
    res = {"n": len(recs), "parts_evaluated": len(allh), "F1c": f1c, "F1c_pass": f1c >= 0.70,
           "part1": sum(hits[1]) / max(1, len(hits[1])), "part1_n": len(hits[1]),
           "part2": sum(hits[2]) / max(1, len(hits[2])), "part2_n": len(hits[2]),
           "correct_a": sum(r["correct_a"] for r in recs) / len(recs),
           "correct_b": sum(r["correct_b"] for r in recs) / len(recs)}
    Path("results/sprint4_f1c.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=1, ensure_ascii=False))
    return res["F1c_pass"]


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1]) else 3)
