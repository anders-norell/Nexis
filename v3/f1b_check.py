"""Sprint 4 steg 1, omtag: F1b på base4b (förregistrerad i results/sprint4_prereg.md, tillägg 26 nov).

F1b: bland texter med båda delsvaren rätt ska argmax av medel-A (lager 9–17, andel per mening) över de svarstoken som
skriver svarsordet i del 1 vara mål A, och över motsvarande token i del 2 vara mål B, i minst 70 %. Texter där något
delsvar saknar innehållstoken räknas som ej klarade.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nexis3.compress import caricature, load  # noqa: E402
from extract_v4 import content_rows  # noqa: E402


def main(d):
    recs = load(sorted(Path(d).glob("base4b_*.pt")))
    ok, n_ok, ex = 0, 0, []
    for r in recs:
        if not (r["correct_a"] and r["correct_b"]):
            continue
        n_ok += 1
        H = caricature(r["att"])
        k = r["part2_start"]
        i1 = content_rows(r["answer_tokens"], r["answer_a"], 0, k)
        i2 = content_rows(r["answer_tokens"], r["answer_b"], k, len(r["answer_tokens"]))
        if i1 and i2:
            a1, a2 = int(H[i1].mean(0).argmax()), int(H[i2].mean(0).argmax())
            hit = a1 == r["roles"]["target_a"] and a2 == r["roles"]["target_b"]
            ok += hit
        if len(ex) < 6:
            ex.append({"answer": r["answer"], "content_tokens": [[r["answer_tokens"][i] for i in i1],
                                                                 [r["answer_tokens"][i] for i in i2]]})
    res = {"n": len(recs), "both_correct": n_ok / len(recs), "F1b": ok / max(1, n_ok), "F1b_n": n_ok,
           "F1b_pass": ok / max(1, n_ok) >= 0.70, "examples": ex}
    Path("results/sprint4_f1b.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1])
