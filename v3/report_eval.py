"""Sprint 4 steg 6: kärntestet på test4, en gång (förregistrering del 2b, reviderad efter granskning före låsning).

    python report_eval.py lock "<Anders svar>"   # efter rapportträningen: sha256 av kod, kartor, rapportmodeller, prereg
    python report_eval.py run                    # på Anders dator; vägrar om något låst ändrats eller testet redan körts
Resultat: results/sprint4_test.json. Alla mått på OOBSERVERADE innehållstoken (t % 4 != 0), parat per text.
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import report_train as rt  # noqa: E402
from nexis3.report import Reporter, load_lora_state  # noqa: E402
from nexis3.compress import load  # noqa: E402
from nexis3.kmap import seq  # noqa: E402
from nexis3.textgen import SYSTEM_PROMPT, make_items_v4  # noqa: E402
from extract_v4 import SPLITS, content_rows  # noqa: E402

LOCK = Path("results/sprint4_lock.json")
OUT = Path("results/sprint4_test.json")
LOCKED = ["report_eval.py", "report_train.py", "kmap_crossfit.py", "nexis3/report.py", "nexis3/kmap.py",
          "nexis3/compress.py", "nexis3/textgen.py", "nexis3/topics_v3.py", "nexis3/attention.py", "extract_v4.py",
          "models/kmap.pt", "models/kmap_fold0.pt", "models/kmap_fold1.pt",
          "models/report_schema.pt", "models/report_control.pt", "results/sprint4_prereg.md"]
HABIT = 3          # vanligaste facit i träningen (mening 4), förregistrerad positionsvana
SEED = 4242


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def paired(a, b):
    d = torch.tensor(a, dtype=torch.float64) - torch.tensor(b, dtype=torch.float64)
    se = float(d.std() / len(d) ** 0.5) if len(d) > 1 else 0.0
    return {"diff": float(d.mean()), "se": se, "z": float(d.mean()) / se if se > 0 else
            (float("inf") if float(d.mean()) > 0 else 0.0), "n_texts": len(d)}


def test_tokens(km=None):
    """Oobserverade innehållstoken i test4 med S från models/kmap.pt (tränad utan test4:s texter och ämnen)."""
    recs = load(sorted(rt.DATA.glob("test4_*.pt")))
    s0, n, pool = SPLITS["test4"]
    items = {it.idx: it for it in make_items_v4(n, s0, pool)}
    km = km or rt.load_kmap()
    rows = []
    rng = random.Random(SEED)
    for r in recs:
        it = items[r["idx"]]
        H, q = seq(r)
        with torch.no_grad():
            S = km.run(H, q).exp()
        T = len(r["answer_tokens"])
        k = r["part2_start"]
        c = sorted(set(content_rows(r["answer_tokens"], r["answer_a"], 0, k) +
                       content_rows(r["answer_tokens"], r["answer_b"], k, T)))
        for t in c:
            if t % 4 == 0:
                continue
            aS, aA = int(S[t].argmax()), int(H[t].argmax())
            txt = it.target_a if t < k else it.target_b      # textvägen: meningen som innehåller svarsordet
            j = rng.choice([x for x in range(4) if x not in (aS, aA, txt)])
            Sm = S[t].clone(); Sm[aS], Sm[j] = S[t][j], S[t][aS]   # dosmatchad: byt plats på topp och j
            rows.append({"idx": r["idx"], "t": t, "part2": t >= k, "since_part2": t - k,
                         "user": it.prompt, "partial": "".join(r["answer_tokens"][:t + 1]),
                         "S": S[t], "Sm": Sm, "j": j, "aS": aS, "aA": aA, "txt": txt,
                         "dose_swap": float((S[t] - Sm).abs().sum()), "dose_SA": float((S[t] - H[t]).abs().sum())})
    return rows


def per_text(rows, f):
    by = {}
    for x in rows:
        by.setdefault(x["idx"], []).append(f(x))
    return {k: sum(v) / len(v) for k, v in by.items()}


def cmp(rows, fa, fb):
    a, b = per_text(rows, fa), per_text(rows, fb)
    ks = sorted(set(a) & set(b))
    return paired([a[k] for k in ks], [b[k] for k in ks])


def rate(rows, f):
    return sum(f(x) for x in rows) / len(rows) if rows else None


def k4a_contrast(x):
    return float(x["R"] == x["aS"]) - float(x["R"] == x["aA"])


def k4a_null(x):
    return float(x["Rc"] == x["aS"]) - float(x["Rc"] == x["aA"])


def summarize(rows):
    D = [x for x in rows if x["aS"] != x["aA"]]
    Dsub = [x for x in D if x["aA"] != x["txt"]]
    res = {
        "n_tokens": len(rows), "n_texts": len({x["idx"] for x in rows}),
        "n_D": len(D), "n_D_texts": len({x["idx"] for x in D}),
        "n_Dsub": len(Dsub), "n_Dsub_texts": len({x["idx"] for x in Dsub}),
        # grind
        "K4a": cmp(D, k4a_contrast, k4a_null),
        "K4b": cmp(rows, lambda x: float(x["Rm"] == x["j"]), lambda x: float(x["R"] == x["j"])),
        "K4c": cmp(rows, lambda x: float(x["R"] == x["aA"]), lambda x: float(HABIT == x["aA"])),
        # redovisas
        "K4a_Dsub": cmp(Dsub, k4a_contrast, k4a_null) if Dsub else None,
        "K4c_vs_control": cmp(rows, lambda x: float(x["R"] == x["aA"]), lambda x: float(x["Rc"] == x["aA"])),
        "K4c_vs_text": cmp(rows, lambda x: float(x["R"] == x["aA"]), lambda x: float(x["txt"] == x["aA"])),
        "control_vs_habit": cmp(rows, lambda x: float(x["Rc"] == x["aA"]), lambda x: float(HABIT == x["aA"])),
        "control_vs_text": cmp(rows, lambda x: float(x["Rc"] == x["aA"]), lambda x: float(x["txt"] == x["aA"])),
        "control_vs_text_Dsub": cmp(Dsub, lambda x: float(x["Rc"] == x["aA"]),
                                    lambda x: float(x["txt"] == x["aA"])) if Dsub else None,
        "dose": {"swap_L1_mean": rate(rows, lambda x: x["dose_swap"]), "S_vs_A_L1_mean_D": rate(D, lambda x: x["dose_SA"])},
        "rates": {
            "all": {k: rate(rows, f) for k, f in [
                ("schema_R_eq_A", lambda x: x["R"] == x["aA"]), ("schema_R_eq_S", lambda x: x["R"] == x["aS"]),
                ("control_R_eq_A", lambda x: x["Rc"] == x["aA"]), ("habit_eq_A", lambda x: HABIT == x["aA"]),
                ("text_eq_A", lambda x: x["txt"] == x["aA"]), ("map_eq_A", lambda x: x["aS"] == x["aA"]),
                ("manip_R_eq_j", lambda x: x["Rm"] == x["j"]), ("true_R_eq_j", lambda x: x["R"] == x["j"])]},
            "D": {k: rate(D, f) for k, f in [
                ("schema_R_eq_S", lambda x: x["R"] == x["aS"]), ("schema_R_eq_A", lambda x: x["R"] == x["aA"]),
                ("control_R_eq_S", lambda x: x["Rc"] == x["aS"]), ("control_R_eq_A", lambda x: x["Rc"] == x["aA"]),
                ("S_is_4", lambda x: x["aS"] == HABIT), ("S_is_text", lambda x: x["aS"] == x["txt"])]},
            "D_where_control_right": {"n": sum(x["Rc"] == x["aA"] for x in D),
                                      "schema_R_eq_S": rate([x for x in D if x["Rc"] == x["aA"]],
                                                            lambda x: x["R"] == x["aS"])},
        },
        "lag_part2": {str(k): {"n": len(g), "S_eq_A": rate(g, lambda x: x["aS"] == x["aA"]),
                               "R_eq_S": rate(g, lambda x: x["R"] == x["aS"]),
                               "R_eq_A": rate(g, lambda x: x["R"] == x["aA"])}
                      for k in range(0, 12)
                      for g in [[x for x in rows if x["part2"] and x["since_part2"] == k]] if g},
    }
    res["K4a_pass"] = res["K4a"]["z"] > 2
    res["K4b_pass"] = res["K4b"]["z"] > 2
    res["K4c_pass"] = res["K4c"]["z"] > 2
    res["grind4"] = res["K4a_pass"] and res["K4b_pass"] and res["K4c_pass"]
    return res


def cmd_lock(answer):
    assert not LOCK.exists(), "redan låst"
    LOCK.write_text(json.dumps({"answer": answer, "sha256": {f: sha(f) for f in LOCKED}}, indent=1,
                               ensure_ascii=False), encoding="utf-8")
    print("låst", LOCK)


@torch.no_grad()
def cmd_run():
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    assert set(lock["sha256"]) == set(LOCKED), "låslistan stämmer inte"
    for f, h in lock["sha256"].items():
        assert sha(f) == h, f"{f} ändrad efter låsning"
    assert not OUT.exists(), "testet är redan kört"
    meas, mods, proj, _ = rt.setup(rt.DEFAULT_MODEL, "auto")
    rep = Reporter(meas, proj, mods)
    rows = test_tokens()
    print(f"{len(rows)} oobserverade innehållstoken i {len({x['idx'] for x in rows})} texter", flush=True)
    uni = torch.full((4,), 0.25)
    for cond in ("schema", "control"):
        load_lora_state(mods, proj, torch.load(f"models/report_{cond}.pt", weights_only=False))
        for i, x in enumerate(rows):
            if cond == "schema":
                x["R"] = int(rep.logits(SYSTEM_PROMPT, x["user"], x["partial"], x["S"]).argmax())
                x["Rm"] = int(rep.logits(SYSTEM_PROMPT, x["user"], x["partial"], x["Sm"]).argmax())
            else:
                x["Rc"] = int(rep.logits(SYSTEM_PROMPT, x["user"], x["partial"], uni).argmax())
            if (i + 1) % 100 == 0:
                print(f"{cond}: {i + 1}/{len(rows)}", flush=True)
    res = summarize(rows)
    res["lock_sha256"] = sha(LOCK)
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ["K4a", "K4b", "K4c", "K4a_Dsub", "control_vs_habit", "control_vs_text",
                                          "K4a_pass", "K4b_pass", "K4c_pass", "grind4"]}, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "lock":
        cmd_lock(sys.argv[2])
    else:
        cmd_run()
