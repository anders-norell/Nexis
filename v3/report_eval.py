"""Sprint 4 steg 6: kärntestet på test4, en gång (förregistrering del 2b). Körs på Anders dator efter låsning.

    python report_eval.py lock "<Anders svar>"   # (här i arbetsytan) sha256 av kod, karta och förregistrering
    python report_eval.py run                    # (på hans dator) vägrar om något låst ändrats eller om testet redan körts
Resultat: results/sprint4_test.json.
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
LOCKED = ["report_eval.py", "report_train.py", "nexis3/report.py", "nexis3/kmap.py", "models/kmap.pt",
          "extract_v4.py", "results/sprint4_prereg.md"]
HABIT = 3          # vanligaste facit i träningen (mening 4), förregistrerat som positionsvana


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def paired(a, b):
    d = torch.tensor(a, dtype=torch.float64) - torch.tensor(b, dtype=torch.float64)
    se = float(d.std() / len(d) ** 0.5) if len(d) > 1 else 0.0
    return {"diff": float(d.mean()), "se": se, "z": float(d.mean()) / se if se > 0 else 0.0, "n": len(d)}


def test_tokens():
    recs = load(sorted(rt.DATA.glob("test4_*.pt")))
    s0, n, pool = SPLITS["test4"]
    items = {it.idx: it for it in make_items_v4(n, s0, pool)}
    km = rt.load_kmap()
    rows = []
    rng = random.Random(4242)
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
            aS, aA = int(S[t].argmax()), int(H[t].argmax())
            j = rng.choice([x for x in range(4) if x != aS])
            Sm = torch.full((4,), 0.05); Sm[j] = 0.85
            rows.append({"idx": r["idx"], "t": t, "obs": t % 4 == 0, "part2": t >= k, "since_part2": t - k,
                         "user": it.prompt, "partial": "".join(r["answer_tokens"][:t + 1]),
                         "S": S[t], "Sm": Sm, "j": j, "aS": aS, "aA": aA})
    return rows


def per_text(rows, f):
    by = {}
    for x in rows:
        v = f(x)
        if v is not None:
            by.setdefault(x["idx"], []).append(v)
    return {k: sum(v) / len(v) for k, v in by.items()}


def cmp(rows, fa, fb):
    a, b = per_text(rows, fa), per_text(rows, fb)
    ks = sorted(set(a) & set(b))
    return paired([a[k] for k in ks], [b[k] for k in ks])


def cmd_lock(answer):
    assert not LOCK.exists(), "redan låst"
    LOCK.write_text(json.dumps({"answer": answer, "sha256": {f: sha(f) for f in LOCKED}}, indent=1,
                               ensure_ascii=False), encoding="utf-8")
    print("låst", LOCK)


@torch.no_grad()
def cmd_run():
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    for f, h in lock["sha256"].items():
        assert sha(f) == h, f"{f} ändrad efter låsning"
    assert not OUT.exists(), "testet är redan kört"
    meas, mods, proj, _ = rt.setup(rt.DEFAULT_MODEL, "auto")
    rep = Reporter(meas, proj)
    rows = test_tokens()
    print(f"{len(rows)} innehållstoken i {len({x['idx'] for x in rows})} texter", flush=True)
    uni = torch.full((4,), 0.25)
    hashes = {}
    for cond in ("schema", "control"):
        p = f"models/report_{cond}.pt"
        hashes[cond] = sha(p)
        load_lora_state(mods, proj, torch.load(p, weights_only=False))
        for i, x in enumerate(rows):
            if cond == "schema":
                x["R"] = int(rep.logits(SYSTEM_PROMPT, x["user"], x["partial"], x["S"]).argmax())
                x["Rm"] = int(rep.logits(SYSTEM_PROMPT, x["user"], x["partial"], x["Sm"]).argmax())
            else:
                x["Rc"] = int(rep.logits(SYSTEM_PROMPT, x["user"], x["partial"], uni).argmax())
            if (i + 1) % 100 == 0:
                print(f"{cond}: {i + 1}/{len(rows)}", flush=True)
    D = [x for x in rows if x["aS"] != x["aA"]]
    E = [x for x in rows if x["aS"] == x["aA"]]
    res = {
        "n_tokens": len(rows), "n_texts": len({x["idx"] for x in rows}), "n_differ": len(D), "model_sha256": hashes,
        "K4a": cmp(D, lambda x: float(x["R"] == x["aS"]), lambda x: float(x["R"] == x["aA"])),
        "K4b": cmp(rows, lambda x: float(x["Rm"] == x["j"]), lambda x: float(x["Rc"] == x["j"])),
        "K4c_vs_control": cmp(E, lambda x: float(x["R"] == x["aA"]), lambda x: float(x["Rc"] == x["aA"])),
        "K4c_vs_habit": cmp(E, lambda x: float(x["R"] == x["aA"]), lambda x: float(HABIT == x["aA"])),
        "control_vs_habit_all": cmp(rows, lambda x: float(x["Rc"] == x["aA"]), lambda x: float(HABIT == x["aA"])),
        "rates": {
            "schema_R_eq_A_all": sum(x["R"] == x["aA"] for x in rows) / len(rows),
            "schema_R_eq_S_all": sum(x["R"] == x["aS"] for x in rows) / len(rows),
            "control_R_eq_A_all": sum(x["Rc"] == x["aA"] for x in rows) / len(rows),
            "habit_eq_A_all": sum(HABIT == x["aA"] for x in rows) / len(rows),
            "map_eq_A_all": len(E) / len(rows),
            "D_schema_R_eq_S": sum(x["R"] == x["aS"] for x in D) / max(1, len(D)),
            "D_schema_R_eq_A": sum(x["R"] == x["aA"] for x in D) / max(1, len(D)),
            "D_control_R_eq_A": sum(x["Rc"] == x["aA"] for x in D) / max(1, len(D)),
            "manip_follow": sum(x["Rm"] == x["j"] for x in rows) / len(rows),
        },
        "lag_part2": {str(k): {"n": len(g), "S_eq_A": sum(x["aS"] == x["aA"] for x in g) / len(g),
                               "R_eq_S": sum(x["R"] == x["aS"] for x in g) / len(g),
                               "R_eq_A": sum(x["R"] == x["aA"] for x in g) / len(g)}
                      for k in range(0, 8)
                      for g in [[x for x in rows if x["part2"] and x["since_part2"] == k]] if g},
    }
    res["K4a_pass"] = res["K4a"]["z"] > 2
    res["K4b_pass"] = res["K4b"]["z"] > 2
    res["K4c_pass"] = res["K4c_vs_control"]["z"] > 2 and res["K4c_vs_habit"]["z"] > 2
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ["K4a", "K4b", "K4c_vs_control", "K4c_vs_habit", "control_vs_habit_all",
                                          "K4a_pass", "K4b_pass", "K4c_pass"]}, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "lock":
        cmd_lock(sys.argv[2])
    else:
        cmd_run()
