"""Baslinje för sprint 1 (v3): attention (A) och rapport (R) utan schema.

För varje text:
  1. Uppgiften ställs; modellen svarar (girigt). A mäts över de fyra meningarna
     medan svaret produceras.
  2. I samma samtal frågas "vilken mening fokuserade du mest på?"; svaret R tolkas
     som första siffran 1–4.
Mått: A pekar ut målet (argmax A = mål), R = mål, R = argmax A, uppgiften rätt,
medel-A på målet. Spridning: binomialt SE och Wilson 95 %-intervall; uppdelat
per målposition.

    python baseline.py --n 100 --seed 0 --out results/baseline_s0.json
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from nexis3.attention import DEFAULT_MODEL, Measurer, load_model, model_info, weights_checksum  # noqa: E402
from nexis3.textgen import REPORT_QUESTION, SYSTEM_PROMPT, answer_correct, make_items, overlap_choice, parse_report  # noqa: E402


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


def prop(flags):
    n = len(flags)
    k = sum(flags)
    p = k / n if n else 0.0
    return {"k": k, "n": n, "p": p, "se": math.sqrt(p * (1 - p) / n) if n else 0.0, "wilson95": wilson(k, n)}


def mean_se(xs):
    n = len(xs)
    if n == 0:
        return {"mean": None, "se": None, "n": 0}
    m = sum(xs) / n
    v = sum((x - m) ** 2 for x in xs) / (n - 1) if n > 1 else 0.0
    return {"mean": m, "se": math.sqrt(v / n), "n": n}


def run_items(meas: Measurer, items, log=print):
    rows = []
    for it in items:
        t0 = time.time()
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": it.prompt}]
        m1 = meas.measure(msgs, max_new_tokens=24, segment_texts=it.sentences)
        msgs2 = msgs + [{"role": "assistant", "content": m1.answer}, {"role": "user", "content": REPORT_QUESTION}]
        # A under rapportsvaret mäts över de fyra meningarna i första meddelandet.
        m2 = meas.measure(msgs2, max_new_tokens=6, segment_texts=it.sentences)
        R = parse_report(m2.answer)
        am = lambda v: max(range(4), key=lambda k: v[k])
        a_arg = am(m1.A)
        row = {
            "idx": it.idx, "target": it.target, "topics": it.topics, "lure": getattr(it, "lure", -1),
            "overlap_choice": overlap_choice(it.question, it.sentences),
            "answer": m1.answer, "task_correct": answer_correct(it, m1.answer),
            "A": m1.A, "A_argmax": a_arg, "mass_in_segments": m1.mass_in_segments,
            "A_mean": m1.A_mean, "A_mean_argmax": am(m1.A_mean), "A_first": m1.A_first, "A_first_argmax": am(m1.A_first),
            "A_per_layer": m1.per_layer,
            "report_text": m2.answer, "R": R, "A_report": m2.A,
            "secs": round(time.time() - t0, 2),
        }
        rows.append(row)
        log(f"[{it.idx:3d}] mål {it.target + 1} A-argmax {a_arg + 1} R {R + 1 if R is not None else '-'} "
            f"uppgift {'rätt' if row['task_correct'] else 'fel'}  A={[round(a, 3) for a in m1.A]}  {row['secs']} s")
    return rows


def summarize(rows):
    def sub(f, rs=rows):
        return prop([f(r) for r in rs])
    out = {
        "A_points_to_target": sub(lambda r: r["A_argmax"] == r["target"]),
        "R_equals_target": sub(lambda r: r["R"] == r["target"]),
        "R_equals_A_argmax": sub(lambda r: r["R"] == r["A_argmax"]),
        "task_correct": sub(lambda r: r["task_correct"]),
        "R_unparsable": sub(lambda r: r["R"] is None),
        "A_mean_points_to_target": sub(lambda r: r["A_mean_argmax"] == r["target"]),
        "A_first_points_to_target": sub(lambda r: r["A_first_argmax"] == r["target"]),
        "A_on_target": mean_se([r["A"][r["target"]] for r in rows]),
        "mass_in_segments": mean_se([r["mass_in_segments"] for r in rows]),
        "chance": 0.25,
    }
    out["overlap_points_to_target"] = sub(lambda r: r["overlap_choice"] == r["target"])
    lr = [r for r in rows if r.get("lure", -1) >= 0]
    if lr:
        out["A_points_to_lure"] = sub(lambda r: r["A_argmax"] == r["lure"], lr)
        out["R_equals_lure"] = sub(lambda r: r["R"] == r["lure"], lr)
        out["A_on_lure"] = mean_se([r["A"][r["lure"]] for r in lr])
    off = [r for r in rows if r["A_argmax"] != r["target"]]  # fallen där A och målet skiljer sig
    out["when_A_not_target"] = {
        "n": len(off),
        "R_equals_A_argmax": sub(lambda r: r["R"] == r["A_argmax"], off) if off else None,
        "R_equals_target": sub(lambda r: r["R"] == r["target"], off) if off else None,
    }
    by_pos = {}
    for p in range(4):
        rs = [r for r in rows if r["target"] == p]
        by_pos[str(p + 1)] = {
            "n": len(rs),
            "A_points_to_target": sub(lambda r: r["A_argmax"] == r["target"], rs)["p"],
            "R_equals_target": sub(lambda r: r["R"] == r["target"], rs)["p"],
        }
    out["by_target_position"] = by_pos
    out["A_argmax_position_counts"] = {str(p + 1): sum(r["A_argmax"] == p for r in rows) for p in range(4)}
    out["R_position_counts"] = {str(p + 1): sum(r["R"] == p for r in rows) for p in range(4)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/baseline_s0.json")
    ap.add_argument("--texts", type=int, default=1, choices=[1, 2], help="testtextversion")
    a = ap.parse_args()

    from nexis3.textgen import make
    items = make(a.texts, a.n, a.seed)
    model, tok = load_model(a.model, a.dtype)
    meas = Measurer(model, tok)
    ck0 = weights_checksum(model)
    t0 = time.time()
    rows = run_items(meas, items)
    ck1 = weights_checksum(model)
    res = {
        "model": a.model, "dtype": str(next(model.parameters()).dtype), "layers": meas.layers,
        "model_info": model_info(model, tok), "system_prompt": SYSTEM_PROMPT,
        "A_definition": "summa över meningens tokens, medel över lager/huvuden/svarspositioner (rad n_p-1..L-2), normerat",
        "n": a.n, "seed": a.seed, "texts_version": a.texts, "minutes": round((time.time() - t0) / 60, 1),
        "weights_sha256_before": ck0, "weights_sha256_after": ck1, "weights_unchanged": ck0 == ck1,
        "platform": platform.platform(), "python": sys.version.split()[0],
        "summary": summarize(rows), "items": [it.to_dict() for it in items], "rows": rows,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    s = res["summary"]
    print("\n=== Baslinje ===")
    for k in ["overlap_points_to_target", "A_points_to_lure", "R_equals_lure", "A_points_to_target", "A_mean_points_to_target", "A_first_points_to_target", "R_equals_target", "R_equals_A_argmax", "task_correct", "R_unparsable"]:
        v = s.get(k)
        if v is None:
            continue
        print(f"{k:22s} {v['k']:3d}/{v['n']}  {v['p']:.2f} ± {v['se']:.2f}  (95 %: {v['wilson95'][0]:.2f}–{v['wilson95'][1]:.2f})")
    w = s["when_A_not_target"]
    if w["n"]:
        print(f"när argmax A != mål (n={w['n']}): R = argmax A {w['R_equals_A_argmax']['p']:.2f}, R = mål {w['R_equals_target']['p']:.2f}")
    print(f"A på målet (medel)      {s['A_on_target']['mean']:.3f} ± {s['A_on_target']['se']:.3f}")
    print(f"vikter orörda: {res['weights_unchanged']}   sparat: {a.out}")


if __name__ == "__main__":
    main()
