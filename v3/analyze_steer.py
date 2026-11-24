"""Bedömning av G2b′ och redovisade mått (results/sprint2_prereg.md) ur results/sprint2_steer_test.jsonl."""
import json
import math
from pathlib import Path

rows = [json.loads(l) for l in Path("results/sprint2_steer_test.jsonl").read_text(encoding="utf-8").splitlines()]


def paired(a, b):
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    m = sum(d) / n
    se = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1) / n) if n > 1 else float("nan")
    return {"n": n, "mean_a": sum(a) / n, "mean_b": sum(b) / n, "diff": m, "se": se, "z": m / se if se else float("inf")}


wrong = lambda c: [int(not r[c]["correct"]) for r in rows]
corr = lambda c: [int(r[c]["correct"]) for r in rows]
res = {"n": len(rows), "beta": rows[0]["beta"]}
res["G2b_prime_wrong_forced_minus_base"] = paired(wrong("forced"), wrong("base"))
res["G2b_prime_pass"] = res["G2b_prime_wrong_forced_minus_base"]["z"] > 2
fil = [r for r in rows if r["forced_is_filler"]]
res["mentions_forced_value_fillers"] = paired([int(r["forced"]["mentions_forced_value"]) for r in fil],
                                              [int(r["base"]["mentions_forced_value"]) for r in fil])
res["A_on_forced_mid"] = paired([r["forced"]["A"][r["forced_k"]] for r in rows], [r["base"]["A"][r["forced_k"]] for r in rows])
res["A_on_forced_late"] = paired([r["forced"]["A_late"][r["forced_k"]] for r in rows], [r["base"]["A_late"][r["forced_k"]] for r in rows])
res["ablation_correct_own_minus_base"] = paired(corr("own"), corr("base"))
res["control_correct_own_minus_perm"] = paired(corr("own"), corr("perm"))
res["A_on_target_own_minus_base"] = paired([r["own"]["A"][r["target"]] for r in rows], [r["base"]["A"][r["target"]] for r in rows])
res["A_late_on_target_own_minus_base"] = paired([r["own"]["A_late"][r["target"]] for r in rows], [r["base"]["A_late"][r["target"]] for r in rows])
res["correct_rates"] = {c: sum(corr(c)) / len(rows) for c in ("base", "own", "perm", "forced")}
lure_base_wrong = [r for r in rows if not r["base"]["correct"]]
res["base_wrong_fixed_by_own"] = {"n": len(lure_base_wrong), "fixed": sum(r["own"]["correct"] for r in lure_base_wrong)}
Path("results/sprint2_steer.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps(res, indent=1, ensure_ascii=False))
