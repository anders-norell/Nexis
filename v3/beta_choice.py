"""Välj beta enligt regeln i results/sprint2_prereg.md (tillägg 24 nov)."""
import json
import sys
from pathlib import Path

rows = [json.loads(l) for l in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()]
base = {json.loads(l)["idx"]: json.loads(l) for l in Path(sys.argv[2]).read_text(encoding="utf-8").splitlines()}
acc0 = sum(b["correct"] for b in base.values()) / len(base)
res = {}
for beta in sorted({r["beta"] for r in rows}):
    rs = [r for r in rows if r["beta"] == beta]
    shift = sum(r["forced"]["A"][r["forced_k"]] - base[r["idx"]]["A"][r["forced_k"]] for r in rs) / len(rs)
    acc = sum(r["own"]["correct"] for r in rs) / len(rs)
    ok_base = [r for r in rs if base[r["idx"]]["correct"]]
    flip = sum(not r["forced"]["correct"] for r in ok_base) / max(1, len(ok_base))
    rule_behaviour = len(sys.argv) > 3 and sys.argv[3] == "behaviour"
    res[beta] = {"n": len(rs), "forced_shift": shift, "forced_flip_rate": flip, "own_acc": acc, "base_acc": acc0,
                 "ok_a": (flip >= 0.20) if rule_behaviour else (shift >= 0.20), "ok_b": acc >= acc0 - 0.05}
cands = [b for b, v in res.items() if v["ok_a"] and v["ok_b"]]
choice = min(cands) if cands else max([b for b, v in res.items() if v["ok_b"]] or [min(res)])
out = {"beta": choice, "table": res, "rule": "results/sprint2_prereg.md, tillägg 24 nov"}
Path("results/sprint2_beta_choice.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps(out, indent=1))
