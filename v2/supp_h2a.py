"""Sprint 4 supplementary analysis after the gate-4 review (NOT a locked criterion).

The review found that the Blandat control for H2 (a) is nearly empty: the locked cells are the
agent's OWN coverage, which the donor's map it receives seldom shows as covered, so the lie barely
changes its input. Here (a) is redone with the cells the received map actually shows as covered
after step 3 (a_cells='shown'). For 1a this is identical to the locked analysis; for Blandat the lie
now removes real marks from its input. Also reports how many cells the lie changes.
Writes results/sprint4_supp_h2a.json."""
import json, os
import numpy as np
from bank import run_bank, bank_seeds
from eval_mvp import load, h2_effect, diff, RES, SEEDS

out = {}
seeds = bank_seeds()
for s in SEEDS:
    ag, _ = load(f"s4_shuffled_s{s}")
    base = run_bank(ag, seeds, a_cells="shown")
    lie = run_bank(ag, seeds, intervention="a", a_cells="shown")
    e = h2_effect(base, lie, "a")
    e["found_with"] = float(lie["found"].mean())
    out[f"shuffled_s{s}"] = e
    print(s, e, flush=True)
mvp = json.load(open(os.path.join(RES, "sprint4_mvp.json")))
cmp = {}
for s in SEEDS:
    f = mvp["runs"][f"s4_fixed_s{s}"]["h2"]["a"]
    b = out[f"shuffled_s{s}"]
    cmp[s] = diff(f["mean"], b["mean"], f["se"], b["se"])
out["fixed_minus_blandat_shown"] = cmp
json.dump(out, open(os.path.join(RES, "sprint4_supp_h2a.json"), "w"), indent=1)
print(cmp)
