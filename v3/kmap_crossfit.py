"""Sprint 4: korsanpassade kartor för rapportträningen (granskning före låsning, anm. 7). [Claudes förslag]

Kartan i models/kmap.pt har sett 938 av train4:s texter, så dess S är bättre där än på test4. Rapporten ska tränas på S
av samma slag som vid test: varje train4-text får S från en karta som inte tränats på den (två delar efter idx % 2).
    python kmap_crossfit.py --data <v4_data>   → models/kmap_fold0.pt, models/kmap_fold1.pt
Fold f-kartan är tränad på texter med idx % 2 != f och används för texter med idx % 2 == f.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from nexis3.compress import load  # noqa: E402
from nexis3.kmap import KMap, fit, seq  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    a = ap.parse_args()
    recs = load(sorted(Path(a.data).glob("train4_*.pt")))
    out = {}
    for f in (0, 1):
        pool = [r for r in recs if r["idx"] % 2 != f]
        random.Random(40 + f).shuffle(pool)
        nv = len(pool) // 7
        model, vl = fit(KMap(), [seq(r) for r in pool[nv:]], [seq(r) for r in pool[:nv]])
        torch.save({"state_dict": model.state_dict(), "fold": f}, f"models/kmap_fold{f}.pt")
        held = [r for r in recs if r["idx"] % 2 == f]
        agree = []
        with torch.no_grad():
            for r in held:
                H, q = seq(r)
                S = model.run(H, q).exp()
                un = [t for t in range(len(H)) if t % 4]
                agree += [float(S[t].argmax() == H[t].argmax()) for t in un]
        out[f"fold{f}"] = {"val_KL": vl, "n_fit": len(pool), "heldout_argmax_agree_unobserved": sum(agree) / len(agree)}
    Path("results/sprint4_kmap_crossfit.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
