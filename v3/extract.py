"""Sprint 2 steg 3: data för schemat.

För varje text (version 2) besvaras uppgiften (girigt) och A mäts som i baslinjen. Dessutom sparas modellens inre
tillstånd i promptens positioner, alltså innan svaret skrivs:
  seg[s]  = medel av dolda tillstånd över meningens tokens, i lagren STATE_LAYERS
  last    = dolt tillstånd vid sista prompttoken (där första svarstoken väljs)
Schemat (tränas i arbetsytan) får bara de här tillstånden som indata och ska förutsäga A.

    python extract.py --split train   # frö 2000, 1200 texter
    python extract.py --split test    # frö 3000, 300 texter
Skrivs i bitar om 50 till results/schema_data/<split>_NNN.pt; avbrutet arbete återupptas.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from nexis3.attention import DEFAULT_MODEL, Measurer, load_model, model_info, weights_checksum  # noqa: E402
from nexis3.textgen import SYSTEM_PROMPT, answer_correct, make_items_v2, make_items_v3, make_twins_v3, overlap_choice  # noqa: E402

SPLITS = {"train": (2000, 1200), "test": (3000, 300),            # sprint 2 (textversion 2)
          "train3": (5000, 1500), "test3": (6000, 300),         # sprint 3 (version 3: undanhållna testämnen)
          "test3b": (7000, 600)}                                # sprint 3 steg 2c: 300 symmetripar (testämnen)
STATE_LAYERS = [9, 13, 17]
CHUNK = 50


def run_chunk(meas, items):
    recs = []
    for it in items:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": it.prompt}]
        m = meas.measure(msgs, max_new_tokens=24, segment_texts=it.sentences)
        recs.append({
            "idx": it.idx, "twin": getattr(it, "twin", 0), "target": it.target, "lure": it.lure, "topics": it.topics, "overlap_choice": overlap_choice(it.question, it.sentences),
            "answer": m.answer, "task_correct": answer_correct(it, m.answer),
            "A": torch.tensor(m.A), "A_first": torch.tensor(m.A_first), "A_mean": torch.tensor(m.A_mean),
            "seg": m.states["seg"], "last": m.states["last"],
        })
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=list(SPLITS), required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--out", default="results/schema_data")
    a = ap.parse_args()
    seed, n = SPLITS[a.split]
    n = a.n or n
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.split == "test3b":
        items = make_twins_v3(n // 2, seed)
    elif a.split.endswith("3"):
        items = make_items_v3(n, seed, "train" if a.split.startswith("train") else "test")
    else:
        items = make_items_v2(n, seed)
    model, tok = load_model(a.model, a.dtype)
    meas = Measurer(model, tok)
    meas.STATE_LAYERS = STATE_LAYERS
    ck0 = weights_checksum(model)
    for c0 in range(0, n, CHUNK):
        f = out / f"{a.split}_{c0 // CHUNK:03d}.pt"
        if f.exists():
            continue
        t0 = time.time()
        recs = run_chunk(meas, items[c0:c0 + CHUNK])
        torch.save({"split": a.split, "seed": seed, "layers": meas.layers, "state_layers": STATE_LAYERS,
                    "model_info": model_info(model, tok), "records": recs}, f)
        ok = sum(r["task_correct"] for r in recs)
        print(f"{f.name}: {len(recs)} texter, uppgift rätt {ok}, {time.time() - t0:.0f} s", flush=True)
    ck1 = weights_checksum(model)
    print(f"klart. vikter orörda: {ck0 == ck1}", flush=True)
    (out / f"{a.split}_done.txt").write_text(f"n={n} seed={seed} weights_unchanged={ck0 == ck1}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
