"""Sprint 4 steg 4: rapportmodell och kontrollmodell (förregistrering del 2b). Körs på Anders dator.

    python report_train.py time                 # 20 exempel: sekunder per exempel och uppskattad total tid
    python report_train.py train --cond schema  # LoRA + projektor med kartans S
    python report_train.py train --cond control # samma exempel, S = likformig
Data: results/v4_data/train4_*.pt (aldrig test4). S från korsanpassade kartor (models/kmap_fold{0,1}.pt). Exempel: per text två innehållstoken (en per del, slumpvis) och en
annan svarstoken; facit = argmax av faktisk A vid token t. Modeller: models/report_<cond>.pt.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from nexis3.attention import DEFAULT_MODEL, Measurer, load_model, weights_checksum  # noqa: E402
from nexis3.compress import load  # noqa: E402
from nexis3.kmap import KMap, seq  # noqa: E402
from nexis3.report import Reporter, SoftProj, add_lora, lora_state  # noqa: E402
from nexis3.textgen import SYSTEM_PROMPT, make_items_v4  # noqa: E402
from extract_v4 import SPLITS, content_rows  # noqa: E402

import os
DATA = Path(os.environ.get("NEXIS_V4_DATA", "results/v4_data"))


def load_kmap(path="models/kmap.pt"):
    ck = torch.load(path, weights_only=False)
    m = KMap()
    m.load_state_dict(ck["state_dict"]); m.eval()
    return m


def examples(split, seed=0):
    """Lista av (system, user, partial_answer, S_t, A_t, target, meta)."""
    recs = load(sorted(DATA.glob(f"{split}_*.pt")))
    s0, n, pool = SPLITS[split]
    items = {it.idx: it for it in make_items_v4(n, s0, pool)}
    # korsanpassning (granskning anm. 7): varje text får S från en karta som inte tränats på den
    km = {f: load_kmap(f"models/kmap_fold{f}.pt") for f in (0, 1)}
    rng = random.Random(seed)
    out = []
    for r in recs:
        it = items[r["idx"]]
        H, q = seq(r)
        with torch.no_grad():
            S = km[r["idx"] % 2].run(H, q).exp()
        T = len(r["answer_tokens"])
        k = r["part2_start"]
        c1 = content_rows(r["answer_tokens"], r["answer_a"], 0, k)
        c2 = content_rows(r["answer_tokens"], r["answer_b"], k, T)
        picks = ([("c1", rng.choice(c1))] if c1 else []) + ([("c2", rng.choice(c2))] if c2 else [])
        other = [t for t in range(T) if t not in set(c1 + c2)]
        if other:
            picks.append(("other", rng.choice(other)))
        for kind, t in picks:
            partial = "".join(r["answer_tokens"][:t + 1])
            out.append((SYSTEM_PROMPT, it.prompt, partial, S[t].clone(), H[t].clone(), int(H[t].argmax()),
                        {"idx": r["idx"], "t": t, "kind": kind, "obs": t % 4 == 0}))
    return out


def setup(model_name, dtype):
    model, tok = load_model(model_name, dtype)
    meas = Measurer(model, tok)
    ck = weights_checksum(model)
    mods = add_lora(model)
    proj = SoftProj(model.config.hidden_size)
    return meas, mods, proj, ck


def train(meas, mods, proj, exs, cond, epochs=1, lr=1e-4, log=None):
    params = [p for m in mods for p in (m.A, m.B)] + list(proj.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    rep = Reporter(meas, proj, mods)
    uni = torch.full((4,), 0.25)
    meas.model.train()
    t0, run = time.time(), 0.0
    for ep in range(epochs):
        order = list(range(len(exs)))
        random.Random(100 + ep).shuffle(order)
        for i, j in enumerate(order):
            sy, us, pa, S, A, y, meta = exs[j]
            lg = rep.logits(sy, us, pa, S if cond == "schema" else uni)
            loss = F.cross_entropy(lg[None], torch.tensor([y]))
            opt.zero_grad(); loss.backward(); opt.step()
            run = 0.98 * run + 0.02 * float(loss) if i else float(loss)
            if log and (i + 1) % 50 == 0:
                log(f"epok {ep} ex {i + 1}/{len(exs)} förlust {run:.3f} {time.time() - t0:.0f} s")
    meas.model.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["time", "train"])
    ap.add_argument("--cond", choices=["schema", "control"], default="schema")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--epochs", type=int, default=1)
    a = ap.parse_args()
    exs = examples("train4")
    log = lambda s: print(s, flush=True)
    log(f"{len(exs)} träningsexempel (andel där kartan ≠ attention: "
        f"{sum(int(e[3].argmax()) != e[5] for e in exs) / len(exs):.2f})")
    meas, mods, proj, ck0 = setup(a.model, a.dtype)
    if a.cmd == "time":
        # omtokenisering (granskning anm. 9): ger det kapade svaret samma token som modellen skrev?
        same = 0
        for sy, us, pa, S, A, y, meta in exs:
            r_tok = meas.tok(pa, add_special_tokens=False)["input_ids"]
            same += len(r_tok) == meta["t"] + 1
        log(f"omtokenisering: samma antal token i {same}/{len(exs)} kapade svar")
        t0 = time.time()
        train(meas, mods, proj, exs[:20], "schema")
        per = (time.time() - t0) / 20
        log(f"{per:.1f} s per exempel; en modell ≈ {per * len(exs) * a.epochs / 3600:.1f} h, båda ≈ "
            f"{2 * per * len(exs) * a.epochs / 3600:.1f} h")
        Path("results/sprint4_lora_time.json").write_text(json.dumps({"s_per_example": per, "n_examples": len(exs)}),
                                                          encoding="utf-8")
        return
    train(meas, mods, proj, exs, a.cond, epochs=a.epochs, log=log)
    for m in mods:
        m.enabled = False
    ck1 = weights_checksum(meas.model)
    for m in mods:
        m.enabled = True
    Path("models").mkdir(exist_ok=True)
    torch.save({**lora_state(mods, proj), "cond": a.cond, "epochs": a.epochs, "n_examples": len(exs),
                "base_weights_unchanged": ck0 == ck1}, f"models/report_{a.cond}.pt")
    log(f"klart {a.cond}. basvikter orörda: {ck0 == ck1}")


if __name__ == "__main__":
    main()
