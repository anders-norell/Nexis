"""Sprint 3b steg 1: attention per svarstoken, per lager och huvud, summerad per textdel.

Plan: projektet claude/nexis_v3_sprint3b_plan.md (godkänd av Anders 25 nov 23:45, "vi kör på det").
För varje text besvaras uppgiften (girigt, som förut). Sedan en framåtkörning över prompt + svar. För varje rad som
väljer en svarstoken (sista prompttoken t.o.m. näst sista svarstoken) sparas
  att[t, l, h, p] = summa av attention från rad t, lager l, huvud h till textdel p
textdelar p: mening 1-4, frågan, övrigt i prompten (systemprompt, mall, instruktion), svaret hittills.
Dessutom: frågans läge q = sista prompttokens tillstånd i lager Q_LAYER (indata till G), A som förut (jämförelse),
rätt/fel (Y) och vilken mening svaret hämtas ur (bi-Y).

    python extract_tokens.py --split tok_test    # frö 8000, 300 texter, testämnen (nya texter)
    python extract_tokens.py --split tok_train   # frö 5000, 1500 texter, träningsämnen (samma texter som train3)
Skrivs i bitar om 50 till results/token_data/<split>_NNN.pt; avbrutet arbete återupptas.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from nexis3.attention import DEFAULT_MODEL, Measurer, load_model, model_info, weights_checksum  # noqa: E402
from nexis3.segment import token_segment_ids  # noqa: E402
from nexis3.textgen import SYSTEM_PROMPT, answer_correct, content_words, make_items_v3  # noqa: E402

SPLITS = {"tok_train": (5000, 1500, "train"), "tok_test": (8000, 300, "test")}
PARTS = ["s1", "s2", "s3", "s4", "question", "rest", "answer"]
Q_LAYER = 13
CHUNK = 50


def answer_source(sentences, answer):
    """Meningen som delar flest innehållsord med svaret; -1 om ingen."""
    a = content_words(answer)
    sc = [len(a & content_words(s)) for s in sentences]
    return max(range(len(sentences)), key=lambda k: sc[k]) if max(sc) > 0 else -1


@torch.no_grad()
def token_attention(meas: Measurer, prompt_ids, answer_ids, seg_ids):
    """att (T, L, H, 7) float16, q (d,) float16. T = antal svarstoken (minst 1)."""
    full = list(prompt_ids) + list(answer_ids)
    if len(full) > meas.MAX_TOKENS:
        raise ValueError("för lång")
    x = torch.tensor([full])
    out = meas.model(x, attention_mask=torch.ones_like(x), output_attentions=True, use_cache=False,
                     output_hidden_states=True)
    n_p = len(prompt_ids)
    rows = list(range(n_p - 1, len(full) - 1)) if answer_ids else [n_p - 1]
    part = torch.full((len(full),), 5, dtype=torch.long)          # övrigt
    sid = torch.tensor(seg_ids, dtype=torch.long)
    part[:n_p] = torch.where(sid >= 0, sid, torch.full_like(sid, 5))
    part[n_p:] = 6                                                 # svaret
    onehot = torch.nn.functional.one_hot(part, len(PARTS)).float()  # (L, 7)
    att = torch.stack([out.attentions[l][0].float()[:, rows, :] @ onehot
                       for l in range(len(out.attentions))])      # (layers, heads, T, 7)
    ql = min(Q_LAYER, len(out.attentions) - 1)                     # testmodellen har färre lager
    q = out.hidden_states[ql + 1][0, n_p - 1].float()
    return att.permute(2, 0, 1, 3).half(), q.half()


def run_chunk(meas, items):
    recs = []
    for it in items:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": it.prompt}]
        m = meas.measure(msgs, max_new_tokens=24, segment_texts=it.sentences)
        prompt = meas.render(msgs)
        ids, offs = meas.encode(prompt)
        segs = meas.locate_given(prompt, it.sentences + [it.question])
        seg_ids = token_segment_ids(offs, segs)                    # 0-3 meningar, 4 frågan, -1 övrigt
        ans_ids = meas.tok(m.answer, add_special_tokens=False)["input_ids"]
        att, q = token_attention(meas, ids, ans_ids, seg_ids)
        recs.append({
            "idx": it.idx, "target": it.target, "lure": it.lure, "topics": it.topics,
            "answer": m.answer, "task_correct": answer_correct(it, m.answer),
            "answer_source": answer_source(it.sentences, m.answer),
            "A": torch.tensor(m.A), "att": att, "q": q, "n_answer_tokens": len(ans_ids),
        })
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=list(SPLITS), required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--out", default="results/token_data")
    a = ap.parse_args()
    seed, n, pool = SPLITS[a.split]
    n = a.n or n
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    items = make_items_v3(n, seed, pool)
    model, tok = load_model(a.model, a.dtype)
    meas = Measurer(model, tok)
    ck0 = weights_checksum(model)
    for c0 in range(0, n, CHUNK):
        f = out / f"{a.split}_{c0 // CHUNK:03d}.pt"
        if f.exists():
            continue
        t0 = time.time()
        recs = run_chunk(meas, items[c0:c0 + CHUNK])
        torch.save({"split": a.split, "seed": seed, "parts": PARTS, "q_layer": Q_LAYER,
                    "model_info": model_info(model, tok), "records": recs}, f)
        ok = sum(r["task_correct"] for r in recs)
        print(f"{f.name}: {len(recs)} texter, uppgift rätt {ok}, {time.time() - t0:.0f} s", flush=True)
    ck1 = weights_checksum(model)
    print(f"klart. vikter orörda: {ck0 == ck1}", flush=True)
    (out / f"{a.split}_done.txt").write_text(f"n={n} seed={seed} weights_unchanged={ck0 == ck1}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
