"""Sprint 4 steg 1–2: uppgift v4 (två delfrågor) – svar och attention per svarstoken.

Plan: projektet claude/nexis_v3_sprint4_plan.md (godkänd av Anders 26 nov 08:42, "då går vi vidare med s4 baserat
på dina förslag"). Samma mätning som extract_tokens.py (lager × huvud × 7 textdelar), plus var del 2 börjar i svaret.

    python extract_v4.py --split base4    # frö 9000, 100 texter, träningsämnen (steg 1: förutsättning F1, underkänd)
    python extract_v4.py --split base4b   # frö 9001, 100 nya texter, träningsämnen (F1 omdefinierad, innehållsord)
    python extract_v4.py --split train4   # frö 9100, 1500 texter, träningsämnen
    python extract_v4.py --split test4    # frö 9200, 300 texter, testämnen
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
from nexis3.textgen import SYSTEM_PROMPT, answer_correct_v4, make_items_v4, split_answer_v4  # noqa: E402
from extract_tokens import PARTS, Q_LAYER, token_attention  # noqa: E402

SPLITS = {"base4": (9000, 100, "train"), "base4b": (9001, 100, "train"), "train4": (9100, 1500, "train"), "test4": (9200, 300, "test")}
CHUNK = 50
MAX_NEW = 48


def part2_start(tok, ans_ids, answer):
    """Index för första svarstoken som ligger i del 2 (eller len om del 2 saknas)."""
    p1, p2 = split_answer_v4(answer)
    if not p2:
        return len(ans_ids)
    for k in range(len(ans_ids) + 1):
        if len(tok.decode(ans_ids[:k], skip_special_tokens=True)) > len(p1):
            return max(0, k - 1)
    return len(ans_ids)


def content_rows(tokens, value, start, stop):
    """Index (inom [start, stop)) för svarstoken som täcker svarsordet: första förekomsten av ordet (skiftlägesokänsligt)
    i den sammanfogade texten för token start..stop-1; token vars teckenintervall överlappar den räknas.
    Rad i i att väljer token i, så index = tokenindex."""
    stop = min(stop, len(tokens))
    spans, pos = [], 0
    for t in tokens[start:stop]:
        spans.append((pos, pos + len(t)))
        pos += len(t)
    text = "".join(tokens[start:stop]).lower()
    j = text.find(value.lower())
    if j < 0:
        return []
    e = j + len(value)
    return [start + k for k, (a, b) in enumerate(spans) if a < e and b > j]


def run_chunk(meas, items):
    recs = []
    for it in items:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": it.prompt}]
        prompt = meas.render(msgs)
        ids, offs = meas.encode(prompt)
        ans_ids = meas.generate(ids, MAX_NEW)
        answer = meas.tok.decode(ans_ids, skip_special_tokens=True)
        segs = meas.locate_given(prompt, it.sentences + [it.question])
        seg_ids = token_segment_ids(offs, segs)
        att, q = token_attention(meas, ids, ans_ids, seg_ids)
        ca, cb = answer_correct_v4(it, answer)
        recs.append({
            "idx": it.idx, "topics": it.topics, "roles": {"target_a": it.target_a, "lure_a": it.lure_a,
                                                          "target_b": it.target_b, "lure_b": it.lure_b},
            "answer": answer, "correct_a": ca, "correct_b": cb, "part2_start": part2_start(meas.tok, ans_ids, answer),
            "att": att, "q": q, "n_answer_tokens": len(ans_ids),
            "answer_a": it.answer_a, "answer_b": it.answer_b,
            "answer_tokens": [meas.tok.decode([t]) for t in ans_ids],
        })
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=list(SPLITS), required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--out", default="results/v4_data")
    a = ap.parse_args()
    seed, n, pool = SPLITS[a.split]
    n = a.n or n
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    items = make_items_v4(n, seed, pool)
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
        ok = sum(r["correct_a"] and r["correct_b"] for r in recs)
        print(f"{f.name}: {len(recs)} texter, båda delarna rätt {ok}, {time.time() - t0:.0f} s", flush=True)
    ck1 = weights_checksum(model)
    print(f"klart. vikter orörda: {ck0 == ck1}", flush=True)
    (out / f"{a.split}_done.txt").write_text(f"n={n} seed={seed} weights_unchanged={ck0 == ck1}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
