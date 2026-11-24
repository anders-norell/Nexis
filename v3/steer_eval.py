"""Sprint 2 steg 4–5: styrning med schemat på den riktiga modellen.

    python steer_eval.py beta   # välj beta på 40 valideringstexter (träningsfröets sista 200, ej använda för vikterna)
    python steer_eval.py test   # 300 testtexter: tre villkor per text

Villkor per text:
  base    beta = 0 (ingen styrning)
  own     beta*, S = schemats egen förutsägelse ur modellens tillstånd innan svaret
  forced  beta*, S = nästan all massa (0,97) på en slumpvis vald icke-målmening (frö 4000)
Mäts: A (lager 9–17, låst mått), A_late (lager 18–27, ej direkt styrda), svaret, om svaret är rätt,
om svaret innehåller den tvingade meningens värde. Rader sparas löpande; avbrutet arbete återupptas.
"""
from __future__ import annotations

import argparse
import json
import re
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from nexis3.attention import DEFAULT_MODEL, Measurer, load_model, weights_checksum  # noqa: E402
from nexis3.schema import Schema  # noqa: E402
from nexis3.segment import token_segment_ids  # noqa: E402
from nexis3.steer import Steerer  # noqa: E402
from nexis3.textgen import _NUM, SYSTEM_PROMPT, TOPICS_V2, answer_correct, make_items_v2  # noqa: E402

STATE_LAYERS = [9, 13, 17]
LATE_LAYERS = list(range(18, 28))
BETAS = [0.5, 1.0, 2.0, 4.0]
FORCED_MASS = 0.97


def load_schema(path):
    ck = torch.load(path, weights_only=False)
    hp = ck["hyperparams"]
    d = ck["state_dict"]["mu"].shape[0] // len(STATE_LAYERS)
    m = Schema(len(STATE_LAYERS), d, hp["h"], hp["dropout"])
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m


def prompt_info(meas, msgs, sentences):
    prompt = meas.render(msgs)
    segs = meas.locate_given(prompt, sentences)
    ids, offs = meas.encode(prompt)
    return ids, token_segment_ids(offs, segs)


@torch.no_grad()
def schema_S(meas, schema, ids, seg_ids, n_seg=4):
    x = torch.tensor([ids])
    out = meas.model(x, attention_mask=torch.ones_like(x), output_hidden_states=True, use_cache=False)
    hs = torch.stack([out.hidden_states[l + 1][0].float() for l in STATE_LAYERS])  # (Ls, n_p, d)
    oh = torch.zeros(len(ids), n_seg)
    sid = torch.tensor(seg_ids)
    m = sid >= 0
    oh[m.nonzero().squeeze(-1), sid[m]] = 1.0
    seg = torch.einsum("lpd,ps->sld", hs, oh) / oh.sum(0).clamp(min=1)[:, None, None]
    last = hs[:, -1]
    return schema(seg.half()[None], last.half()[None])[0].tolist()


def forced_S(k, n=4):
    rest = (1 - FORCED_MASS) / (n - 1)
    return [FORCED_MASS if i == k else rest for i in range(n)]


def value_of(sentence):
    for _, tmpl, _, _, vals in TOPICS_V2:
        pre, post = tmpl.split("{v}")
        if sentence.startswith(pre) and sentence.endswith(post):
            return sentence[len(pre):len(sentence) - len(post)]
    return None  # lockmening: inget värde


def mentions(answer, value):
    """Ordgräns och siffra för talord, som answer_correct."""
    if not value:
        return False
    a, v = answer.lower(), value.lower()
    hit = lambda w: re.search(r"\b" + re.escape(w) + r"\b", a) is not None
    return hit(v) or (v in _NUM and hit(_NUM[v]))


def read_jsonl(f):
    """Läser rader; en avhuggen sista rad (avbruten körning) hoppas över."""
    if not f.exists():
        return []
    out = []
    for line in f.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def run(meas, steerer, it, S, beta):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": it.prompt}]
    ids, seg_ids = prompt_info(meas, msgs, it.sentences)
    if S is None or beta == 0:
        m = meas.measure(msgs, max_new_tokens=24, segment_texts=it.sentences)
    else:
        c0 = steerer.calls
        with steerer.active(Steerer.make_bias(S, seg_ids, beta), start=len(ids) - 1):
            m = meas.measure(msgs, max_new_tokens=24, segment_texts=it.sentences)
        if steerer.calls == c0:
            raise RuntimeError("styrningen anropades aldrig (fel transformers-version?)")
    return {"A": m.A, "A_late": m.A_extra, "answer": m.answer, "correct": answer_correct(it, m.answer)}


def item_rows(meas, steerer, schema, it, beta, forced_k, conds=("base", "own", "perm", "forced"), roll=1):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": it.prompt}]
    ids, seg_ids = prompt_info(meas, msgs, it.sentences)
    S_own = schema_S(meas, schema, ids, seg_ids)
    row = {"idx": it.idx, "target": it.target, "lure": it.lure, "S_own": S_own, "forced_k": forced_k, "beta": beta,
           "roll": roll, "forced_is_filler": value_of(it.sentences[forced_k]) is not None}
    for c in conds:
        if c == "base":
            S = None
        elif c == "own":
            S = S_own
        elif c == "perm":   # kontroll: samma värden i S, flyttade till andra meningar
            S = S_own[-roll:] + S_own[:-roll]
        else:
            S = forced_S(forced_k)
        r = run(meas, steerer, it, S, 0 if c == "base" else beta)
        r["mentions_forced_value"] = mentions(r["answer"], value_of(it.sentences[forced_k]))
        row[c] = r
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["beta", "test"])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--dtype", default="float32")  # samma som schemadatan
    ap.add_argument("--schema", default="models/schema_s.pt")
    ap.add_argument("--n", type=int, default=None)
    a = ap.parse_args()
    out = Path("results")
    out.mkdir(exist_ok=True)
    model, tok = load_model(a.model, a.dtype)
    meas = Measurer(model, tok)
    meas.EXTRA_LAYERS = LATE_LAYERS
    steerer = Steerer(model, meas.layers)
    schema = load_schema(a.schema)
    ck0 = weights_checksum(model)
    rng = random.Random(4000)
    if a.phase == "beta":
        items = make_items_v2(1200, 2000)[1000:1000 + (a.n or 40)]  # valideringsdelen
        f = out / "sprint2_beta.jsonl"
        done = {r["key"] for r in read_jsonl(f)}
        for it in items:
            fk = rng.choice([i for i in range(4) if i != it.target])
            for beta in BETAS:
                key = f"{it.idx}-{beta}"
                if key in done:
                    continue
                t0 = time.time()
                row = item_rows(meas, steerer, schema, it, beta, fk, conds=("own", "forced"))
                row["key"] = key
                with open(f, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"[{it.idx}] beta {beta}: tvingad A {row['forced']['A'][fk]:.2f}, egen rätt {row['own']['correct']}"
                      f"  {time.time() - t0:.0f} s", flush=True)
        fb = out / "sprint2_beta_base.jsonl"
        done_b = {r["idx"] for r in read_jsonl(fb)}
        with open(fb, "a", encoding="utf-8") as fh:
            for it in items:
                if it.idx in done_b:
                    continue
                r = run(meas, steerer, it, None, 0)
                fh.write(json.dumps({"idx": it.idx, "target": it.target, **r}, ensure_ascii=False) + "\n")
                fh.flush()
                print(f"[{it.idx}] bas A mål {r['A'][it.target]:.2f}", flush=True)
    else:
        lock = out / "sprint2_beta_choice.json"
        if not lock.exists():
            sys.exit("Välj beta först (results/sprint2_beta_choice.json saknas).")
        beta = json.loads(lock.read_text(encoding="utf-8"))["beta"]
        items = make_items_v2(300, 3000)[:a.n] if a.n else make_items_v2(300, 3000)
        f = out / "sprint2_steer_test.jsonl"
        done = {r["idx"] for r in read_jsonl(f)}
        for it in items:
            fk = rng.choice([i for i in range(4) if i != it.target])  # dras även för klara texter: samma följd
            roll = rng.choice([1, 2, 3])
            if it.idx in done:
                continue
            t0 = time.time()
            row = item_rows(meas, steerer, schema, it, beta, fk, roll=roll)
            with open(f, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"[{it.idx}] mål {it.target + 1} tvingad {fk + 1}: A på tvingad bas {row['base']['A'][fk]:.2f} "
                  f"-> {row['forced']['A'][fk]:.2f}; rätt bas/egen/tvingad {row['base']['correct']:d}"
                  f"{row['own']['correct']:d}{row['forced']['correct']:d}  {time.time() - t0:.0f} s", flush=True)
    print(f"klart. vikter orörda: {ck0 == weights_checksum(model)}", flush=True)


if __name__ == "__main__":
    main()
