"""Invarianter för sprint 1 (v3), var och en med ett test som ska kunna fallera.

I1  Attention per mening summerar till 1 och ändras när texten ändras.
I2  Segmentgränserna stämmer med meningarna (kända exempel + token-koppling).
I3  Basmodellens vikter är orörda (kontrollsumma före/efter).

För varje invariant finns ett mutanttest: kontrollen körs på en medvetet
trasig variant och måste då slå larm.
"""
import copy

import pytest
import torch

from nexis3.attention import Measurer, weights_checksum
from nexis3.invariants import check_boundaries, check_changes, check_sums_to_one, check_weights_unchanged
from nexis3.segment import Segment, split_sentences, token_segment_ids
from nexis3.textgen import make_items

ITEMS = make_items(8, seed=123)

KNOWN = [
    ("Dr. Smith arrived at 3.5 p.m. today. He left! Did she? Yes.",
     ["Dr. Smith arrived at 3.5 p.m. today.", "He left!", "Did she?", "Yes."]),
    ("Jag heter Anders. Vad heter du?", ["Jag heter Anders.", "Vad heter du?"]),
    ("Vi köpte t.ex. äpplen och päron. Sedan gick vi hem.",
     ["Vi köpte t.ex. äpplen och päron.", "Sedan gick vi hem."]),
    ("Read these four sentences.\n\n1. My dog is called Rex.\n2. Erik plays the piano.\n\nQuestion: Who? Answer.",
     ["Read these four sentences.", "1. My dog is called Rex.", "2. Erik plays the piano.", "Question: Who?", "Answer."]),
    ('He said "Stop." Then he left.', ['He said "Stop."', "Then he left."]),
    ("no punctuation at all", ["no punctuation at all"]),
    ("Ends with spaces.   ", ["Ends with spaces."]),
    ("Line one\nLine two", ["Line one", "Line two"]),
    ("The price is 3.50 kr. It rose.", ["The price is 3.50 kr.", "It rose."]),
]


# ---------------- I1 ------------------------------------------------------------

def _measure(meas, item, sentences=None, answer=None):
    sents = sentences or item.sentences
    body = "\n".join(f"{k + 1}. {s}" for k, s in enumerate(sents))
    prompt = item.prompt.split("\n\n")[0] + "\n\n" + body + "\n\n" + item.prompt.split("\n\n")[-1]
    return meas.measure([{"role": "user", "content": prompt}], answer=answer, max_new_tokens=8, segment_texts=sents)


@pytest.mark.parametrize("item", ITEMS[:4], ids=lambda it: f"item{it.idx}")
def test_I1_sums_to_one(measurer, item):
    m = _measure(measurer, item)
    check_sums_to_one(m.A)
    assert len(m.A) == 4
    assert 0 < m.mass_in_segments <= 1 + 1e-6
    for row in m.per_layer:
        check_sums_to_one(row)


def test_I1_sums_to_one_free_chat(measurer):
    msgs = [{"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hej. Jag heter Anders. Vad heter du?"},
            {"role": "assistant", "content": "Jag heter Nexis."},
            {"role": "user", "content": "Bra. Berätta något."}]
    m = measurer.measure(msgs, max_new_tokens=6)
    assert [s.text for s in m.segments] == ["Hej.", "Jag heter Anders.", "Vad heter du?", "Jag heter Nexis.",
                                            "Bra.", "Berätta något."]
    check_sums_to_one(m.A)


@pytest.mark.parametrize("item", ITEMS[:4], ids=lambda it: f"item{it.idx}")
def test_I1_changes_when_text_changes(measurer, item):
    fixed = "Here is a fixed answer."
    m1 = _measure(measurer, item, answer=fixed)
    other = make_items(8, seed=999)[item.idx]
    sents = list(item.sentences)
    sents[1] = other.sentences[1] if other.sentences[1] != sents[1] else other.sentences[2]
    m2 = _measure(measurer, item, sentences=sents, answer=fixed)
    check_changes(m1.A, m2.A)


def test_I1_deterministic(measurer):
    a = _measure(measurer, ITEMS[0], answer="Rex.").A
    b = _measure(measurer, ITEMS[0], answer="Rex.").A
    assert max(abs(x - y) for x, y in zip(a, b)) < 1e-6


def test_I1_matches_bruteforce(measurer):
    """Oberoende loop-implementation av definitionen i attention.py."""
    it = ITEMS[1]
    m = _measure(measurer, it, answer="An answer.")
    ids, offs = measurer.encode(m.prompt)
    seg = token_segment_ids(offs, m.segments)
    ans = measurer.tok("An answer.", add_special_tokens=False)["input_ids"]
    full = torch.tensor([ids + ans])
    with torch.no_grad():
        att = measurer.model(full, output_attentions=True, use_cache=False).attentions
    raw = [0.0] * 4
    cnt = 0
    for l in measurer.layers:
        H = att[l].shape[1]
        for h in range(H):
            for q in range(len(ids), len(ids) + len(ans)):
                cnt += 1
                for k in range(len(ids)):
                    if seg[k] >= 0:
                        raw[seg[k]] += float(att[l][0, h, q, k])
    ref = [r / sum(raw) for r in raw]
    assert max(abs(a - b) for a, b in zip(ref, m.A)) < 1e-5


def test_I1_mutants_fail(measurer):
    m = _measure(measurer, ITEMS[0], answer="Rex.")
    unnormalized = [a * 1.3 for a in m.A]
    with pytest.raises(AssertionError):
        check_sums_to_one(unnormalized)
    constant = [0.25] * 4  # en mätning som inte bryr sig om texten
    with pytest.raises(AssertionError):
        check_changes(constant, constant)


# ---------------- I2 ------------------------------------------------------------

@pytest.mark.parametrize("text,expected", KNOWN)
def test_I2_known_examples(text, expected):
    check_boundaries(text, split_sentences(text), expected)


@pytest.mark.parametrize("item", ITEMS, ids=lambda it: f"item{it.idx}")
def test_I2_generated_items(item):
    body = item.prompt.split("\n\n")[1]
    got = split_sentences(body)
    check_boundaries(body, got, [f"{k + 1}. {s}" for k, s in enumerate(item.sentences)])


def test_I2_tokens_map_to_sentences(measurer):
    msgs = [{"role": "user", "content": "Hej. Jag heter Anders. Vad heter du?"}]
    prompt = measurer.render(msgs)
    segs, _ = measurer.locate_segments(prompt, msgs)
    ids, offs = measurer.encode(prompt)
    sid = token_segment_ids(offs, segs)
    for k, s in enumerate(segs):
        toks = [i for i, x in zip(ids, sid) if x == k]
        assert measurer.tok.decode(toks).strip() == s.text
    # tokens utanför meningarna (mallmarkörer) får -1
    assert sid.count(-1) > 0


def test_I2_given_segments_located(measurer):
    it = ITEMS[2]
    prompt = measurer.render([{"role": "user", "content": it.prompt}])
    segs = measurer.locate_given(prompt, it.sentences)
    check_boundaries(prompt, segs, it.sentences)


def test_I2_mutant_fails():
    text, expected = KNOWN[0]
    segs = split_sentences(text)
    shifted = [Segment(s.text, s.start + 1, s.end + 1) for s in segs]
    with pytest.raises(AssertionError):
        check_boundaries(text, shifted, expected)
    naive = text.split(". ")  # naiv delning bryter vid "Dr."
    with pytest.raises(AssertionError):
        check_boundaries(text, segs, naive)


# ---------------- I3 ------------------------------------------------------------

def test_I3_weights_unchanged_after_use(model_tok):
    model, tok = model_tok
    ck0 = weights_checksum(model)
    meas = Measurer(model, tok)
    for it in ITEMS[:2]:
        meas.measure([{"role": "user", "content": it.prompt}], max_new_tokens=6, segment_texts=it.sentences)
    check_weights_unchanged(ck0, weights_checksum(model))
    assert not any(p.requires_grad for p in model.parameters())
    assert not model.training


def test_I3_mutant_fails(model_tok):
    model, _ = model_tok
    if sum(p.numel() for p in model.parameters()) > 50_000_000:
        pytest.skip("mutanttestet kopierar modellen; görs på testmodellen")
    ck0 = weights_checksum(model)
    m2 = copy.deepcopy(model)
    with torch.no_grad():
        next(m2.parameters()).view(-1)[0] += 1e-3
    with pytest.raises(AssertionError):
        check_weights_unchanged(ck0, weights_checksum(m2))
