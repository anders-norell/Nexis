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
    raw, first, allmass = [0.0] * 4, [0.0] * 4, 0.0
    # rad q förutsäger token q+1: från sista prompttoken till näst sista svarstoken
    qs = range(len(ids) - 1, len(ids) + len(ans) - 1)
    for l in measurer.layers:
        H = att[l].shape[1]
        for h in range(H):
            for q in qs:
                for k in range(q + 1):
                    w = float(att[l][0, h, q, k])
                    allmass += w
                    if k < len(seg) and seg[k] >= 0:
                        raw[seg[k]] += w
                        if q == len(ids) - 1:
                            first[seg[k]] += w
    ref = [r / sum(raw) for r in raw]
    assert max(abs(a - b) for a, b in zip(ref, m.A)) < 1e-5
    assert abs(sum(raw) / allmass - m.mass_in_segments) < 1e-5
    assert max(abs(a - b) for a, b in zip([f / sum(first) for f in first], m.A_first)) < 1e-5
    ntok = [seg.count(k) for k in range(4)]
    mean = [r / n for r, n in zip(raw, ntok)]
    assert max(abs(a - b) for a, b in zip([x / sum(mean) for x in mean], m.A_mean)) < 1e-5


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


CHAT_CASES = [[{"role": "user", "content": "Hej. Jag heter Anders. Vad heter du?"}]] + [
    [{"role": "user", "content": it.prompt}] for it in ITEMS]


@pytest.mark.parametrize("msgs", CHAT_CASES, ids=lambda m: m[0]["content"][:20])
def test_I2_tokens_map_to_sentences(measurer, msgs):
    prompt = measurer.render(msgs)
    segs, _ = measurer.locate_segments(prompt, msgs)
    ids, offs = measurer.encode(prompt)
    sid = token_segment_ids(offs, segs)
    for k, s in enumerate(segs):
        toks = [i for i, x in zip(ids, sid) if x == k]
        assert measurer.tok.decode(toks).strip() == s.text
    # tokens utanför meningarna (mallmarkörer) får -1
    assert sid.count(-1) > 0


def test_I2_not_fooled_by_default_system_prompt(measurer):
    """Användartext som också finns i mallens automatiska systemprompt."""
    prompt0 = measurer.render([{"role": "user", "content": "x"}])
    sys_sentence = split_sentences(prompt0.split("\n")[1])[0].text  # första meningen i standardsystemprompten
    msgs = [{"role": "user", "content": sys_sentence}]
    prompt = measurer.render(msgs)
    segs, _ = measurer.locate_segments(prompt, msgs)
    assert len(segs) == 1 and segs[0].start == prompt.rfind(sys_sentence) != prompt.find(sys_sentence)


def test_I2_duplicate_sentence_located_in_order(measurer):
    s = "My dog is called Rex."
    msgs = [{"role": "user", "content": f"{s}\nErik plays the piano.\n{s}"}]
    prompt = measurer.render(msgs)
    segs = measurer.locate_given(prompt, [s, "Erik plays the piano.", s])
    assert segs[0].start < segs[1].start < segs[2].start
    segs2, _ = measurer.locate_segments(prompt, msgs)
    assert [x.start for x in segs2] == [x.start for x in segs]


def test_I2_crossing_token_goes_to_first_sentence():
    segs = [Segment("Ab.", 0, 3), Segment("Cd.", 4, 7)]
    assert token_segment_ids([(2, 5), (5, 7), (3, 4), (0, 0)], segs) == [0, 1, -1, -1]


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
    n = len(list(model.parameters()))
    for j in (0, n // 2, n - 1):  # första, mitten, sista parametern
        m2 = copy.deepcopy(model)
        with torch.no_grad():
            list(m2.parameters())[j].view(-1)[-1] += 1e-3
        with pytest.raises(AssertionError):
            check_weights_unchanged(ck0, weights_checksum(m2))
    check_weights_unchanged(ck0, weights_checksum(copy.deepcopy(model)))


# ---------------- positiv kontroll (bara riktiga modellen) ---------------------

def test_A_positive_control_copy(measurer):
    """Svaret är målmeningens unika värde; argmax A ska då peka på den meningen.
    Gräns 36/40 [Claudes förslag]: en grov rimlighetskontroll, inte ett mått. Svag eftersom frågan
    delar ord med målmeningen (granskningen, punkt 2)."""
    if measurer.model.config.num_hidden_layers < 12:
        pytest.skip("kräver den riktiga modellen (NEXIS_REAL_MODEL)")
    items = make_items(40, seed=555)
    hits = 0
    for it in items:
        m = measurer.measure([{"role": "user", "content": it.prompt}], answer=it.answer + ".",
                             segment_texts=it.sentences)
        hits += max(range(4), key=lambda k: m.A[k]) == it.target
    assert hits >= 36, f"argmax A = målet i bara {hits}/40"


def test_I2_short_and_repeated_messages(measurer):
    """Korta texter som finns i mallens markörer, och text som upprepas i senare meddelanden."""
    msgs = [{"role": "user", "content": "e"}, {"role": "assistant", "content": "end"},
            {"role": "user", "content": "Rex."}, {"role": "assistant", "content": "m"},
            {"role": "user", "content": "Rex."}]
    prompt = measurer.render(msgs)
    segs, owner = measurer.locate_segments(prompt, msgs)
    assert [x.text for x in segs] == ["e", "end", "Rex.", "m", "Rex."] and owner == [0, 1, 2, 3, 4]
    for x, m in zip(segs, msgs):
        before = prompt[: x.start]
        assert before.endswith("<|im_start|>" + m["role"] + "\n"), before[-30:]


def test_too_long_conversation_rejected(measurer):
    old = measurer.MAX_TOKENS
    measurer.MAX_TOKENS = 20
    try:
        with pytest.raises(ValueError):
            measurer.measure([{"role": "user", "content": ITEMS[0].prompt}], answer="Rex.")
    finally:
        measurer.MAX_TOKENS = old


def test_generate_is_greedy_without_repetition_penalty(measurer, monkeypatch):
    seen = {}
    orig = measurer.model.generate

    def spy(*a, **k):
        seen.update(k)
        return orig(*a, **k)
    monkeypatch.setattr(measurer.model, "generate", spy)
    measurer.generate(measurer.encode(measurer.render([{"role": "user", "content": "Hej."}]))[0], 2)
    assert seen["do_sample"] is False and seen["repetition_penalty"] == 1.0
