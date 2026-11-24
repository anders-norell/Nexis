"""Testtexter version 2: ingen ordöverlapp mellan fråga och mål, lockmening delar ord med frågan."""
import re

from nexis3.textgen import TOPICS_V2, content_words, make_items_v2, overlap_choice, answer_correct
from nexis3.segment import split_sentences


def test_topic_bank_overlap_rules():
    for name, s, q, lure, vals in TOPICS_V2:
        for v in vals:
            assert not (content_words(q) & content_words(s.format(v=v))), (name, v)
        assert content_words(q) & content_words(lure), name
        for v in vals:
            assert not re.search(r"\b" + re.escape(v.lower()) + r"\b", lure.lower()), (name, v)


def test_items_v2_properties():
    items = make_items_v2(400, seed=0)
    assert sorted(i.target for i in items) == sorted([0, 1, 2, 3] * 100)
    hits = sum(overlap_choice(i.question, i.sentences) == i.target for i in items)
    assert hits / len(items) <= 0.30, hits          # förregistrerad gräns i planen
    lure_hits = sum(overlap_choice(i.question, i.sentences) == i.lure for i in items)
    assert lure_hits / len(items) >= 0.9, lure_hits
    for it in items:
        assert it.lure != it.target and it.topics[it.lure] == it.topics[it.target] + "-lure"
        assert len(set(it.sentences)) == 4
        assert sum(bool(re.search(r"\b" + re.escape(it.answer.lower()) + r"\b", s.lower())) for s in it.sentences) == 1
        body = it.prompt.split("\n\n")[1]
        assert [s.text for s in split_sentences(body)] == [f"{k + 1}. {s}" for k, s in enumerate(it.sentences)]


def test_items_v2_seeded_and_disjoint_seeds():
    assert [i.prompt for i in make_items_v2(20, 3)] == [i.prompt for i in make_items_v2(20, 3)]
    assert [i.prompt for i in make_items_v2(20, 3)] != [i.prompt for i in make_items_v2(20, 4)]


def test_overlap_null_model_detects_v1_leak():
    """Kontroll av nollmodellen: på version 1 pekar ordöverlapp ut målet (läckaget som granskningen fann)."""
    from nexis3.textgen import make_items
    items = make_items(100, 0)
    assert sum(overlap_choice(i.question, i.sentences) == i.target for i in items) >= 95
