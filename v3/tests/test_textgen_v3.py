"""Testtexter version 3: undanhållna testämnen förekommer aldrig i träningstexter."""
import re

from nexis3.textgen import content_words, make_items_v3, overlap_choice
from nexis3.topics_v3 import TEST_TOPICS, TOPICS_V3, TRAIN_TOPICS


def test_bank_rules():
    assert len(TOPICS_V3) == 40 and len(TEST_TOPICS) == 10 and not (TEST_TOPICS & TRAIN_TOPICS)
    for name, s, q, lure, vals in TOPICS_V3:
        for v in vals:
            assert not (content_words(q) & content_words(s.format(v=v))), (name, v)
            assert not re.search(r"\b" + re.escape(v.lower()) + r"\b", lure.lower()), (name, v)
        assert content_words(q) & content_words(lure), name


def test_split_topics_disjoint():
    tr = make_items_v3(500, 5000, "train")
    te = make_items_v3(300, 6000, "test")
    base = lambda t: t.replace("-lure", "")
    assert all(base(t) in TRAIN_TOPICS for it in tr for t in it.topics)
    assert all(base(t) in TEST_TOPICS for it in te for t in it.topics)
    for it in te:
        for s in it.sentences:  # ingen testmening förekommer i träningsmängden
            assert all(s not in jt.sentences for jt in tr[:50])
    assert sum(overlap_choice(i.question, i.sentences) == i.target for i in te) / len(te) <= 0.30
    assert sorted(i.target for i in te) == sorted([0, 1, 2, 3] * 75)
