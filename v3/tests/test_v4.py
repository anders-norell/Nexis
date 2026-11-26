import torch

import extract_v4 as ev
from nexis3.textgen import answer_correct_v4, make_items_v4, split_answer_v4
from nexis3.topics_v3 import TEST_TOPICS, TRAIN_TOPICS


def test_items_v4_roles_and_topics():
    for split, pool in (("train", TRAIN_TOPICS), ("test", TEST_TOPICS)):
        for it in make_items_v4(60, 9000, split):
            roles = [it.target_a, it.lure_a, it.target_b, it.lure_b]
            assert sorted(roles) == [0, 1, 2, 3]
            assert {t.replace("-lure", "") for t in it.topics} <= pool
            assert it.answer_a.lower() in it.sentences[it.target_a].lower()
            assert it.answer_b.lower() in it.sentences[it.target_b].lower()
            assert it.prompt.index("Question 1") < it.prompt.index("Question 2")


def test_items_v4_deterministic_and_positions_balanced():
    a = make_items_v4(400, 9100, "train")
    assert [i.prompt for i in a[:5]] == [i.prompt for i in make_items_v4(5, 9100, "train")]
    from collections import Counter
    c = Counter(i.target_a for i in a)
    assert min(c.values()) > 60


def test_split_and_correct():
    it = make_items_v4(1, 9000, "train")[0]
    ans = f"1) {it.answer_a} 2) {it.answer_b}"
    assert answer_correct_v4(it, ans) == (True, True)
    assert answer_correct_v4(it, f"1) {it.answer_b} 2) {it.answer_a}") == (False, False)
    assert split_answer_v4("1) x") == ("1) x", "")


def test_chunk_records(measurer):
    recs = ev.run_chunk(measurer, make_items_v4(2, 9000, "train"))
    for r in recs:
        T = max(1, r["n_answer_tokens"])
        assert r["att"].shape[0] == T and 0 <= r["part2_start"] <= r["n_answer_tokens"]
        assert torch.allclose(r["att"].float().sum(-1), torch.ones(r["att"].shape[:3]), atol=2e-2)


def test_splits_distinct():
    seeds = [s for s, _, _ in ev.SPLITS.values()]
    assert len(set(seeds)) == 3 and not set(seeds) & {5000, 6000, 7000, 8000}


def test_part2_start(measurer):
    tok = measurer.tok
    ans = "1) red bike 2) blue cake"
    ids = tok(ans, add_special_tokens=False)["input_ids"]
    k = ev.part2_start(tok, ids, ans)
    assert tok.decode(ids[:k]).strip().endswith("bike") or "2" not in tok.decode(ids[:k])
    assert "2" in tok.decode(ids[k:k + 2])


def test_content_rows():
    toks = ["1", ")", " T", "ulips", " ", "2", ")", " Car", "rot"]
    assert ev.content_rows(toks, "tulips", 0, 5) == [2, 3]
    assert ev.content_rows(toks, "carrot", 5, 9) == [7, 8]
    assert ev.content_rows(toks, "carrot", 0, 5) == []


def test_records_have_tokens(measurer):
    r = ev.run_chunk(measurer, make_items_v4(1, 9001, "train"))[0]
    assert len(r["answer_tokens"]) == r["n_answer_tokens"] and r["answer_a"] and r["answer_b"]
