from nexis3.textgen import make_twins_v3, make_items_v3
from nexis3.topics_v3 import TEST_TOPICS


def test_twins_structure():
    it = make_twins_v3(50, 7000)
    assert len(it) == 100
    for x, y in zip(it[::2], it[1::2]):
        assert x.idx == y.idx and x.twin == 0 and y.twin == 1
        assert x.target == y.target and x.lure == y.lure
        bx = {t.replace("-lure", "") for t in x.topics}
        by = {t.replace("-lure", "") for t in y.topics}
        assert not (bx & by) and bx | by <= TEST_TOPICS


def test_twins_deterministic_and_v3_unchanged():
    assert [i.prompt for i in make_twins_v3(5, 7000)] == [i.prompt for i in make_twins_v3(5, 7000)]
    a = make_items_v3(20, 6000, "test")
    b = make_items_v3(20, 6000, "test")
    assert [i.prompt for i in a] == [i.prompt for i in b]
