"""Schemats indata: bara promptens positioner, så tillstånden är oberoende av svaret."""
import torch

from nexis3.textgen import make_items_v2
import extract


def test_states_independent_of_answer(measurer):
    it = make_items_v2(1, 5)[0]
    msgs = [{"role": "user", "content": it.prompt}]
    measurer.STATE_LAYERS = [1, 3]
    try:
        a = measurer.measure(msgs, answer="Rex.", segment_texts=it.sentences)
        b = measurer.measure(msgs, answer="Something completely different here.", segment_texts=it.sentences)
    finally:
        measurer.STATE_LAYERS = None
    assert a.states["seg"].shape == (4, 2, measurer.model.config.hidden_size)
    assert torch.allclose(a.states["seg"].float(), b.states["seg"].float(), atol=1e-3)
    assert torch.allclose(a.states["last"].float(), b.states["last"].float(), atol=1e-3)
    assert not torch.allclose(a.states["seg"][0].float(), a.states["seg"][1].float())


def test_states_off_by_default_and_A_unchanged(measurer):
    it = make_items_v2(1, 6)[0]
    msgs = [{"role": "user", "content": it.prompt}]
    a = measurer.measure(msgs, answer="Rex.", segment_texts=it.sentences)
    assert a.states is None
    measurer.STATE_LAYERS = [2]
    try:
        b = measurer.measure(msgs, answer="Rex.", segment_texts=it.sentences)
    finally:
        measurer.STATE_LAYERS = None
    assert max(abs(x - y) for x, y in zip(a.A, b.A)) < 1e-6


def test_chunk_records(measurer):
    measurer.STATE_LAYERS = [1]
    try:
        recs = extract.run_chunk(measurer, make_items_v2(2, 7))
    finally:
        measurer.STATE_LAYERS = None
    assert len(recs) == 2 and abs(float(recs[0]["A"].sum()) - 1) < 1e-5 and recs[0]["seg"].shape[0] == 4


def test_splits_disjoint_from_baseline_seed():
    seeds = [s for s, _ in extract.SPLITS.values()]
    assert len(set(seeds)) == len(seeds) and 1000 not in seeds
    tr = {i.prompt for i in make_items_v2(200, extract.SPLITS["train"][0])}
    te = {i.prompt for i in make_items_v2(200, extract.SPLITS["test"][0])}
    assert len(tr & te) <= 2  # samma text kan uppstå av slump, men inte systematiskt
