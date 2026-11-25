"""Attention per svarstoken: summerar till 1 per rad, huvud och lager; mening/fråga/övrigt/svar rätt fördelat."""
import torch

import extract_tokens as et
from nexis3.textgen import make_items_v3


def test_token_attention_sums_and_shape(measurer):
    recs = et.run_chunk(measurer, make_items_v3(2, 11, "test"))
    cfg = measurer.model.config
    for r in recs:
        att = r["att"].float()
        T = max(1, r["n_answer_tokens"])
        assert att.shape == (T, cfg.num_hidden_layers, cfg.num_attention_heads, len(et.PARTS))
        assert torch.allclose(att.sum(-1), torch.ones_like(att.sum(-1)), atol=2e-2)
        assert float(att[0, :, :, 6].abs().max()) < 1e-3   # första raden (sista prompttoken) ser inget svar
        assert float(att[..., 4].sum()) > 0 and float(att[..., :4].sum()) > 0
        assert r["q"].shape == (cfg.hidden_size,)


def test_sentence_parts_match_A(measurer):
    """Mening 1–4 i lagren för A, normerat, ska ge samma A som den låsta mätningen."""
    r = et.run_chunk(measurer, make_items_v3(1, 12, "test"))[0]
    L = measurer.layers
    s = r["att"].float()[:, L, :, :4].mean(dim=(0, 1, 2))
    assert torch.allclose(s / s.sum(), r["A"].float(), atol=5e-3)


def test_splits_and_topics():
    from nexis3.topics_v3 import TEST_TOPICS
    assert et.SPLITS["tok_train"][0] == 5000 and et.SPLITS["tok_test"][0] not in (5000, 6000, 7000)
    for it in make_items_v3(50, et.SPLITS["tok_test"][0], "test"):
        assert {t.replace("-lure", "") for t in it.topics} <= TEST_TOPICS


def test_answer_source():
    s = ["The boat is red.", "The ferry leaves.", "Chess on Monday.", "Bike is blue."]
    assert et.answer_source(s, "Red boat") == 0 and et.answer_source(s, "xyz") == -1
