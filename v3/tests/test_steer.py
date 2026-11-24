"""Invarianter för styrningen (sprint 2)."""
import torch

import steer_eval
from nexis3.schema import Schema
from nexis3.steer import Steerer
from nexis3.textgen import SYSTEM_PROMPT, make_items_v2

ITEMS = make_items_v2(4, 11)


def _attn(model, ids):
    x = torch.tensor([ids])
    with torch.no_grad():
        return model(x, output_attentions=True, use_cache=False).attentions


def _setup(measurer, it):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": it.prompt}]
    ids, seg_ids = steer_eval.prompt_info(measurer, msgs, it.sentences)
    return msgs, ids, seg_ids


def test_inactive_steerer_changes_nothing(measurer):
    it = ITEMS[0]
    msgs, _, _ = _setup(measurer, it)
    a = measurer.measure(msgs, answer="Rex.", segment_texts=it.sentences).A
    st = Steerer(measurer.model, measurer.layers)
    try:
        b = measurer.measure(msgs, answer="Rex.", segment_texts=it.sentences).A
        assert st.calls == 0
    finally:
        st.remove()
    assert max(abs(x - y) for x, y in zip(a, b)) == 0.0


def test_beta_zero_and_uniform_S_change_nothing(measurer):
    it = ITEMS[1]
    msgs, ids, seg_ids = _setup(measurer, it)
    base = _attn(measurer.model, ids)
    st = Steerer(measurer.model, measurer.layers)
    try:
        for S, beta in (([0.7, 0.1, 0.1, 0.1], 0.0), ([0.25] * 4, 3.0)):
            with st.active(Steerer.make_bias(S, seg_ids, beta), start=len(ids) - 1):
                att = _attn(measurer.model, ids)
            assert all(torch.allclose(x, y, atol=1e-6) for x, y in zip(base, att))
    finally:
        st.remove()


def test_decoupling_S_ignored_without_steering(measurer, monkeypatch):
    it = ITEMS[2]
    st = Steerer(measurer.model, measurer.layers)
    try:
        r1 = steer_eval.run(measurer, st, it, [0.97, 0.01, 0.01, 0.01], 0)
        r2 = steer_eval.run(measurer, st, it, [0.01, 0.01, 0.01, 0.97], 0)
    finally:
        st.remove()
    assert r1["A"] == r2["A"] and r1["answer"] == r2["answer"]


def test_bias_only_selected_layers_rows_and_sentence_keys(measurer):
    it = ITEMS[3]
    msgs, ids, seg_ids = _setup(measurer, it)
    ids2 = ids + measurer.tok("Rex.", add_special_tokens=False)["input_ids"]
    start = len(ids) - 1
    base = _attn(measurer.model, ids2)
    st = Steerer(measurer.model, measurer.layers)
    bias = Steerer.make_bias([0.7, 0.1, 0.1, 0.1], seg_ids, 2.0)
    try:
        with st.active(bias, start=start):
            att = _attn(measurer.model, ids2)
    finally:
        st.remove()
    first = measurer.layers[0]
    for l in range(first):                       # lager före styrningen: helt oförändrade
        assert torch.allclose(base[l], att[l], atol=1e-6)
    for l in measurer.layers:                    # rader före start: oförändrade i alla lager (kausalt)
        assert torch.allclose(base[l][0, :, :start], att[l][0, :, :start], atol=1e-5)
    # första styrda lagret: indata oförändrade, så kvoten styrd/bas per nyckel = exp(bias) upp till normering
    sid = torch.tensor(seg_ids)
    r = att[first][0, :, start, :len(ids)] / base[first][0, :, start, :len(ids)]
    r = r / r[:, (sid < 0).nonzero()[0, 0]][:, None]          # normera mot en nyckel utanför meningarna
    assert torch.allclose(r[:, sid < 0], torch.ones_like(r[:, sid < 0]), atol=1e-4)
    assert torch.allclose(r[:, sid >= 0], torch.exp(bias[sid >= 0])[None].expand_as(r[:, sid >= 0]), rtol=1e-3)
    assert not torch.allclose(base[first][0, :, start], att[first][0, :, start])


def test_forced_steering_moves_A(measurer):
    it = ITEMS[0]
    k = (it.target + 1) % 4
    st = Steerer(measurer.model, measurer.layers)
    try:
        b = steer_eval.run(measurer, st, it, None, 0)
        f = steer_eval.run(measurer, st, it, steer_eval.forced_S(k), 4.0)
    finally:
        st.remove()
    assert f["A"][k] > b["A"][k] + 0.1


def test_schema_S_matches_extract_states(measurer, monkeypatch):
    it = ITEMS[1]
    monkeypatch.setattr(steer_eval, "STATE_LAYERS", [1, 3])
    msgs, ids, seg_ids = _setup(measurer, it)
    d = measurer.model.config.hidden_size
    sch = Schema(2, d, 8, 0.0).eval()
    S = steer_eval.schema_S(measurer, sch, ids, seg_ids)
    measurer.STATE_LAYERS = [1, 3]
    try:
        m = measurer.measure(msgs, answer="x", segment_texts=it.sentences)
    finally:
        measurer.STATE_LAYERS = None
    S2 = sch(m.states["seg"][None], m.states["last"][None])[0].tolist()
    assert max(abs(x - y) for x, y in zip(S, S2)) < 1e-3 and abs(sum(S) - 1) < 1e-5


def test_value_of_and_forced_S():
    it = ITEMS[0]
    assert steer_eval.value_of(it.sentences[it.target]) == it.answer
    assert steer_eval.value_of(it.sentences[it.lure]) is None
    S = steer_eval.forced_S(2)
    assert abs(sum(S) - 1) < 1e-9 and S[2] == steer_eval.FORCED_MASS
