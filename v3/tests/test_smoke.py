"""Röktest av baslinjeskriptet och chattservern på testmodellen."""
import json
import re
import threading
import urllib.request
from http.server import ThreadingHTTPServer

from nexis3.attention import weights_checksum
from nexis3.textgen import make_items, parse_report, answer_correct

import baseline
import chat


def test_textgen_balanced_and_seeded():
    a, b = make_items(100, 0), make_items(100, 0)
    assert [x.prompt for x in a] == [x.prompt for x in b]
    assert sorted(x.target for x in a) == sorted([0, 1, 2, 3] * 25)
    for it in a:
        assert len(set(it.topics)) == 4
        assert it.question in it.prompt
        assert sum(bool(re.search(r"\b" + re.escape(it.answer.lower()) + r"\b", s.lower())) for s in it.sentences) == 1
    assert make_items(10, 1)[0].prompt != a[0].prompt


def test_parse_report_and_answer():
    assert parse_report("3") == 2 and parse_report("Sentence 1.") == 0 and parse_report("none") is None
    it = make_items(1, 0)[0]
    assert answer_correct(it, f"It is {it.answer.upper()}.")


def test_baseline_pipeline(measurer):
    rows = baseline.run_items(measurer, make_items(4, 7), log=lambda *a: None)
    s = baseline.summarize(rows)
    assert s["A_points_to_target"]["n"] == 4
    for r in rows:
        assert abs(sum(r["A"]) - 1) < 1e-5 and abs(sum(r["A_report"]) - 1) < 1e-5
    json.dumps(s)


def test_chat_server(measurer):
    info = {"model": "tiny", "layers": measurer.layers, "n_layers": 6, "dtype": "f32",
            "weights_sha256": weights_checksum(measurer.model)}
    srv = ThreadingHTTPServer(("127.0.0.1", 0), chat.make_handler(measurer, info))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        assert b"Nexis v3" in urllib.request.urlopen(url + "/").read()
        body = json.dumps({"messages": [{"role": "user", "content": "Hej. Vad heter du?"}], "max_new_tokens": 5,
                           "system": "Var kort."}).encode()
        r = json.loads(urllib.request.urlopen(urllib.request.Request(url + "/api/chat", body,
                                              {"Content-Type": "application/json"})).read())
        assert [s["text"] for s in r["segments"]] == ["Hej.", "Vad heter du?"]
        assert all(s["msg"] == 0 for s in r["segments"])
        assert abs(sum(s["A"] for s in r["segments"]) - 1) < 1e-5
        ck = json.loads(urllib.request.urlopen(url + "/api/checksum").read())
        assert ck["unchanged"]
    finally:
        srv.shutdown()
