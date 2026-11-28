"""Kärntestets mått på syntetiska rapporter: en rapport som bara läser texten eller vanan klarar inte K4a; en som följer
S klarar K4a och K4b; nollrapport klarar inget."""
import random

import torch

import report_eval as re_


def rows(n_texts=60, seed=0):
    g = random.Random(seed)
    out = []
    for i in range(n_texts):
        for t in range(6):
            aA = g.randrange(4); txt = aA if g.random() < 0.65 else g.randrange(4)
            aS = aA if g.random() < 0.6 else g.randrange(4)
            j = g.choice([x for x in range(4) if x not in (aS, aA, txt)])
            out.append({"idx": i, "t": t, "part2": t >= 3, "since_part2": t - 3, "aS": aS, "aA": aA, "txt": txt,
                        "j": j, "dose_swap": 0.3, "dose_SA": 0.4})
    return out


def fill(rs, R, Rm, Rc):
    for x in rs:
        x["R"], x["Rm"], x["Rc"] = R(x), Rm(x), Rc(x)
    return re_.summarize(rs)


def test_follow_S_passes_text_reader_fails():
    r = fill(rows(), lambda x: x["aS"], lambda x: x["j"], lambda x: x["txt"])
    assert r["K4a_pass"] and r["K4b_pass"]
    r = fill(rows(), lambda x: x["txt"], lambda x: x["txt"], lambda x: x["txt"])
    assert not r["K4a_pass"] and not r["K4b_pass"]
    r = fill(rows(), lambda x: 3, lambda x: 3, lambda x: 3)
    assert not (r["K4a_pass"] or r["K4b_pass"] or r["K4c_pass"])
