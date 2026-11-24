"""Meningsindelning och koppling token -> mening.

Invariant: segmentens teckenintervall motsvarar exakt meningarna i texten
(testas i tests/test_segment.py på kända exempel).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Förkortningar som inte avslutar en mening (engelska och svenska).
_ABBREV = {
    "dr", "mr", "mrs", "ms", "prof", "st", "jr", "sr", "vs", "etc", "e.g", "i.e",
    "t.ex", "bl.a", "d.v.s", "dvs", "osv", "ca", "nr", "kl", "s.k", "fr.o.m", "t.o.m",
}


@dataclass(frozen=True)
class Segment:
    text: str
    start: int  # teckenindex i källsträngen (inklusive)
    end: int    # teckenindex i källsträngen (exklusive)


def split_sentences(text: str, offset: int = 0) -> list[Segment]:
    """Delar text i meningar. Radbrytningar avslutar alltid en mening.

    En mening slutar vid . ! ? (eventuellt följt av citattecken/parentes) följt av
    blanksteg och versal/siffra/citattecken, eller vid textens slut. Decimaltal
    (3.5) och förkortningar (Dr., t.ex.) delar inte.
    Blanksteg mellan meningar hör inte till någon mening.
    """
    segs: list[Segment] = []
    for line_m in re.finditer(r"[^\n]+", text):
        line = line_m.group(0)
        base = line_m.start()
        i = 0
        start = None
        n = len(line)
        while i < n:
            if start is None:
                if line[i].isspace():
                    i += 1
                    continue
                start = i
            ch = line[i]
            if ch in ".!?":
                j = i + 1
                while j < n and line[j] in ".!?\"')”’»":
                    j += 1
                at_end = j >= n or line[j:].strip() == ""
                boundary = False
                if at_end:
                    boundary = True
                elif line[j].isspace():
                    k = j
                    while k < n and line[k].isspace():
                        k += 1
                    nxt = line[k] if k < n else ""
                    if nxt and (nxt.isupper() or nxt.isdigit() or nxt in "\"'“”«(["):
                        boundary = True
                if boundary and ch == "." and re.fullmatch(r"\d+", line[start:i]):
                    boundary = False  # listnumrering "1. " inleder meningen
                if boundary and ch == ".":
                    word = re.search(r"([\w.]+)\.?$", line[start:i + 1])
                    w = word.group(1).lower().rstrip(".") if word else ""
                    if w in _ABBREV or (len(w) == 1 and w.isalpha()):
                        boundary = False
                if boundary:
                    end = j
                    segs.append(Segment(line[start:end], offset + base + start, offset + base + end))
                    start = None
                    i = j
                    continue
            i += 1
        if start is not None:
            seg_text = line[start:].rstrip()
            if seg_text:
                segs.append(Segment(seg_text, offset + base + start, offset + base + start + len(seg_text)))
    return segs


def token_segment_ids(offsets: list[tuple[int, int]], segments: list[Segment]) -> list[int]:
    """Segmentindex per token, -1 om token inte hör till något segment.

    En token hör till det segment som innehåller tokenens första icke-blanka
    tecken. Hanteras via teckenintervallet: token (a, b) tillhör segment s om
    s.start <= a' < s.end där a' = första tecknet i [a, b) som ligger inom
    något segment. Specialtokens har (0, 0) och får -1.
    """
    ids = []
    for a, b in offsets:
        sid = -1
        if b > a:
            for k, s in enumerate(segments):
                lo, hi = max(a, s.start), min(b, s.end)
                if lo < hi:
                    sid = k
                    break
        ids.append(sid)
    return ids
