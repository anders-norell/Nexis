"""Invarianterna för sprint 1 som kontrollfunktioner.

Testerna kör dem på den riktiga koden (ska passera) och på medvetet trasiga
varianter (ska fallera), så att det syns att kontrollerna kan slå larm.
"""
from __future__ import annotations


def check_sums_to_one(A, tol=1e-5):
    assert len(A) > 0, "tomt A"
    assert all(a >= -1e-9 for a in A), f"negativ attention: {A}"
    assert abs(sum(A) - 1.0) < tol, f"A summerar till {sum(A)}, inte 1"


def check_changes(A1, A2, tol=1e-6):
    assert len(A1) == len(A2)
    d = max(abs(a - b) for a, b in zip(A1, A2))
    assert d > tol, f"A ändrades inte när texten ändrades (största skillnad {d})"


def check_boundaries(source: str, segments, expected: list[str] | None = None):
    for s in segments:
        assert source[s.start:s.end] == s.text, f"intervall {s.start}:{s.end} != {s.text!r}"
        assert s.text == s.text.strip(), f"segment med blanksteg i kanten: {s.text!r}"
    if expected is not None:
        got = [s.text for s in segments]
        assert got == expected, f"fel meningsgränser:\n  fick    {got}\n  väntade {expected}"


def check_weights_unchanged(ck_before: str, ck_after: str):
    assert ck_before == ck_after, "basmodellens vikter har ändrats"
