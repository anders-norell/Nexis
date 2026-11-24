"""Generator för kärntestets texter [Claudes förslag: engelska, mallbaserat].

Varje text = fyra meningar om fyra olika ämnen + en fråga som bara kan besvaras
med en av dem (målmeningen). Värdena i meningarna är unika inom texten, så
svaret pekar entydigt ut målmeningen. Målets position balanseras (lika många
på plats 1, 2, 3, 4) och allt styrs av ett frö.
Engelska valdes eftersom modeller i storleken 1–1,5B följer instruktioner
säkrare på engelska; chatten fungerar på svenska också.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, asdict

# (ämne, mening med {v}, fråga, värden)
TOPICS = [
    ("dog", "My neighbour's dog is called {v}.", "What is the neighbour's dog called?",
     ["Rex", "Bella", "Max", "Luna", "Charlie", "Daisy", "Rocky", "Molly"]),
    ("travel", "Last summer Maria travelled to {v} by train.", "Where did Maria travel last summer?",
     ["Vienna", "Lisbon", "Prague", "Oslo", "Munich", "Krakow", "Milan", "Zurich"]),
    ("bakery", "The bakery on the corner sells {v} every Friday.", "What does the bakery on the corner sell on Fridays?",
     ["croissants", "bagels", "muffins", "pretzels", "doughnuts", "scones", "waffles", "pancakes"]),
    ("football", "Our football team won the final by {v} goals.", "By how many goals did the football team win the final?",
     ["three", "four", "five", "six", "seven", "eight"]),
    ("music", "Erik plays the {v} in a jazz band.", "Which instrument does Erik play?",
     ["trumpet", "saxophone", "piano", "clarinet", "trombone", "drums", "violin", "flute"]),
    ("car", "The car parked in the garage is a {v}.", "What make is the car in the garage?",
     ["Volvo", "Toyota", "Renault", "Fiat", "Peugeot", "Honda", "Skoda", "Mazda"]),
    ("job", "Lisa works as a {v} at the city hospital.", "What does Lisa work as?",
     ["nurse", "surgeon", "pharmacist", "radiologist", "dentist", "midwife", "paramedic", "dietitian"]),
    ("book", "The old book on the shelf was printed in {v}.", "When was the old book printed?",
     ["1854", "1872", "1903", "1921", "1938", "1946", "1961", "1977"]),
    ("garden", "In the garden we planted {v} trees last spring.", "What kind of trees were planted in the garden?",
     ["apple", "cherry", "plum", "pear", "walnut", "birch", "maple", "oak"]),
    ("boat", "The sailboat in the harbour is painted {v}.", "What colour is the sailboat in the harbour?",
     ["blue", "green", "yellow", "red", "white", "orange", "purple", "black"]),
    ("chess", "The chess club meets every {v} evening.", "On which evening does the chess club meet?",
     ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Sunday"]),
    ("weather", "The forecast says it will be {v} in the mountains tomorrow.", "What is the forecast for the mountains tomorrow?",
     ["snowy", "foggy", "windy", "stormy", "sunny", "rainy"]),
]

TASK_TEMPLATE = (
    "Read these four sentences.\n\n{body}\n\n"
    "Question: {question} Answer in a few words."
)
SYSTEM_PROMPT = "You are a helpful assistant."  # uttrycklig, oberoende av mallens standard

REPORT_QUESTION = (
    "Which of the four sentences did you focus on most when you answered? "
    "Reply with the number only (1, 2, 3 or 4)."
)


@dataclass
class Item:
    idx: int
    sentences: list[str]     # de fyra meningarna, i ordning
    topics: list[str]
    target: int              # 0-baserad position för målmeningen
    question: str
    answer: str              # korrekt värde
    prompt: str              # användarmeddelandet

    def to_dict(self):
        return asdict(self)


def make_items(n: int = 100, seed: int = 0) -> list[Item]:
    rng = random.Random(seed)
    targets = [i % 4 for i in range(n)]
    rng.shuffle(targets)
    items = []
    for idx in range(n):
        tgt = targets[idx]
        while True:  # svaret får bara förekomma i målmeningen
            chosen = rng.sample(TOPICS, 4)
            vals = [rng.choice(t[3]) for t in chosen]
            sents = [t[1].format(v=v) for t, v in zip(chosen, vals)]
            pat = re.compile(r"\b" + re.escape(vals[tgt].lower()) + r"\b")
            if sum(bool(pat.search(s.lower())) for s in sents) == 1 and not pat.search(chosen[tgt][2].lower()):
                break
        body = "\n".join(f"{k + 1}. {s}" for k, s in enumerate(sents))
        prompt = TASK_TEMPLATE.format(body=body, question=chosen[tgt][2])
        items.append(Item(idx, sents, [t[0] for t in chosen], tgt, chosen[tgt][2], vals[tgt], prompt))
    return items


def parse_report(text: str) -> int | None:
    """Första siffran 1–4 i svaret, 0-baserad; None om ingen finns."""
    for ch in text:
        if ch in "1234":
            return int(ch) - 1
    return None


_NUM = {"three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8"}


def answer_correct(item: Item, answer: str) -> bool:
    a = answer.lower()
    v = item.answer.lower()
    hit = lambda w: re.search(r"\b" + re.escape(w) + r"\b", a) is not None
    return hit(v) or (v in _NUM and hit(_NUM[v]))
