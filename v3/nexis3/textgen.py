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


# ---------------------------------------------------------------------------
# Version 2 (sprint 2, Anders beslut 24 nov): frågan har andra ord än målmeningen,
# och en lockmening delar ord med frågan utan att svara på den. [Claudes förslag]
# (ämne, målmening med {v}, omskriven fråga, lockmening, värden)
TOPICS_V2 = [
    ("dog", "Our neighbour's dog is called {v}.", "What name does the pet next door respond to?",
     "The pet shop next door is closing this month.",
     ["Rex", "Bella", "Max", "Luna", "Charlie", "Daisy", "Rocky", "Molly"]),
    ("travel", "Last summer Maria travelled to {v} by train.", "Which city did the woman visit by rail during the warm season?",
     "The city added a new rail bridge during the warm season.",
     ["Vienna", "Lisbon", "Prague", "Oslo", "Munich", "Krakow", "Milan", "Zurich"]),
    ("bakery", "The bakery on the corner sells {v} every Friday.", "What pastry can people buy at the shop near the crossing at the end of each week?",
     "People buy their newspapers at the shop near the crossing.",
     ["croissants", "bagels", "muffins", "pretzels", "doughnuts", "scones", "waffles", "pancakes"]),
    ("football", "Our football team won the final by {v} goals.", "What was the winning margin in the championship match?",
     "The championship match drew a record crowd.",
     ["three", "four", "five", "six", "seven", "eight"]),
    ("music", "My cousin plays the {v} in a jazz band.", "Which instrument does the young musician in the family perform on?",
     "The young musician in the family also writes poems.",
     ["trumpet", "saxophone", "piano", "clarinet", "trombone", "drums", "violin", "flute"]),
    ("car", "The car parked in the garage is a {v}.", "Which brand is the automobile kept indoors at home?",
     "The plants kept indoors at home need water daily.",
     ["Volvo", "Toyota", "Renault", "Fiat", "Peugeot", "Honda", "Skoda", "Mazda"]),
    ("job", "Lisa works as a {v} at the city hospital.", "What is the profession of the woman employed at the medical centre?",
     "The medical centre employed two new cleaners.",
     ["nurse", "surgeon", "pharmacist", "radiologist", "dentist", "midwife", "paramedic", "dietitian"]),
    ("book", "The old book on the shelf was printed in {v}.", "In which year was the antique volume produced?",
     "The antique clock was produced by a Swiss workshop.",
     ["1854", "1872", "1903", "1921", "1938", "1946", "1961", "1977"]),
    ("garden", "In the garden we planted {v} trees last spring.", "Which species was put in the ground in the yard a few months ago?",
     "A new fence was put in the ground in the yard.",
     ["apple", "cherry", "plum", "pear", "walnut", "birch", "maple", "oak"]),
    ("boat", "The sailboat in the harbour is painted {v}.", "What colour is the yacht in the port?",
     "The ferry in the port leaves for Norway tomorrow.",
     ["blue", "green", "yellow", "red", "white", "orange", "purple", "black"]),
    ("chess", "The chess club meets every {v} evening.", "On which night does the board game society gather?",
     "The board game shop opened a second store last year.",
     ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Sunday"]),
    ("weather", "The forecast says it will be {v} in the mountains tomorrow.", "What conditions are expected on the peaks the next day?",
     "Hikers are expected on the peaks during the holidays.",
     ["snowy", "foggy", "windy", "stormy", "sunny", "rainy"]),
]

_STOP = set("""a an the of in on at to by for from with and or but is are was were be been it its this that these
those what which who whom whose when where why how does do did can could will would should may might
our my your his her their we you he she they i me us them there here every each next""".split())


def content_words(text: str) -> set[str]:
    ws = re.findall(r"[a-z']+", text.lower())
    return {w.removesuffix("'s").rstrip("s") or w for w in ws if w not in _STOP and len(w) > 2}


def overlap_choice(question: str, sentences: list[str]) -> int:
    """Nollmodell: meningen som delar flest innehållsord med frågan (första vid lika)."""
    q = content_words(question)
    scores = [len(q & content_words(s)) for s in sentences]
    return max(range(len(sentences)), key=lambda k: scores[k])


@dataclass
class Item2(Item):
    lure: int = -1           # 0-baserad position för lockmeningen
    version: int = 2


def make_items_v2(n: int = 100, seed: int = 0, bank=None) -> list[Item2]:
    """Fyra meningar: mål (ämne A), lockmening för A:s fråga, två fyllnadsmeningar
    (ämnen B, C). Målposition balanserad; lockmeningens position slumpas bland resten."""
    rng = random.Random(seed)
    targets = [i % 4 for i in range(n)]
    rng.shuffle(targets)
    items = []
    for idx in range(n):
        tgt = targets[idx]
        while True:
            a, b, c = rng.sample(bank or TOPICS_V2, 3)
            va, vb, vc = (rng.choice(t[4]) for t in (a, b, c))
            others = [i for i in range(4) if i != tgt]
            rng.shuffle(others)
            sents = [None] * 4
            sents[tgt] = a[1].format(v=va)
            sents[others[0]] = a[3]
            sents[others[1]] = b[1].format(v=vb)
            sents[others[2]] = c[1].format(v=vc)
            pat = re.compile(r"\b" + re.escape(va.lower()) + r"\b")
            if sum(bool(pat.search(s.lower())) for s in sents) == 1 and not pat.search(a[2].lower()):
                break
        topics = [""] * 4
        topics[tgt], topics[others[0]], topics[others[1]], topics[others[2]] = a[0], a[0] + "-lure", b[0], c[0]
        body = "\n".join(f"{k + 1}. {s}" for k, s in enumerate(sents))
        prompt = TASK_TEMPLATE.format(body=body, question=a[2])
        items.append(Item2(idx, sents, topics, tgt, a[2], va, prompt, lure=others[0]))
    return items


def make_items_v3(n: int, seed: int, split: str) -> list[Item2]:
    """Sprint 3: alla fyra meningar från samma ämnespool (träning eller undanhållna testämnen)."""
    from .topics_v3 import TEST_TOPICS, TOPICS_V3, TRAIN_TOPICS
    pool = TRAIN_TOPICS if split == "train" else TEST_TOPICS
    return make_items_v2(n, seed, bank=[t for t in TOPICS_V3 if t[0] in pool])


def make(version: int, n: int, seed: int):
    return make_items(n, seed) if version == 1 else make_items_v2(n, seed)


@dataclass
class Item3(Item2):
    twin: int = 0            # 0 = original, 1 = tvilling (samma uppbyggnad, andra ämnen)


def make_twins_v3(n_pairs: int, seed: int) -> list[Item3]:
    """Sprint 3 steg 2c: symmetripar på testämnena. Varje par har samma målposition, lockposition och
    fyllnadspositioner men tre helt andra ämnen (inga gemensamma ämnen inom paret). Ordning: x0, x0', x1, x1', ...
    Båda i paret har samma idx. [Claudes förslag]"""
    from .topics_v3 import TEST_TOPICS, TOPICS_V3
    bank = [t for t in TOPICS_V3 if t[0] in TEST_TOPICS]
    rng = random.Random(seed)
    targets = [i % 4 for i in range(n_pairs)]
    rng.shuffle(targets)

    def build(idx, tgt, others, a, b, c):
        va, vb, vc = (rng.choice(t[4]) for t in (a, b, c))
        sents = [None] * 4
        sents[tgt] = a[1].format(v=va)
        sents[others[0]] = a[3]
        sents[others[1]] = b[1].format(v=vb)
        sents[others[2]] = c[1].format(v=vc)
        pat = re.compile(r"\b" + re.escape(va.lower()) + r"\b")
        if sum(bool(pat.search(s.lower())) for s in sents) != 1 or pat.search(a[2].lower()):
            return None
        topics = [""] * 4
        topics[tgt], topics[others[0]], topics[others[1]], topics[others[2]] = a[0], a[0] + "-lure", b[0], c[0]
        body = "\n".join(f"{k + 1}. {s}" for k, s in enumerate(sents))
        return Item3(idx, sents, topics, tgt, a[2], va, TASK_TEMPLATE.format(body=body, question=a[2]), lure=others[0])

    items = []
    for idx in range(n_pairs):
        tgt = targets[idx]
        while True:
            six = rng.sample(bank, 6)
            others = [i for i in range(4) if i != tgt]
            rng.shuffle(others)
            x = build(idx, tgt, others, *six[:3])
            y = build(idx, tgt, others, *six[3:])
            if x is not None and y is not None:
                break
        y.twin = 1
        items += [x, y]
    return items
