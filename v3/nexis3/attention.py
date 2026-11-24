"""Mätning av attention per mening.

Definition (A) [Claudes förslag]:
  För ett svar med tokenpositioner Q (svarstokens; finns inga, används sista
  prompttoken) och valda lager L (default mittersta tredjedelen):
      raw[s] = medel över l i L, huvud h, q i Q av  sum_{k i segment s} attn[l, h, q, k]
      A[s]   = raw[s] / sum_s raw[s]
  A summerar alltså till 1 över segmenten (invariant 1). Attention på tokens
  utanför segmenten (systemprompt, mallmarkörer, frågan) räknas inte i A men
  redovisas som `mass_in_segments` (andel av all attention som går till segmenten).

Basmodellens vikter rörs aldrig: modellen fryses och en kontrollsumma tas vid
laddning och kan jämföras efteråt (invariant 3).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import torch

from .segment import Segment, split_sentences, token_segment_ids

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def load_model(model_id: str = DEFAULT_MODEL, dtype: str = "auto"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    if dtype == "auto":  # float32 om minnet räcker, annars bfloat16 [Claudes förslag]
        ram = total_ram_gb()
        dt = torch.float32 if ram is None or ram >= 14 else torch.bfloat16
    else:
        dt = getattr(torch, dtype)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dt, attn_implementation="eager")
    freeze(model)
    return model, tok


def total_ram_gb():
    try:
        import os
        if os.name == "nt":
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("l", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                            ("avail", ctypes.c_ulonglong), ("tp", ctypes.c_ulonglong), ("ap", ctypes.c_ulonglong),
                            ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong), ("ae", ctypes.c_ulonglong)]
            m = MS()
            m.l = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return m.total / 2**30
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    except Exception:
        return None


def freeze(model):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def weights_checksum(model) -> str:
    """sha256 över alla parametrar och buffertar, i namnordning."""
    h = hashlib.sha256()
    sd = model.state_dict()
    for name in sorted(sd):
        t = sd[name].detach().to("cpu").contiguous().reshape(-1)
        h.update(name.encode())
        h.update(str(t.dtype).encode())
        h.update(t.view(torch.uint8).numpy())  # buffert, ingen kopia
    return h.hexdigest()


def default_layers(n_layers: int) -> list[int]:
    return list(range(n_layers // 3, (2 * n_layers) // 3))


@dataclass
class Measurement:
    segments: list[Segment]          # meningarna (text + teckenintervall i prompten)
    seg_owner: list[int]             # index i messages som meningen kommer från
    A: list[float]                   # attention per mening, summa 1
    mass_in_segments: float          # andel av all attention som går till meningarna
    per_layer: list[list[float]] = field(default_factory=list)  # A per valt lager (normerat)
    layers: list[int] = field(default_factory=list)
    answer: str = ""
    prompt: str = ""


class Measurer:
    def __init__(self, model, tok, layers: list[int] | None = None):
        self.model = model
        self.tok = tok
        n = model.config.num_hidden_layers
        self.layers = layers if layers is not None else default_layers(n)

    # ---- prompt och segment -------------------------------------------------
    def render(self, messages: list[dict], add_generation_prompt: bool = True) -> str:
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)

    def locate_segments(self, prompt: str, messages: list[dict], which: list[int] | None = None):
        """Meningar i de valda meddelandena (default: alla utom system), med
        teckenintervall i den renderade prompten."""
        segs, owner = [], []
        cursor = 0
        for i, m in enumerate(messages):
            pos = prompt.find(m["content"], cursor)
            if pos < 0:
                raise ValueError(f"meddelande {i} hittades inte i prompten")
            cursor = pos + len(m["content"])
            if which is None:
                take = m["role"] != "system"
            else:
                take = i in which
            if take:
                for s in split_sentences(m["content"], offset=pos):
                    segs.append(s)
                    owner.append(i)
        return segs, owner

    def locate_given(self, prompt: str, texts: list[str]):
        """Givna segmenttexter i ordning, med teckenintervall i prompten."""
        segs, cursor = [], 0
        for t in texts:
            pos = prompt.find(t, cursor)
            if pos < 0:
                raise ValueError(f"segmentet {t!r} hittades inte i prompten")
            segs.append(Segment(t, pos, pos + len(t)))
            cursor = pos + len(t)
        return segs

    def encode(self, prompt: str):
        enc = self.tok(prompt, return_offsets_mapping=True, add_special_tokens=False)
        return enc["input_ids"], [tuple(o) for o in enc["offset_mapping"]]

    # ---- generering ---------------------------------------------------------
    @torch.no_grad()
    def generate(self, ids: list[int], max_new_tokens: int = 64) -> list[int]:
        x = torch.tensor([ids])
        out = self.model.generate(
            x, attention_mask=torch.ones_like(x), max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id,
            temperature=None, top_p=None, top_k=None,
        )
        new = out[0, len(ids):].tolist()
        eos = self.model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos]) if eos is not None else set()
        if self.tok.eos_token_id is not None:
            eos.add(self.tok.eos_token_id)
        while new and new[-1] in eos:
            new.pop()
        return new

    # ---- mätning ------------------------------------------------------------
    @torch.no_grad()
    def attention_per_segment(self, prompt_ids, answer_ids, seg_ids, n_seg):
        full = list(prompt_ids) + list(answer_ids)
        x = torch.tensor([full])
        out = self.model(x, attention_mask=torch.ones_like(x), output_attentions=True, use_cache=False)
        n_p = len(prompt_ids)
        q = list(range(n_p, len(full))) if answer_ids else [n_p - 1]
        owner = torch.full((len(full),), -1, dtype=torch.long)
        owner[:n_p] = torch.tensor(seg_ids, dtype=torch.long)
        onehot = torch.zeros(len(full), n_seg)
        m = owner >= 0
        onehot[m.nonzero().squeeze(-1), owner[m]] = 1.0
        per_layer_raw = []
        total = []
        for l in self.layers:
            att = out.attentions[l][0].float()        # (heads, L, L)
            rows = att[:, q, :]                       # (heads, Q, L)
            seg_mass = rows @ onehot                  # (heads, Q, S)
            per_layer_raw.append(seg_mass.mean(dim=(0, 1)))
            total.append(rows.sum(-1).mean())
        raw = torch.stack(per_layer_raw).mean(0)
        tot = torch.stack(total).mean()
        A = raw / raw.sum()
        per_layer = [(r / r.sum()).tolist() for r in per_layer_raw]
        return A.tolist(), float(raw.sum() / tot), per_layer

    def measure(self, messages: list[dict], answer: str | None = None, max_new_tokens: int = 64,
                which: list[int] | None = None, segment_texts: list[str] | None = None) -> Measurement:
        """Genererar ett svar på `messages` (om `answer` inte ges) och mäter A
        över meningarna i kontexten medan svaret produceras. Med `segment_texts`
        mäts bara över de givna textbitarna (första förekomst efter föregående)."""
        prompt = self.render(messages)
        if segment_texts is not None:
            segs = self.locate_given(prompt, segment_texts)
            owner = [-1] * len(segs)
        else:
            segs, owner = self.locate_segments(prompt, messages, which)
        if not segs:
            raise ValueError("inga meningar att mäta på")
        ids, offs = self.encode(prompt)
        seg_ids = token_segment_ids(offs, segs)
        if answer is None:
            ans_ids = self.generate(ids, max_new_tokens)
            answer = self.tok.decode(ans_ids, skip_special_tokens=True)
        else:
            ans_ids = self.tok(answer, add_special_tokens=False)["input_ids"]
        A, mass, per_layer = self.attention_per_segment(ids, ans_ids, seg_ids, len(segs))
        return Measurement(segs, owner, A, mass, per_layer, list(self.layers), answer, prompt)
