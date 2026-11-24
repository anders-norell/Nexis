"""Styrning (sprint 2, krit. 3) [Claudes förslag].

Schemats utdata S läggs som en bias på attention-poängen i de valda lagren:
    bias[k] = beta * log(S[s] * n_seg)   för token k i mening s   (0 om S är jämn)
    bias[k] = 0                           för tokens utanför meningarna
Biasen läggs bara på rader (frågepositioner) från sista prompttoken och framåt, alltså medan svaret skrivs,
och bara i lagren `layers`. Basmodellens vikter rörs inte; biasen läggs på attention-masken via en
forward-pre-hook på respektive attention-modul.
"""
from __future__ import annotations

import contextlib
import math

import torch


class Steerer:
    def __init__(self, model, layers: list[int]):
        self.layers = list(layers)
        self.key_bias: torch.Tensor | None = None   # (n_prompt,)
        self.start: int = 0
        self.calls = 0
        self._handles = [model.model.layers[l].self_attn.register_forward_pre_hook(self._hook, with_kwargs=True)
                         for l in self.layers]

    def remove(self):
        for h in self._handles:
            h.remove()

    @staticmethod
    def make_bias(S, seg_ids, beta: float) -> torch.Tensor:
        n = len(S)
        logs = torch.tensor([math.log(max(float(s), 1e-6) * n) for s in S])
        sid = torch.tensor(seg_ids)
        b = torch.zeros(len(seg_ids))
        m = sid >= 0
        b[m] = beta * logs[sid[m]]
        return b

    @contextlib.contextmanager
    def active(self, key_bias: torch.Tensor, start: int):
        self.key_bias, self.start = key_bias, start
        try:
            yield self
        finally:
            self.key_bias = None

    def _hook(self, module, args, kwargs):
        if self.key_bias is None:
            return None
        mask = kwargs.get("attention_mask")
        pos = kwargs["position_ids"][0]                       # absoluta positioner för raderna
        Q, K = mask.shape[-2], mask.shape[-1]
        if mask.dtype == torch.bool:
            mask = torch.zeros(mask.shape, dtype=torch.float32).masked_fill(~mask, float("-inf"))
        b = torch.zeros(K, dtype=mask.dtype)
        n = min(K, len(self.key_bias))
        b[:n] = self.key_bias[:n].to(mask.dtype)
        rows = (pos >= self.start).to(mask.dtype)             # (Q,)
        kwargs["attention_mask"] = mask + rows[None, None, :, None] * b[None, None, None, :]
        self.calls += 1
        return args, kwargs
