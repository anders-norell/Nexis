"""Sprint 4: rapportmodellen – Qwen + LoRA + mjuka token från kartan. [Claudes förslag]

Förregistrering: results/sprint4_prereg.md (del 2b). Rapportfrågan ställs mitt i svaret: svaret kapas efter token t,
och i en ny användarvända ligger kartans tillstånd S_t som N_SOFT mjuka token före frågan. Facit = argmax av den
FAKTISKA attentionen A_t (inte kartan). Kontrollmodellen tränas likadant men får alltid S = likformig (ingen karta).
Basvikterna fryses; LoRA (rang R) på q_proj och v_proj i alla lager, egen implementation (inget peft-beroende).
"""
from __future__ import annotations

import math

import torch
from torch import nn

N_SOFT = 4
R = 8
ALPHA = 16
REPORT_Q = "Which of the four sentences (1, 2, 3 or 4) were you using for the last word you wrote? Answer with the number only."
DIGITS = ["1", "2", "3", "4"]


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=R, alpha=ALPHA):
        super().__init__()
        self.base = base
        self.A = nn.Parameter(torch.randn(r, base.in_features) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        self.scale = alpha / r
        self.enabled = True

    def forward(self, x):
        y = self.base(x)
        if self.enabled:
            y = y + ((x.to(self.A.dtype) @ self.A.t() @ self.B.t()) * self.scale).to(y.dtype)
        return y


def add_lora(model):
    """Fryser basvikterna och lägger LoRA på q_proj och v_proj. Returnerar listan med LoRA-moduler."""
    for p in model.parameters():
        p.requires_grad_(False)
    mods = []
    for layer in model.model.layers:
        att = layer.self_attn
        for name in ("q_proj", "v_proj"):
            lin = getattr(att, name)
            if isinstance(lin, LoRALinear):
                mods.append(lin)
                continue
            w = LoRALinear(lin)
            w.A.data = w.A.data.to(lin.weight.device, torch.float32)   # LoRA i float32 (granskning anm. 11)
            w.B.data = w.B.data.to(lin.weight.device, torch.float32)
            setattr(att, name, w)
            mods.append(w)
    for m in mods:
        m.A.requires_grad_(True); m.B.requires_grad_(True)
    return mods


def set_lora(mods, enabled: bool):
    for m in mods:
        m.enabled = enabled


class SoftProj(nn.Module):
    """S (4 sannolikheter) → N_SOFT inbäddningar."""
    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(4, N_SOFT * d)
        nn.init.normal_(self.lin.weight, std=0.02)
        nn.init.zeros_(self.lin.bias)
        self.float()
        self.d = d

    def forward(self, s):
        return self.lin(s).view(N_SOFT, self.d)


def lora_state(mods, proj):
    return {"lora": [(m.A.detach().cpu(), m.B.detach().cpu()) for m in mods], "proj": proj.state_dict()}


def load_lora_state(mods, proj, st):
    for m, (a, b) in zip(mods, st["lora"]):
        m.A.data.copy_(a); m.B.data.copy_(b)
    proj.load_state_dict(st["proj"])


class Reporter:
    """Bygger rapportprompten och ger sannolikheter över svaren 1–4."""
    def __init__(self, meas, proj, mods=None):
        """mods: LoRA-modulerna. Prefixet (samtalet fram till de mjuka token) räknas då med LoRA avstängd och utan
        gradient, så rapporten läser samma inre tillstånd som gav attentionen (granskning anm. 9); LoRA verkar bara på
        de mjuka token och frågan."""
        self.meas, self.proj, self.mods = meas, proj, mods
        tok = meas.tok
        self.digit_ids = [tok(d, add_special_tokens=False)["input_ids"][0] for d in DIGITS]

    def parts(self, system, user, partial_answer):
        """(prefix-id, suffix-id): prefix = samtal fram till där de mjuka token ska in, suffix = frågan + assistentstart."""
        tok = self.meas.tok
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user},
                {"role": "assistant", "content": partial_answer}, {"role": "user", "content": "⁣SOFT⁣ " + REPORT_Q}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        a, b = text.split("⁣SOFT⁣")
        pre = tok(a, add_special_tokens=False)["input_ids"]
        suf = tok(b, add_special_tokens=False)["input_ids"]
        return pre, suf

    def logits(self, system, user, partial_answer, s):
        model = self.meas.model
        emb = model.get_input_embeddings()
        pre, suf = self.parts(system, user, partial_answer)
        dev = emb.weight.device
        tail = torch.cat([self.proj(s.to(dev).float()).to(emb.weight.dtype), emb(torch.tensor(suf, device=dev))])[None]
        if self.mods is None:
            e = torch.cat([emb(torch.tensor(pre, device=dev))[None], tail], 1)
            out = model(inputs_embeds=e, attention_mask=torch.ones(e.shape[:2], device=dev), use_cache=False)
            return out.logits[0, -1, self.digit_ids].float()
        was = [m.enabled for m in self.mods]
        set_lora(self.mods, False)
        with torch.no_grad():
            pkv = model(input_ids=torch.tensor([pre], device=dev), use_cache=True).past_key_values
        for m, w in zip(self.mods, was):
            m.enabled = w
        n = len(pre) + tail.shape[1]
        out = model(inputs_embeds=tail, past_key_values=pkv, attention_mask=torch.ones((1, n), device=dev),
                    use_cache=False)
        return out.logits[0, -1, self.digit_ids].float()
