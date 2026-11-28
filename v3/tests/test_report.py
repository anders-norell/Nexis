"""Rapportmodellen: LoRA ändrar inget när B = 0 eller avstängd; basvikter fryses; facit kommer från A, inte S."""
import torch

from nexis3.report import LoRALinear, Reporter, SoftProj, add_lora, set_lora
from nexis3.textgen import SYSTEM_PROMPT, make_items_v4


def test_lora_identity_at_init_and_when_disabled(measurer):
    m = measurer.model
    x = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        ref = m(x).logits.clone()
    mods = add_lora(m)
    with torch.no_grad():
        assert torch.allclose(m(x).logits, ref, atol=1e-5)          # B = 0 vid start
        for md in mods:
            md.B.normal_()
        set_lora(mods, False)
        assert torch.allclose(m(x).logits, ref, atol=1e-5)          # avstängd = basmodell
        set_lora(mods, True)
        assert not torch.allclose(m(x).logits, ref, atol=1e-3)
    assert all(not p.requires_grad for n, p in m.named_parameters() if ".A" not in n and ".B" not in n)


def test_reporter_soft_tokens_matter(measurer):
    m = measurer.model
    add_lora(m)
    proj = SoftProj(m.config.hidden_size)
    with torch.no_grad():
        proj.lin.weight.normal_()
    rep = Reporter(measurer, proj)
    it = make_items_v4(1, 9100, "train")[0]
    with torch.no_grad():
        a = rep.logits(SYSTEM_PROMPT, it.prompt, "1) x", torch.tensor([1.0, 0, 0, 0]))
        b = rep.logits(SYSTEM_PROMPT, it.prompt, "1) x", torch.tensor([0, 0, 0, 1.0]))
    assert a.shape == (4,) and not torch.allclose(a, b)


def test_frozen_prefix_matches_full_pass_and_trains(measurer):
    """Med LoRA avstängd ger cache-vägen samma logits som en hel körning; med LoRA på påverkas bara slutet, och
    gradient når både LoRA och projektorn."""
    m = measurer.model
    mods = add_lora(m)
    proj = SoftProj(m.config.hidden_size)
    with torch.no_grad():
        proj.lin.weight.normal_()
    it = make_items_v4(1, 9100, "train")[0]
    s = torch.tensor([0.1, 0.6, 0.2, 0.1])
    full, cached = Reporter(measurer, proj), Reporter(measurer, proj, mods)
    set_lora(mods, False)
    with torch.no_grad():
        assert torch.allclose(full.logits(SYSTEM_PROMPT, it.prompt, "1) x", s),
                              cached.logits(SYSTEM_PROMPT, it.prompt, "1) x", s), atol=1e-4)
    set_lora(mods, True)
    with torch.no_grad():
        for md in mods:
            md.B.normal_()
    lg = cached.logits(SYSTEM_PROMPT, it.prompt, "1) x", s)
    lg.sum().backward()
    assert proj.lin.weight.grad is not None and mods[-1].A.grad is not None
    assert all(md.A.dtype == torch.float32 for md in mods)
