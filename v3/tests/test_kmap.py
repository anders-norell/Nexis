import torch

from nexis3.kmap import M, KMap, baselines


def test_map_is_distribution_and_uses_only_observations():
    m = KMap()
    H = torch.softmax(torch.randn(9, 4), -1)
    q = torch.randn(1536)
    S = m.run(H, q)
    assert torch.allclose(S.exp().sum(-1), torch.ones(9), atol=1e-5)
    H2 = H.clone()
    H2[1:4] = torch.softmax(torch.randn(3, 4), -1)   # ändra bara oobserverade token
    assert torch.allclose(m.run(H2, q)[:4], S[:4])


def test_baselines_hold_last_observation():
    H = torch.softmax(torch.randn(6, 4), -1)
    obs = torch.arange(6) % M == 0
    B = baselines(H, obs, torch.full((10, 4), 0.25), torch.full((4,), 0.25))
    assert torch.allclose(B["hold"][3], H[0]) and torch.allclose(B["hold"][5], H[4])
