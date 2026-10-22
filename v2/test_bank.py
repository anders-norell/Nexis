"""Tests for the sprint 4 seed bank and H2 interventions (bank.py).

Only untrained agents are used here (fresh random weights). H2 code is never run on the pilot's
trained agents, so that no H2 result is seen before the MVP runs.

  python3 test_bank.py
"""
import numpy as np
from env import EnvConfig, SearchEnv
from agent_rl import make_agent
from bank import BankEnv, run_bank, streams, bank_seeds, h1_seeds, h1_overrides, H2_T

CFG = EnvConfig()
SEEDS = bank_seeds(60)


def agent(mode, seed=3):
    return make_agent(CFG, seed=seed, schema_mode=mode)


def trace_equal(t1, t2, upto=None, keys=("a", "intended", "actual", "active", "captured")):
    n = len(t1) if upto is None else upto
    for s in range(n):
        for k in keys:
            if not np.array_equal(t1[s][k], t2[s][k]):
                return False
    return True


def test_bank_episode_matches_single_reset():
    env_rng, _, _ = streams(SEEDS[5])
    ref = SearchEnv(CFG, seed=0)
    ref.rng = env_rng
    ref.reset(1)
    be = BankEnv(CFG)
    be.reset_bank([streams(s)[0] for s in SEEDS[:10]])
    for k in ("n", "w", "k", "budget", "color", "shape", "target", "salient", "sal_obs"):
        assert np.array_equal(getattr(be, k)[5], getattr(ref, k)[0]), k


def test_reproducible():
    for mode in ("off", "fixed", "shuffled"):
        ag = agent(mode)
        r1 = run_bank(ag, SEEDS, record=True, block=30)
        r2 = run_bank(ag, SEEDS, record=True, block=30)
        for k in ("found", "steps", "revisits", "aim_a", "aim_b"):
            assert np.array_equal(r1[k], r2[k]), (mode, k)
        for t1, t2 in zip(r1["trace"], r2["trace"]):
            assert trace_equal(t1, t2)


def test_batch_independence():
    """An episode evolves the same whatever else runs in its batch (not for Blandat, whose donor
    is by design the next episode in the block)."""
    for mode in ("off", "fixed"):
        ag = agent(mode)
        big = run_bank(ag, SEEDS[:12], block=12)
        for i in range(12):
            one = run_bank(ag, SEEDS[i:i + 1], block=1)
            assert one["found"][0] == big["found"][i] and one["steps"][0] == big["steps"][i], (mode, i)
            assert one["revisits"][0] == big["revisits"][i]


def test_donor_never_self():
    ag = agent("shuffled")
    r = run_bank(ag, SEEDS[:10], record=True, block=10)
    sch = r["trace"][0][1]["schema"]               # schema at t=1 = donor's first window
    act = r["trace"][0][0]["actual"]
    for b in range(10):
        d = (b + 1) % 10
        r0, c0 = act[d]
        assert sch[b, r0, c0, 0] == 1.0             # shows the donor's window, not its own


def _runs(mode, kind, seeds=SEEDS):
    ag = agent(mode)
    base = run_bank(ag, seeds, record=True, block=30)
    lie = run_bank(ag, seeds, intervention=kind, record=True, block=30)
    return base, lie


def test_identical_until_intervention():
    for mode in ("fixed", "shuffled"):
        for kind in ("a", "b", "c"):
            base, lie = _runs(mode, kind)
            for tb, tl, cb, cl in zip(base["trace"], lie["trace"], base["cells"], lie["cells"]):
                # commands 1-3 and their outcomes identical
                assert trace_equal(tb, tl, upto=H2_T), (mode, kind)
                for s in range(1, H2_T):
                    assert np.array_equal(tb[s]["schema"], tl[s]["schema"])
                # same cells chosen in both runs
                for key in ("A", "B", "C"):
                    assert np.array_equal(cb[key], cl[key])
            for k in ("elig_a", "elig_b", "elig_c", "n_cells_b"):
                assert np.array_equal(base[k], lie[k])


def test_lie_changes_exactly_the_cells_at_fourth_command():
    for mode in ("fixed", "shuffled"):
        for kind, ch in (("a", 1), ("b", 1), ("c", 0)):
            base, lie = _runs(mode, kind)
            off = 0
            for tb, tl, cells in zip(base["trace"], lie["trace"], lie["cells"]):
                if len(tb) <= H2_T:
                    off += len(tb[0]["a"]); continue
                sb, sl = tb[H2_T]["schema"], tl[H2_T]["schema"]
                B = sb.shape[0]
                elig = lie["elig_" + kind][off:off + B]
                M = {"a": cells["A"], "b": cells["B"], "c": cells["C"]}[kind]
                other = 1 - ch
                assert np.array_equal(sb[..., other], sl[..., other])
                for b in range(B):
                    diff = sb[b, ..., ch] != sl[b, ..., ch]
                    if not elig[b]:
                        assert not diff.any()
                        continue
                    allowed = M[b] | (sb[b, ..., 0] > 0) if kind == "c" else M[b]   # (c) replaces the window
                    assert not (diff & ~allowed).any(), (mode, kind, b)    # nothing outside the cells
                    if kind == "a":
                        assert (sl[b, ..., 1][M[b]] == 0).all()
                        if mode == "fixed":
                            assert (sb[b, ..., 1][M[b]] == 1).all()        # truly covered
                    elif kind == "b":
                        assert (sl[b, ..., 1][M[b]] == 1).all()
                        if mode == "fixed":
                            assert (sb[b, ..., 1][M[b]] == 0).all()        # truly not covered
                    else:
                        assert (sl[b, ..., 0] == M[b]).all()
                off += B


def test_cells_b_rules():
    _, lie = _runs("fixed", "b")
    off = 0
    for tl, cells in zip(lie["trace"], lie["cells"]):
        Bm, tcol, tgt = cells["B"], cells["tcol"], cells["target"]
        nB = Bm.shape[0]
        for b in range(nB):
            n = int(Bm[b].sum())
            assert n <= 3
            assert n == min(3, lie["n_cells_b"][off + b])
            assert not Bm[b, tgt[b, 0], tgt[b, 1]]                  # never the target
            assert tcol[b][Bm[b]].all()                              # target's colour
        off += nB
    assert lie["elig_b"].sum() > 0
    assert np.array_equal(lie["elig_b"], lie["elig_a"] & (lie["n_cells_b"] > 0))


def test_lie_persists_and_new_coverage_registers():
    for kind in ("a", "b"):
        _, lie = _runs("fixed", kind)
        off = 0
        for tl, cells in zip(lie["trace"], lie["cells"]):
            B = tl[0]["a"].shape[0]
            M = cells["A"] if kind == "a" else cells["B"]
            elig = lie["elig_" + kind][off:off + B]
            G = M.shape[1]
            post = np.zeros((B, G, G), bool)
            w = lie["w"][off:off + B]
            for s in range(H2_T, len(tl)):
                if s > H2_T:                                  # coverage by step s-1's actual window
                    prev = tl[s - 1]
                    for b in np.nonzero(prev["active"])[0]:
                        r0, c0 = prev["actual"][b]
                        post[b, r0:r0 + w[b], c0:c0 + w[b]] = True
                sch = tl[s]["schema"][..., 1]
                for b in np.nonzero(elig & tl[s]["active"])[0]:
                    if kind == "a":
                        assert np.array_equal(sch[b][M[b]] > 0, post[b][M[b]]), (s, b)
                    else:
                        assert (sch[b][M[b]] == 1).all()
            off += B


def test_not_running_not_counted():
    """Eligible = still running after three steps; others get no lie and no command is counted."""
    _, lie = _runs("fixed", "a")
    assert np.array_equal(lie["elig_a"], lie["steps"] > H2_T)
    assert (~lie["elig_a"]).sum() > 0
    for k in ("aim_a", "aim_b", "aim_c"):
        assert (lie[k][~lie["elig_a"]] == 0).all()


def test_aim_counts_at_most_three():
    _, lie = _runs("fixed", "a")
    assert lie["aim_a"].max() <= 3 and lie["aim_b"].max() <= 3


def test_h1_bank_overrides():
    ag = agent("off")
    r = run_bank(ag, h1_seeds(8, 20), overrides=h1_overrides(8), block=20)
    assert (r["n"] == 8).all() and (r["w"] == 1).all() and (r["k"] == 8).all()
    assert len(set(h1_seeds(4)) & set(bank_seeds())) == 0


def test_schema_matches_training_rule():
    """Without intervention, the bank builds the schema exactly as training does (agent_rl.schema_input)."""
    from agent_rl import schema_input
    for mode in ("fixed", "shuffled"):
        ag = agent(mode)
        r = run_bank(ag, SEEDS[:30], record=True, block=30)
        tr = r["trace"][0]
        G = CFG.G
        donor = (np.arange(30) + 1) % 30

        class E:                       # what schema_input reads from the environment
            pass
        e = E(); e.cfg = CFG; e.B = 30; e.w = r["w"]
        visited = np.zeros((30, G, G))
        for s in range(len(tr) - 1):
            info = dict(actual=tr[s]["actual"], active=tr[s]["active"])
            sch, visited = schema_input(mode, e, info, visited, None, donor)
            assert np.array_equal(sch, tr[s + 1]["schema"]), (mode, s)


def test_a_cells_shown_equals_own_for_1a():
    ag = agent("fixed")
    r1 = run_bank(ag, SEEDS, intervention="a", block=30)
    r2 = run_bank(ag, SEEDS, intervention="a", block=30, a_cells="shown")
    for k in ("found", "steps", "aim_a", "elig_a"):
        assert np.array_equal(r1[k], r2[k])


def test_lesion_mode_has_no_schema():
    ag = agent("fixed")
    r = run_bank(ag, SEEDS[:20], mode="off", record=True, block=20)
    assert all(s["schema"] is None for s in r["trace"][0])


if __name__ == "__main__":
    import sys
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    fails = 0
    for f in tests:
        try:
            f(); print("ok  ", f.__name__)
        except Exception as e:
            fails += 1; print("FAIL", f.__name__, repr(e))
    print(f"{len(tests) - fails}/{len(tests)} passed")
    sys.exit(1 if fails else 0)
