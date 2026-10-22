"""Sprint 4 (MVP): the fixed seed bank for test episodes and the H2 interventions.

Seed bank (locked at gate 3)
----------------------------
Test episode i has its own seed. From it three independent streams are spawned
(numpy SeedSequence(seed).spawn(3)):

  env     everything the environment draws: grid, objects, salient distractors, captures,
          the salience noise in the periphery, the fovea reveal
  agent   the agent's own choices (Gumbel noise for sampling its stochastic policy)
  h2      which target-coloured cells H2 (b) marks falsely as covered, and where the
          false "window last step" goes. Kept apart so that drawing it never shifts the other two
          streams: the runs with and without an intervention stay identical up to the intervention.

The spec names two streams (environment, agent). The third is a technical addition that only
serves H2's random choice of cells; it is drawn from the same episode seed.

An episode therefore starts the same way in every condition and every training seed, and it
evolves the same way as long as the agent makes the same choices, whatever other episodes run in
the same batch (tested in test_bank.py).

H2 interventions (locked at gate 3)
-----------------------------------
The lie is put into the schema map "covered so far", in the schema the agent receives before its
fourth command, in episodes still running after three steps. It lasts for the rest of the episode;
coverage after the intervention is registered as usual.

  a  falsely unvisited  every cell actually covered in steps 1-3 is shown as not covered
  b  falsely visited    up to three uncovered cells with an object of the target's colour (never the
                        target) are shown as covered; none -> the episode is not counted in (b)
  c  false window       (reported, not a criterion) the channel "window last step" shows a window of
                        the same size elsewhere, not overlapping the true one, for the fourth command

"Aims there": the command's intended window overlaps at least one manipulated cell.
Measure per episode: how many of the next three commands (4th-6th, those actually given) aim there,
with the intervention minus without it, paired on the same bank episode.

Blandat schema (control): the agent receives another episode's schema (the donor). The same lie is
applied to what it receives: the cells are chosen from the agent's own episode, as for 1a; for (a)
a cell stays falsely unvisited until the donor's coverage after the intervention reaches it, exactly
as for 1a with its own coverage. Donor of bank episode i = the next episode in its block of 500
(cyclic), so no episode is its own donor. In the intervention run every eligible episode in the block
is intervened, donors included.
"""
import numpy as np
from env import SearchEnv, window_origin

BANK_SEED = 2_000_000          # test episode i: seed BANK_SEED + i (locked)
N_BANK = 4000
H1_SEED = 3_000_000            # H1 bank: seed H1_SEED + 10 000 k + i, i < 2 000 (technical choice)
H1_KS = (4, 8, 12, 16)
H1_N = 2000
BLOCK = 500


def bank_seeds(n=N_BANK, base=BANK_SEED):
    return [base + i for i in range(n)]


def h1_seeds(k, n=H1_N):
    return [H1_SEED + 10_000 * k + i for i in range(n)]


def h1_overrides(k):
    """8 x 8, window 1, k distractors, budget ceil(1.2 (k+1)); capture as standard (as in the pilot)."""
    return dict(n=8, w=1, k=k, budget=int(np.ceil(1.2 * (k + 1))))


def streams(seed):
    env_ss, agent_ss, h2_ss = np.random.SeedSequence(seed).spawn(3)
    return np.random.default_rng(env_ss), np.random.default_rng(agent_ss), np.random.default_rng(h2_ss)


class BankEnv(SearchEnv):
    """SearchEnv where every episode in the batch draws from its own stream."""

    def __init__(self, cfg):
        super().__init__(cfg, seed=0)
        self.env_rngs = None

    def reset_bank(self, env_rngs, overrides=None):
        """Start one episode per stream. Each is set up exactly as SearchEnv.reset(1) with that stream."""
        B = len(env_rngs)
        parts = []
        self.env_rngs = None                       # per-episode reset: base draws from self.rng
        for g in env_rngs:
            self.rng = g
            SearchEnv.reset(self, 1, overrides)
            parts.append({k: getattr(self, k) for k in self._STATE})
        for k in self._STATE:
            setattr(self, k, np.concatenate([p[k] for p in parts], 0))
        self.B = B
        self.t = 0
        self.env_rngs = list(env_rngs)
        return self.observe()

    _STATE = ("n", "w", "k", "budget", "color", "shape", "valid", "salient", "target", "target_color",
              "target_shape", "sal_pos", "sal_val", "steps", "done", "found", "visits", "last_actual",
              "last_window", "last_intended", "last_revealed", "sal_obs")

    def reset(self, B, overrides=None):
        raise RuntimeError("BankEnv: use reset_bank")

    def _normal_noise(self):
        if self.env_rngs is None:
            return super()._normal_noise()
        G = self.cfg.G
        return np.stack([g.normal(0, self.cfg.sal_noise, (G, G)) for g in self.env_rngs])

    def _uniform(self):
        if self.env_rngs is None:
            return super()._uniform()
        return np.array([g.random() for g in self.env_rngs])


def _window_mask(G, r0, c0, w):
    m = np.zeros((G, G), bool)
    m[r0:r0 + w, c0:c0 + w] = True
    return m


def run_bank(agent, seeds, mode=None, intervention=None, overrides=None, block=BLOCK, record=False):
    """Run the bank episodes `seeds` with the agent's stochastic policy, in blocks.

    mode          schema mode to use ('off' = no schema / lesion; default the agent's own)
    intervention  None, 'a', 'b' or 'c' (H2), applied in every eligible episode
    Returns per-episode arrays (length len(seeds)), see _run_block.
    """
    out = []
    for s in range(0, len(seeds), block):
        out.append(_run_block(agent, seeds[s:s + block], mode or agent.schema_mode, intervention,
                              overrides, record))
    res = {}
    for k in out[0]:
        if isinstance(out[0][k], np.ndarray):
            res[k] = np.concatenate([o[k] for o in out], 0)
        else:
            res[k] = sum((o[k] for o in out), [])
    return res


H2_T = 3          # the lie is in the schema before the 4th command (t = 3, counting from 0)
H2_SPAN = 3       # commands counted: t = 3, 4, 5


def _run_block(agent, seeds, mode, intervention, overrides, record):
    assert mode in ("off", "fixed", "shuffled"), mode
    cfg = agent.cfg
    G = cfg.G
    B = len(seeds)
    st = [streams(s) for s in seeds]
    env = BankEnv(cfg)
    obs = env.reset_bank([x[0] for x in st], overrides)
    arng = [x[1] for x in st]
    hrng = [x[2] for x in st]
    h = agent.init_state(B)
    donor = (np.arange(B) + 1) % B if B > 1 else np.zeros(1, int)
    src = donor if mode == "shuffled" else np.arange(B)

    vis = np.zeros((B, G, G))                 # true coverage so far (own episode)
    post = np.zeros((B, G, G))                # coverage since the intervention
    cur = np.zeros((B, G, G))                 # actual window last step
    A = np.zeros((B, G, G), bool)             # cells for (a)
    Bm = np.zeros((B, G, G), bool)            # cells for (b)
    Cm = np.zeros((B, G, G), bool)            # false window for (c)
    elig_a = np.zeros(B, bool)
    elig_b = np.zeros(B, bool)
    elig_c = np.zeros(B, bool)
    lie = np.zeros(B, bool)
    n_cand_b = np.zeros(B, int)
    aim_a = np.zeros(B, int)
    aim_b = np.zeros(B, int)
    aim_c = np.zeros(B, int)
    revisits = np.zeros(B, int)
    covered = np.zeros((B, G, G), bool)
    tcol = env.color == env.target_color[:, None, None]
    tmask = np.zeros((B, G, G), bool)
    tmask[np.arange(B), env.target[:, 0], env.target[:, 1]] = True
    trace = []
    schema = None
    t = 0
    while not env.done.all():
        if t == H2_T:
            # cells for the interventions, computed in every run (with or without the lie), from the
            # state after three steps; the h2 stream is drawn the same way in every run.
            running = ~env.done
            for b in range(B):
                cand = np.argwhere(tcol[b] & ~tmask[b] & (vis[b] == 0) & env.valid[b])
                n_cand_b[b] = len(cand)
                g = hrng[b]
                pick = g.permutation(len(cand))[:3] if len(cand) else []
                for j in pick:
                    Bm[b, cand[j][0], cand[j][1]] = True
                A[b] = vis[b] > 0
                # (c): a window of the same size, not overlapping the true last window
                n, w = env.n[b], env.w[b]
                r0, c0 = env.last_actual[b]
                opts = [(r, c) for r in range(n - w + 1) for c in range(n - w + 1)
                        if not (_window_mask(G, r, c, w) & _window_mask(G, r0, c0, w)).any()]
                k = int(g.integers(len(opts))) if opts else -1
                if k >= 0:
                    Cm[b] = _window_mask(G, opts[k][0], opts[k][1], w)
            elig_a = running.copy()
            elig_b = running & (n_cand_b > 0)
            elig_c = running & Cm.any((1, 2))
            if intervention == "a":
                lie = elig_a.copy()
            elif intervention == "b":
                lie = elig_b.copy()
            elif intervention == "c":
                lie = elig_c.copy()
        # schema for this command
        if mode == "off" or t == 0:
            schema = None
        else:
            sv = vis[src].copy()
            sc = cur[src].copy()
            if intervention == "a":
                sv[lie] = np.where(A[lie], post[src][lie], sv[lie])
            elif intervention == "b":
                sv[lie] = np.where(Bm[lie], 1.0, sv[lie])
            elif intervention == "c" and t == H2_T:
                sc[lie] = Cm[lie].astype(float)
            schema = np.stack([sc, sv], -1)
        idx = np.nonzero(~env.done)[0]
        x, m = agent.features(env, obs, t, schema)
        hs, logp, _, _, _ = agent.step(x[idx], m[idx], h[idx])
        h[idx] = hs
        a = np.zeros(B, int)
        for j, b in enumerate(idx):
            u = arng[b].random(logp.shape[1])
            a[b] = int((logp[j] - np.log(-np.log(u))).argmax())
        obs, r, done, info = env.step(a)
        ir0, ic0 = info["intended"].T
        r0, c0 = info["actual"].T
        newcur = np.zeros((B, G, G))
        for b in np.nonzero(info["active"])[0]:
            w = env.w[b]
            iw = _window_mask(G, ir0[b], ic0[b], w)
            if H2_T <= t < H2_T + H2_SPAN:
                aim_a[b] += bool((iw & A[b]).any())
                aim_b[b] += bool((iw & Bm[b]).any())
                aim_c[b] += bool((iw & Cm[b]).any())
            aw = _window_mask(G, r0[b], c0[b], w)
            if covered[b][aw].all():
                revisits[b] += 1
            covered[b] |= aw
            newcur[b] = aw
            if t >= H2_T:
                post[b] = np.maximum(post[b], aw)
        cur = newcur
        vis = np.maximum(vis, cur)
        if record:
            trace.append(dict(t=t, a=a.copy(), intended=info["intended"].copy(), actual=info["actual"].copy(),
                              active=info["active"].copy(), captured=info["captured"].copy(),
                              schema=None if schema is None else schema.copy()))
        t += 1
    res = dict(found=env.found.copy(), steps=env.steps.copy(), revisits=revisits, k=env.k.copy(),
               n=env.n.copy(), w=env.w.copy(),
               elig_a=elig_a, elig_b=elig_b, elig_c=elig_c, n_cells_b=n_cand_b,
               aim_a=aim_a, aim_b=aim_b, aim_c=aim_c)
    if record:
        res["trace"] = [trace]
        res["cells"] = [dict(A=A, B=Bm, C=Cm, target=env.target.copy(), tcol=tcol)]
    return res
