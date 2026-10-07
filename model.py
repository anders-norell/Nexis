"""Agents for the Nexis PoC (Sprint 1 baseline; Sprint 2 adds the attention schema).

Per step t:
  logits_att = R*tanh(query(h_{t-1})/R) + BETA * flash_t  (bounded top-down + bottom-up)
  a_t        = softmax(logits_att)                     (16 patches)
  glimpse_t  = sum_i a_t[i] * fovea_t[i]               (4 pixels, location-free)
  s_t        = schema([top-down_t, flash_t], s_{t-1})    (Sprint 2; predicts a_t)
  h_t        = GRU([periphery_t, glimpse_t, s_t], h_{t-1})   (Sprint 1 also fed flash_t)
  guess_t    = readout(h_t)                            (64-way target pixel)

The controller never sees a_t directly; it only knows what it asked for
(through h) and what came back (glimpse). From Sprint 2 it also does not see
the flash, so only the schema can tell it that attention was pulled away.
"""
import numpy as np
from autodiff import T, Param, tanh, sigmoid, softmax, weighted_sum, concat, cross_entropy, soft_cross_entropy, mean
from env import N_PATCH, T_STEPS, FIELD

BETA = 10.0        # strength of bottom-up capture
QUERY_RANGE = 4.0  # top-down logits are bounded to +-QUERY_RANGE, so a flash can win

# v2: maps 16 patches (4x4 grid) to 4 quadrants
_q = np.zeros((N_PATCH, 4))
for _p in range(N_PATCH):
    _q[_p, (_p // 4 // 2) * 2 + (_p % 4) // 2] = 1.0
QUAD = T(_q)


def init(rng, n_in, n_out, scale=None):
    scale = scale or 1.0 / np.sqrt(n_in)
    return Param(rng.normal(0, scale, size=(n_in, n_out)))


class GRU:
    def __init__(self, rng, n_in, n_h):
        self.Wz, self.Wr, self.Wn = (init(rng, n_in, n_h) for _ in range(3))
        self.Uz, self.Ur, self.Un = (init(rng, n_h, n_h) for _ in range(3))
        self.bz, self.br, self.bn = (Param(np.zeros((1, n_h))) for _ in range(3))
        self.n_h = n_h

    def params(self):
        return [self.Wz, self.Wr, self.Wn, self.Uz, self.Ur, self.Un, self.bz, self.br, self.bn]

    def __call__(self, x, h):
        z = sigmoid(x @ self.Wz + h @ self.Uz + self.bz)
        r = sigmoid(x @ self.Wr + h @ self.Ur + self.br)
        n = tanh(x @ self.Wn + (r * h) @ self.Un + self.bn)
        return (T(1.0) - z) * n + z * h


class Agent:
    """One agent; the experimental conditions are switches.

    attention     : 'learned' (top-down query) or 'uniform' (Sprint 1 sanity control)
    flash_to_ctrl : controller sees the flash as input (Sprint 1 only)
    schema_width  : 0 = no schema; k > 0 = attention schema with a k-unit state
    ablate        : feed the schema's mean state to the controller instead of the live one

    The schema is a small GRU. Each step it sees the top-down query (what the
    agent asked for) and the flash (bottom-up salience), predicts the attention
    map that results, and is trained on the real attention map. Its state is fed
    to the controller, so the controller can use it to steer.
    """

    def __init__(self, seed=0, n_h=64, attention="learned", gain=1.0,
                 flash_to_ctrl=True, schema_width=0, v2=False, trace_to="none"):
        rng = np.random.default_rng(seed)
        self.attention = attention
        self.gain = gain
        self.flash_to_ctrl = flash_to_ctrl
        self.k = schema_width
        self.ablate = False
        self.schema_mean = None
        self.v2, self.trace_to = v2, trace_to
        assert trace_to in ("none", "schema", "ctrl")
        n_in = N_PATCH + 4 + (N_PATCH if flash_to_ctrl else 0) + self.k + (4 if trace_to == "ctrl" else 0)
        self.ctrl = GRU(rng, n_in, n_h)
        self.Wq = init(rng, n_h, N_PATCH, scale=0.1)
        self.bq = Param(np.zeros((1, N_PATCH)))
        self.Wo = init(rng, n_h, FIELD * FIELD)
        self.bo = Param(np.zeros((1, FIELD * FIELD)))
        # belief about which patch holds the target; attention follows it
        self.Wp = init(rng, n_h, N_PATCH)
        self.bp = Param(np.zeros((1, N_PATCH)))
        if self.k:
            # v1 schema sees query + flash; v2 schema sees query + coarse trace
            self.schema = GRU(rng, N_PATCH + (4 if v2 else N_PATCH), self.k)
            self.Wa = init(rng, self.k, N_PATCH)
            self.ba = Param(np.zeros((1, N_PATCH)))
        self.n_h = n_h

    def params(self):
        ps = self.ctrl.params() + [self.Wq, self.bq, self.Wo, self.bo, self.Wp, self.bp]
        if self.k:
            ps += self.schema.params() + [self.Wa, self.ba]
        return ps

    def n_params(self):
        return sum(p.v.size for p in self.params())

    def forward(self, batch):
        B = batch["periphery"].shape[0]
        h = T(np.zeros((B, self.n_h)))
        s = T(np.zeros((B, self.k))) if self.k else None
        belief = T(np.zeros((B, N_PATCH)))
        out = dict(att=[], guess=[], loss_terms=[], schema_pred=[], schema_state=[], intended=[])
        for t in range(T_STEPS):
            flash = T(batch["flash"][:, t])
            if self.attention == "uniform":
                a = T(np.full((B, N_PATCH), 1.0 / N_PATCH))
                top_down = T(np.zeros((B, N_PATCH)))
            else:
                # top-down = belief from last step + free query; bottom-up = flash
                query = belief * self.gain + h @ self.Wq + self.bq
                top_down = tanh(query * (1.0 / QUERY_RANGE)) * QUERY_RANGE
                logits_att = top_down + flash * BETA
                if self.v2:
                    logits_att = logits_att + T(batch["drift"][:, t])  # internal drift
                a = softmax(logits_att)
            # input order kept as in Sprint 1: periphery, [flash], glimpse, [schema]
            parts = [T(batch["periphery"][:, t])]
            if self.flash_to_ctrl:
                parts.append(flash)
            parts.append(weighted_sum(a, T(batch["fovea"][:, t])))
            if self.v2:
                trace = a @ QUAD + T(batch["trace_noise"][:, t])  # coarse view of actual attention
                if self.trace_to == "ctrl":
                    parts.append(trace)
            if self.k:
                s_in = concat([top_down, trace]) if self.v2 else concat([top_down, flash])
                s = self.schema(s_in, s)
                pred = s @ self.Wa + self.ba
                # the schema learns to model attention; it does not change attention itself
                out["loss_terms"].append(soft_cross_entropy(pred, a.v) * 0.5)
                out["schema_pred"].append(softmax(pred).v)
                out["schema_state"].append(s.v)
                if self.ablate:
                    parts.append(T(np.broadcast_to(self.schema_mean, (B, self.k)).copy()))
                else:
                    parts.append(s)
            h = self.ctrl(concat(parts), h)
            logits = h @ self.Wo + self.bo
            belief = h @ self.Wp + self.bp
            out["att"].append(a.v)
            out["intended"].append(top_down.v)
            out["guess"].append(logits.v.argmax(-1))
            out["final_term"] = cross_entropy(logits, batch["target_pixel"][:, t])
            out["loss_terms"].append(out["final_term"])
            out["loss_terms"].append(cross_entropy(belief, batch["target_patch"][:, t]) * 0.5)
        # all steps contribute; the final pixel report counts double
        out["loss"] = mean(out["loss_terms"] + [out["final_term"]])
        out["att"] = np.stack(out["att"], 1)      # (B,T,16)
        out["guess"] = np.stack(out["guess"], 1)  # (B,T)
        out["intended"] = np.stack(out["intended"], 1)  # (B,T,16) top-down logits
        if self.k:
            out["schema_pred"] = np.stack(out["schema_pred"], 1)
            out["schema_state"] = np.stack(out["schema_state"], 1)
        return out
