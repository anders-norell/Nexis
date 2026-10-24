"""Sprint 6 (1c) pilot evaluation, blind to where the probe's errors go.

  python3 pilot_1c.py eval      evaluate every finished c6_* run (final checkpoint) -> results/sprint6_pilot.json
  python3 pilot_1c.py level     apply the level rule and print the next phase
  python3 pilot_1c.py ref1b     the same probe (blind part) on 1b-E's pilot runs p5_ehalf* (seeds 20, 21)

Per run, on the pilot bank (episode i: seed 4 000 000 + i, i < 4 000, streams as in bank.py):
  found rate and SE; benefit against Baslinje of the same seed (difference, unpaired SE as in the MVP);
  bits per step, at steps after a capture and after none ("deviation code or map");
  collapse flag (under 1 bit per step); the critic's share of the gradient into the module, from the
  training log (mean over the last quarter of training); for the capacity baseline the share of steps whose
  donor is still running; for 1c with alarm a sanity check with the channel shuffled at test time;
  the probe (trained on the probe bank, seeds 7 000 000 + i, balanced) and its blind summary: accuracy with
  and without capture, (c) mass on the actual window vs uniform at H5 events, whether q can be matched and
  the bar (null's mass on the intended window at that q).

Level rule (product owner, 24 Oct 23:45): the lowest level (fewest bits) at which 1c with alarm gives a
benefit > 2 SE in both pilot seeds. "Without" runs at the same level. If "with" has no benefit at any level,
"without" decides by the same rule. If neither: collapse check, then cost 0.0001, then capture risk doubled,
then stop.
"""
import glob
import json
import os
import sys
import numpy as np
from env import EnvConfig
from bank import bank_seeds
from schema_1c import make_agent_1c
import probe_1c as P

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = os.path.join(RES, "sprint6_pilot.json")
PILOT_BANK, N_PILOT = 4_000_000, 4000
LEVELS = ("44.2", "22.1", "11.05")
SEEDS = (40, 41)
COLLAPSE_BITS = 1.0


def load(path):
    z = np.load(path)
    a = json.loads(str(z["m_args"]))
    cfg = EnvConfig(reveal=a["reveal"], p_capture=a["p_capture"],
                    windows=tuple(int(x) for x in str(a["windows"]).split(",")))
    if a["schema"] == "1c":
        ag = make_agent_1c(cfg, d=a["d"], dh=a["dh"], seed=a["seed"], alarm=a["alarm"], shuffle=a["shuffle"])
        kind = "1c"
    else:
        from agent_rl import make_agent
        ag = make_agent(cfg, d=a["d"], dh=a["dh"], seed=a["seed"], schema_mode=a["schema"])
        kind = "1b" if a["schema"] in ("channel", "channel_e") else "off"
    meta = ag.load(path)
    if hasattr(ag, "module") and "beta" in meta:
        ag.module.beta = float(meta["beta"])
    return ag, a, kind


def log_summary(name):
    """Critic share and donor share over the last quarter of training, from the log."""
    path = os.path.join(RES, name + ".log.jsonl")
    if not os.path.exists(path):
        return {}
    rows = [json.loads(l) for l in open(path)]
    if not rows:
        return {}
    last = max(r.get("episodes", 0) for r in rows)
    late = lambda r: r.get("episodes", 0) >= 0.75 * last
    d = [r["diag"] for r in rows if "diag" in r and late(r)]
    out = {}
    if d:
        out["critic_share"] = float(np.mean([x["critic_share"] for x in d]))
        out["critic_actor_cos"] = float(np.mean([x["cos"] for x in d]))
        out["module_grad_norm_policy"] = float(np.mean([x["total_norm"] for x in d]))
    dr = [r["donor_running"] for r in rows if "donor_running" in r and late(r)]
    if dr:
        out["donor_running"] = float(np.mean(dr))
    tm = [r["minutes"] for r in rows if "minutes" in r]
    if tm:
        out["minutes"] = tm[-1]
    b = [r["beta"] for r in rows if "beta" in r and late(r)]
    if b:
        out["beta_late"] = float(np.mean(b))
    return out


def evaluate(name, res, force=False):
    if name in res["runs"] and not force:
        return res["runs"][name]
    ag, a, kind = load(os.path.join(RES, name + ".final.npz"))
    seeds = bank_seeds(N_PILOT, PILOT_BANK)
    r = P.collect(ag, kind, seeds, with_events=kind != "off")
    out = dict(kind=kind, seed=a["seed"], alarm=a.get("alarm"), shuffle=a.get("shuffle"),
               rate_target=a.get("rate_target"), rate_cost=a.get("rate_cost"), p_capture=a["p_capture"],
               **P.found_summary(r), log=log_summary(name))
    if kind != "off":
        out["bits"] = P.bits_summary(r)
        out["collapse"] = bool(out["bits"]["bits_per_step"] < COLLAPSE_BITS)
        pb, info = P.train_probe(ag, kind)
        out["probe"] = dict(**info, **P.blind_summary(pb, r))
        if kind == "1c" and a.get("alarm") and not a.get("shuffle"):
            out["test_shuffle_found"] = P.found_summary(P.collect(ag, kind, seeds, test_shuffle=True,
                                                                  with_events=False))["found"]
    res["runs"][name] = out
    json.dump(res, open(OUT, "w"), indent=1)
    show = {k: v for k, v in out.items() if k not in ("probe", "log")}
    print(name, json.dumps(show), flush=True)
    if "probe" in out:
        print("   probe", json.dumps(out["probe"]), flush=True)
    return out


def benefit(res, name, seed, base="off"):
    r, b = res["runs"].get(f"{name}_s{seed}"), res["runs"].get(f"c6_{base}_s{seed}")
    if r is None or b is None:
        return None
    d = r["found"] - b["found"]
    se = float(np.sqrt(r["found_se"] ** 2 + b["found_se"] ** 2))
    return dict(diff=d, se=se, z=d / se, passed=bool(d > 2 * se))


def level_rule(res):
    out = {}
    for cond in ("e", "u"):
        for L in LEVELS:
            out[f"{cond}{L}"] = [benefit(res, f"c6_{cond}{L}", s) for s in SEEDS]
    pick = lambda cond: next((L for L in reversed(LEVELS)
                              if all(b is not None and b["passed"] for b in out[f"{cond}{L}"])), None)
    le = pick("e")
    lu = pick("u")
    decision = dict(benefits=out, level_with=le, level_without=lu)
    if le is not None:
        decision.update(level=le, decided_by="with", next=f"sh run_sprint6.sh B {le}")
    elif all(all(b is not None for b in out[f"u{L}"]) for L in LEVELS):
        decision.update(level=lu, decided_by="without" if lu else None,
                        next=f"sh run_sprint6.sh B {lu}" if lu else "collapse check, then sh run_sprint6.sh C")
    else:
        decision.update(level=None, next="sh run_sprint6.sh U (no benefit for 'with' at any level)")
    res["level_rule"] = decision
    json.dump(res, open(OUT, "w"), indent=1)
    return decision


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"runs": {}}
    what = sys.argv[1] if len(sys.argv) > 1 else "eval"
    if what == "eval":
        for p in sorted(glob.glob(os.path.join(RES, "c6_*.final.npz"))):
            evaluate(os.path.basename(p)[:-len(".final.npz")], res)
    elif what == "level":
        print(json.dumps(level_rule(res), indent=1))
    elif what == "ref1b":
        for L in LEVELS:
            for s in (20, 21):
                evaluate(f"p5_ehalf{L}_s{s}", res)


if __name__ == "__main__":
    main()
