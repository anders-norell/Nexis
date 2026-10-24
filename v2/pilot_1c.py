"""Sprint 6 (1c) pilot evaluation, blind to where the probe's errors go.

  python3 pilot_1c.py eval      evaluate every finished c6_* run (final checkpoint) -> results/sprint6_pilot.json
  python3 pilot_1c.py level     apply the level rule and print the next phase
  python3 pilot_1c.py b L e|u   precondition (b) in the pilot: 1c vs the matched capacity baseline at level L
  python3 pilot_1c.py ref1b     the same probe (blind part) on 1b-E's pilot runs p5_ehalf* (seeds 20, 21)

Per run, on the pilot bank (episode i: seed 4 000 000 + i, i < 4 000, streams as in bank.py):
  found rate and SE; benefit against Baslinje of the same seed (difference, unpaired SE as in the MVP);
  bits per step, at steps after a capture and after none ("deviation code or map");
  collapse flag (under 1 bit per step); the critic's share of the gradient into the module, from the
  training log (mean over the last quarter of training); for the capacity baseline the share of steps whose
  donor is still running; for 1c with alarm a sanity check with the channel shuffled at test time;
  whether the run reaches its level (achieved bits within 20 % of the target);
  the probe (trained on the probe bank, seeds 7 000 000 + i, balanced) and its blind summary: accuracy
  without capture, and at capture steps with at least three candidates only (review point 4: with two the
  mass on the intended window is 1 - mass on actual) accuracy, (c) and whether q can be matched with the bar.

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
TARGET_TOL = 0.20                  # achieved bits within 20 % of the level (review point 5)


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
        tgt = a.get("rate_target") or 0.0
        if tgt > 0:                                  # review point 5: does the run reach its level?
            ratio = out["bits"]["bits_per_step"] / tgt
            out["bits_vs_target"] = dict(target=tgt, ratio=ratio, within_20pct=bool(abs(ratio - 1) <= TARGET_TOL))
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
    """Waits until every needed run is evaluated (review point 6). A level counts for a condition only if the
    benefit is > 2 SE in both seeds AND both runs reach the level (bits within 20 %)."""
    out, reach = {}, {}
    for cond in ("e", "u"):
        for L in LEVELS:
            out[f"{cond}{L}"] = [benefit(res, f"c6_{cond}{L}", s) for s in SEEDS]
            reach[f"{cond}{L}"] = [res["runs"].get(f"c6_{cond}{L}_s{s}", {}).get("bits_vs_target", {}).get("within_20pct")
                                   for s in SEEDS]
    ok = lambda key: all(b is not None and b["passed"] for b in out[key]) and all(reach[key])
    have = lambda cond: all(b is not None for L in LEVELS for b in out[f"{cond}{L}"])
    pick = lambda cond: next((L for L in reversed(LEVELS) if ok(f"{cond}{L}")), None)
    decision = dict(benefits=out, reaches_level=reach)
    if not have("e"):
        decision.update(level=None, next="wait: phase A not finished or not evaluated")
    elif pick("e") is not None:
        L = pick("e")
        decision.update(level=L, decided_by="with", next=f"sh run_sprint6.sh B {L} e")
    elif not have("u"):
        decision.update(level=None, next="sh run_sprint6.sh U (no benefit for 'with' at any level)")
    elif pick("u") is not None:
        L = pick("u")
        decision.update(level=L, decided_by="without", next=f"sh run_sprint6.sh B {L} u")
    else:
        decision.update(level=None, next="no benefit: collapse check (field 'collapse'), then sh run_sprint6.sh C, "
                                         "then D (phases C and D are judged by hand with benefit())")
    res["level_rule"] = decision
    json.dump(res, open(OUT, "w"), indent=1)
    return decision


def precondition_b(res, level, cond="e"):
    """Precondition (b) in the pilot (review point 5): 1c minus the matched capacity baseline at the same level,
    per seed (unpaired SE), and whether the capacity baseline reaches the level."""
    cap = f"cap{level}" if cond == "e" else f"capu{level}"
    rows = {}
    for s in SEEDS:
        a, b = res["runs"].get(f"c6_{cond}{level}_s{s}"), res["runs"].get(f"c6_{cap}_s{s}")
        if a is None or b is None:
            rows[s] = None
            continue
        d = a["found"] - b["found"]
        se = float(np.sqrt(a["found_se"] ** 2 + b["found_se"] ** 2))
        rows[s] = dict(diff=d, se=se, z=d / se, passed=bool(d > 2 * se),
                       cap_bits_vs_target=b.get("bits_vs_target"), cap_donor_running=b["log"].get("donor_running"))
    res["precondition_b"] = dict(level=level, cond=cond, per_seed=rows)
    json.dump(res, open(OUT, "w"), indent=1)
    return res["precondition_b"]


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"runs": {}}
    what = sys.argv[1] if len(sys.argv) > 1 else "eval"
    if what == "eval":
        for p in sorted(glob.glob(os.path.join(RES, "c6_*.final.npz"))):
            evaluate(os.path.basename(p)[:-len(".final.npz")], res)
    elif what == "level":
        print(json.dumps(level_rule(res), indent=1))
    elif what == "b":                      # python3 pilot_1c.py b <level> <e|u>
        print(json.dumps(precondition_b(res, sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "e"), indent=1))
    elif what == "ref1b":
        for L in LEVELS:
            for s in (20, 21):
                evaluate(f"p5_ehalf{L}_s{s}", res)


if __name__ == "__main__":
    main()
