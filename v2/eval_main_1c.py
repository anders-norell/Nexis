"""Sprint 6 (1c) main run: judgement against the criteria locked at gate 6a (results/h5_1c_lock.json).

  python3 eval_main_1c.py lock "<approval text>"   write the lock (before any main-run training)
  python3 eval_main_1c.py run                        evaluate every finished main run -> results/sprint6_main.json

Runs (run_sprint6.sh M): results/c6_mL-{off,e1e-4,u1e-4,cap1e-4,1bE}_s{30,31,32}, capture risk 0.5, cost 0.0001,
1 000 000 episodes, final checkpoints. Test bank: the MVP bank (episode i: seed 2 000 000 + i, i < 4 000), with the
environment of the run (capture risk 0.5). Probes: probe bank (7 000 000 + i, 4 000 episodes, balanced), one primary
probe (channel at t + 1) and one sensitivity probe (t + 1 and t) per run.

Criteria (locked; every comparison per seed, unpaired SE over test episodes as in the MVP unless stated):
  (a) benefit      1c minus Baslinje, found rate, > 2 SE in all three seeds
  (b) capacity     1c with alarm minus the capacity baseline, > 2 SE in all three seeds (for 1c without alarm the
                   capacity baseline with alarm is the comparison as well; it has more parameters, so passing is
                   conservative)
  (c) self-reference  primary probe's mass on the actual window minus 1 / K at H5 events (all K), > 2 SE in all seeds
                   (bootstrap over episodes)
  H5 "om"          only for a 1c condition that passes (a), (b) and (c): role-swap control of the primary probe
                   (probe trained on capture steps only; mass on the intended window minus on the decoy, another
                   wrong salient candidate, at H5 events with a decoy), > 2 SE in all three seeds (bootstrap over
                   episodes). Calibrated (test_probe_1c.py): no synthetic channel without role information passes,
                   passive transmission of the intended window's input comes out negative.
  AST-specific     only if "om" passes for 1c with alarm: its control minus 1b-E's, > 2 SE in all three seeds
                   (SE = root of the sum of the two bootstrap variances)
  Reported: the same control for an untrained shadow module (same architecture and seed, run on the same episodes,
  its channel not fed to the policy) and the trained-minus-shadow difference, paired per event.
Reported, not judged: the null-adjusted "om" and the control with the null, the sensitivity probe's control, the split
by whether command t + 1 aims at the intended window again (a yes is read as AST only if the no-repeat group points
the same way), "var", bits after capture / no capture, test-time shuffled channel, the critic's share.
"""
import hashlib
import json
import os
import sys
import numpy as np
from bank import bank_seeds
import probe_1c as P
from pilot_1c import load, log_summary
from env import EnvConfig

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = os.path.join(RES, "sprint6_main.json")
SEEDS = (30, 31, 32)
CONDS = ("off", "e1e-4", "u1e-4", "cap1e-4", "1bE")
BANK, N_BANK = 2_000_000, 4000
PARAMS = dict(seeds=list(SEEDS), conditions=list(CONDS), p_capture=0.5, cost=0.0001, episodes=1_000_000,
              test_bank=[BANK, N_BANK], probe_bank=[P.PROBE_BANK, P.N_PROBE], n_boot=P.N_BOOT,
              primary_cols=P.COLS_PRIMARY, sens_cols=P.COLS_SENS, checkpoint="final")


def name(cond, s):
    return f"c6_mL-{cond}_s{s}"


def diff(a, b):
    d = a["found"] - b["found"]
    se = float(np.sqrt(a["found_se"] ** 2 + b["found_se"] ** 2))
    return dict(diff=d, se=se, z=d / se, passed=bool(d > 2 * se))


def checkpoint_ok(path, a, meta, cond):
    """Pre-lock review finding 3: the run must be the decided one (capture risk, cost, length, seed)."""
    errs = []
    if abs(a["p_capture"] - 0.5) > 1e-9:
        errs.append("p_capture")
    if int(meta.get("episodes", 0)) < 1_000_000:
        errs.append(f"episodes {meta.get('episodes')}")
    if cond != "off" and abs(a.get("rate_cost", 0) - 0.0001) > 1e-12:
        errs.append("rate_cost")
    if cond != "off" and a.get("rate_target", 0) not in (0, 0.0):
        errs.append("rate_target")
    if cond == "cap1e-4" and not (a.get("shuffle") and a.get("alarm")):
        errs.append("capacity flags")
    if cond == "e1e-4" and not (a.get("alarm") and not a.get("shuffle")):
        errs.append("alarm flags")
    if cond == "u1e-4" and (a.get("alarm") or a.get("shuffle")):
        errs.append("no-alarm flags")
    if cond == "1bE" and a.get("schema") != "channel_e":
        errs.append("1b-E schema")
    if errs:
        raise RuntimeError(f"{path}: not the decided run: {errs}")


def load_untrained(path):
    """The same architecture, configuration and seed as the run, at its initial weights (not loaded)."""
    z = np.load(path)
    a = json.loads(str(z["m_args"]))
    cfg = EnvConfig(reveal=a["reveal"], p_capture=a["p_capture"],
                    windows=tuple(int(x) for x in str(a["windows"]).split(",")))
    if a["schema"] == "1c":
        from schema_1c import make_agent_1c
        ag = make_agent_1c(cfg, d=a["d"], dh=a["dh"], seed=a["seed"], alarm=a["alarm"], shuffle=a["shuffle"])
        ag.module.beta = a["rate_cost"]
        return ag, "1c"
    from agent_rl import make_agent
    ag = make_agent(cfg, d=a["d"], dh=a["dh"], seed=a["seed"], schema_mode=a["schema"])
    return ag, "1b"


def control_diff(x, y):
    d = x["value"] - y["value"]
    se = float(np.sqrt(x["se"] ** 2 + y["se"] ** 2))
    return dict(diff=d, se=se, z=d / se if se > 0 else float("nan"), passed=bool(d > 2 * se))


def evaluate_run(n, res):
    path = os.path.join(RES, n + ".final.npz")
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    if n in res["runs"] and res["runs"][n].get("checkpoint_sha256") == sha:     # review finding 4
        return res["runs"][n]
    ag, a, kind = load(path)
    z = np.load(path)
    meta = {k[2:]: z[k] for k in z.files if k.startswith("m_")}
    checkpoint_ok(path, a, meta, n.split("mL-")[1].rsplit("_s", 1)[0])
    seeds = bank_seeds(N_BANK, BANK)
    shadow = load_untrained(path)[0].module if kind != "off" else None
    r = P.collect(ag, kind, seeds, with_events=kind != "off", shadow=shadow)
    out = dict(kind=kind, checkpoint_sha256=sha, **P.found_summary(r), log=log_summary(n))
    if kind != "off":
        out["bits"] = P.bits_summary(r)
        pbs, info = P.train_probes(ag, kind, shadow=shadow)
        out["probe_info"] = info
        out["c_all"] = P.c_all(pbs["primary"], r)
        out["h5_primary"] = P.measure(pbs["primary"], r)
        out["h5_sens"] = P.measure(pbs["sens"], r)
        out["shadow_h5_primary"] = P.measure(pbs["shadow_primary"], r, fk="F_sh")
        out["trained_minus_shadow"] = P.measure_pair(pbs["primary"], pbs["shadow_primary"], r)
        out["event_counts"] = dict(h5=len([e for e in r["events"] if e["h5"]]),
                                   h5_with_decoy=len([e for e in r["events"] if e["h5"] and e["decoy"] is not None]),
                                   per_episode=len([e for e in r["events"] if e["h5"]]) / N_BANK)
        if kind == "1c" and not ag.module.shuffle:
            out["test_shuffle_found"] = P.found_summary(P.collect(ag, kind, seeds, test_shuffle=True,
                                                                  with_events=False))["found"]
    res["runs"][n] = out
    json.dump(res, open(OUT, "w"), indent=1)
    print(n, json.dumps({k: v for k, v in out.items() if k not in ("log", "probe_info")})[:2000], flush=True)
    return out


def judge(res):
    R = res["runs"]
    have = lambda c: all(name(c, s) in R for s in SEEDS)
    J = {}
    for c in ("e1e-4", "u1e-4"):
        if not (have(c) and have("off") and have("cap1e-4")):
            continue
        a = [diff(R[name(c, s)], R[name("off", s)]) for s in SEEDS]
        b = [diff(R[name(c, s)], R[name("cap1e-4", s)]) for s in SEEDS]
        cc = [R[name(c, s)]["c_all"] for s in SEEDS]
        pre = all(x["passed"] for x in a) and all(x["passed"] for x in b) and all(x["passed"] for x in cc)
        ctrl = [R[name(c, s)]["h5_primary"]["control"] for s in SEEDS]
        J[c] = dict(a=a, b=b, c=cc, preconditions=pre, om_control=ctrl,
                    trained_minus_shadow=[R[name(c, s)]["trained_minus_shadow"] for s in SEEDS],
                    om_passed=bool(pre and all(x["passed"] for x in ctrl)),
                    norepeat=[R[name(c, s)]["h5_primary"]["split"]["norepeat"]["control"] for s in SEEDS],
                    sens_control=[R[name(c, s)]["h5_sens"]["control"] for s in SEEDS])
    if "e1e-4" in J and have("1bE"):
        sp = []
        for s in SEEDS:
            x, y = R[name("e1e-4", s)]["h5_primary"]["control"], R[name("1bE", s)]["h5_primary"]["control"]
            d = x["value"] - y["value"]
            se = float(np.sqrt(x["se"] ** 2 + y["se"] ** 2))
            sp.append(dict(diff=d, se=se, z=d / se if se > 0 else float("nan"), passed=bool(d > 2 * se)))
        J["ast_specific"] = dict(per_seed=sp, judged=J["e1e-4"]["om_passed"],
                                 passed=bool(J["e1e-4"]["om_passed"] and all(x["passed"] for x in sp)))
    res["judgement"] = J
    json.dump(res, open(OUT, "w"), indent=1)
    return J


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "run"
    if what == "lock":
        P.write_lock(sys.argv[2], dict(params=PARAMS))
        print("locked", P.LOCK)
        return
    P._check_lock()
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"runs": {}}
    for c in CONDS:
        for s in SEEDS:
            if os.path.exists(os.path.join(RES, name(c, s) + ".final.npz")):
                evaluate_run(name(c, s), res)
    print(json.dumps(judge(res), indent=1))


if __name__ == "__main__":
    main()
