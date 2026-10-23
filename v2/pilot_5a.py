"""Sprint 5a pilot evaluation (blind to where the schema's errors go).

Runs (results/p5_*): Baslinje p5_off_s{20,21}; 1b with the costly channel:
  p5_ref1e-3_s{20,21}   fixed cost 0.001 per bit and step (the reference, condition b)
  p5_ref1e-4_s{20,21}   fixed cost 0.0001 (near cost-free: the precondition variant)
  p5_lvl{50,25,12.5}_s{20,21}   cost regulated to 50 / 25 / 12.5 % of the reference's information

Test on the pilot bank: episode i has seed 4 000 000 + i, i < 4 000 (streams as in bank.py).

  python3 pilot_5a.py eref    the same for alternative E (runs p5_eref*, global alarm)
  python3 pilot_5a.py ref     evaluate the reference runs, decide the reference (condition b), print the
                              level targets in bits per step
  python3 pilot_5a.py all     evaluate every finished run, write results/sprint5a_pilot.json

Reported per run: hit rate (schema's use, vs Baslinje), information per step, accuracy over all steps,
number of H5 events, accuracy at H5 events (map's most probable cell in the actual window; precondition
and "the bottleneck bites"), the restricted observer's mass on the actual window at the same events, and
the behaviour measure's dose check as yes/no (never the counts).
"""
import glob
import json
import os
import sys
import numpy as np
from env import EnvConfig
from agent_rl import make_agent
from bank import bank_seeds
from h5_bank import run_bank_module, summary

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
PILOT_BANK = 4_000_000
N_PILOT = 4000
OUT = os.path.join(RES, "sprint5a_pilot.json")
LEVELS = (0.5, 0.25, 0.125)


def load(path):
    z = np.load(path)
    a = json.loads(str(z["m_args"]))
    cfg = EnvConfig(reveal=a["reveal"], p_capture=a["p_capture"],
                    windows=tuple(int(x) for x in str(a["windows"]).split(",")))
    ag = make_agent(cfg, d=a["d"], dh=a["dh"], seed=a["seed"], schema_mode=a["schema"])
    meta = ag.load(path)
    if hasattr(ag, "module") and "beta" in meta:
        ag.module.beta = float(meta["beta"])
    return ag, a, {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in meta.items() if k != "args"}


def evaluate(name, res):
    if name in res["runs"]:
        return res["runs"][name]
    ag, a, meta = load(os.path.join(RES, name + ".best.npz"))
    r = run_bank_module(ag, bank_seeds(N_PILOT, PILOT_BANK))
    out = dict(schema=a["schema"], seed=a["seed"], rate_cost=a.get("rate_cost", 0.0),
               rate_target=a.get("rate_target", 0.0), checkpoint=meta, **summary(r),
               dose_groups=r["dose_groups"])
    res["runs"][name] = out
    json.dump(res, open(OUT, "w"), indent=1)
    print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items() if k != "checkpoint"},
          flush=True)
    return out


def finished(name):
    return os.path.exists(os.path.join(RES, name + ".final.npz"))


def decide_reference(res, prefix="ref"):
    """Condition b (locked): reference = information per step of the 0.001 variant, mean over the two
    seeds. If it differs from the 0.0001 variant by more than a factor 1.5, the reference is not well
    defined and the halving rule applies instead."""
    r3 = [evaluate(f"p5_{prefix}1e-3_s{s}", res)["bits_per_step"] for s in (20, 21)]
    r4 = [evaluate(f"p5_{prefix}1e-4_s{s}", res)["bits_per_step"] for s in (20, 21)]
    pre = [res["runs"][f"p5_{prefix}1e-4_s{s}"]["acc_at_h5"] for s in (20, 21)]
    m3, m4 = float(np.mean(r3)), float(np.mean(r4))
    ratio = max(m3, m4) / max(min(m3, m4), 1e-9)
    ref = dict(bits_1e3=r3, bits_1e4=r4, mean_1e3=m3, mean_1e4=m4, ratio=ratio,
               well_defined=bool(ratio <= 1.5), precondition_acc_at_h5=pre,
               precondition_met=bool(all(a >= 0.5 for a in pre)))
    if ref["well_defined"]:
        ref["reference_bits"] = m3
        ref["targets"] = {f"lvl{int(l * 1000) / 10:g}": m3 * l for l in LEVELS}
    res["reference" if prefix == "ref" else f"reference_{prefix}"] = ref
    json.dump(res, open(OUT, "w"), indent=1)
    return ref


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"runs": {}}
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("ref", "eref"):
        ref = decide_reference(res, what)
        print("REFERENCE", json.dumps(ref), flush=True)
        return
    for p in sorted(glob.glob(os.path.join(RES, "p5_*.final.npz"))):
        evaluate(os.path.basename(p)[:-len(".final.npz")], res)


if __name__ == "__main__":
    main()
