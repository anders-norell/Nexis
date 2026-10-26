"""Sprint 6, step 0: evaluate the direct-alarm policy against the main run (pre-registered 26 Oct, see
results/sprint6_step0_prereg.json).

  python3 eval_step0.py   -> results/sprint6_step0.json

Same test bank as the main run (MVP bank, seed 2 000 000 + i, 4 000 episodes), final checkpoints, unpaired SE.
Rule (pre-registered): if, in every seed, the direct-alarm policy is within 2 SE of 1c with alarm or above it, the
module added nothing beyond the comparator signal. Reported also: direct alarm minus Baslinje.
"""
import json
import os
import numpy as np
from env import EnvConfig
from agent_rl import PriorityMapAgent
from bank import bank_seeds
import probe_1c as P

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
SEEDS = (30, 31, 32)


def load(path):
    z = np.load(path)
    a = json.loads(str(z["m_args"]))
    assert a.get("direct_alarm") and abs(a["p_capture"] - 0.5) < 1e-9, path
    assert int(z["m_episodes"]) >= 1_000_000, path
    cfg = EnvConfig(reveal=a["reveal"], p_capture=a["p_capture"],
                    windows=tuple(int(x) for x in str(a["windows"]).split(",")))
    ag = PriorityMapAgent(cfg, d=a["d"], dh=a["dh"], seed=a["seed"], schema_mode="off", matching=True, alarm=True)
    ag.load(path)
    return ag


def diff(a, b):
    d = a["found"] - b["found"]
    se = float(np.sqrt(a["found_se"] ** 2 + b["found_se"] ** 2))
    return dict(diff=d, se=se, z=d / se)


def main():
    main_res = json.load(open(os.path.join(RES, "sprint6_main.json")))["runs"]
    out = dict(runs={}, comparison={})
    for s in SEEDS:
        ag = load(os.path.join(RES, f"c6_mL-alarmdirect_s{s}.final.npz"))
        r = P.collect(ag, "off", bank_seeds(4000, 2_000_000), with_events=False)
        f = P.found_summary(r)
        out["runs"][s] = f
        e, o = main_res[f"c6_mL-e1e-4_s{s}"], main_res[f"c6_mL-off_s{s}"]
        vs_1c = diff(f, e)
        out["comparison"][s] = dict(vs_1c_with_alarm=vs_1c, vs_baseline=diff(f, o),
                                    within_2se_or_above=bool(vs_1c["diff"] > -2 * vs_1c["se"]))
        print(s, json.dumps(out["comparison"][s]), flush=True)
    out["module_added_nothing"] = all(c["within_2se_or_above"] for c in out["comparison"].values())
    json.dump(out, open(os.path.join(RES, "sprint6_step0.json"), "w"), indent=1)
    print("module_added_nothing:", out["module_added_nothing"])


if __name__ == "__main__":
    main()
