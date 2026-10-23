"""H5 on 1b, gate 5a (product owner's decisions 24 Oct 2026). A descriptive test, not a main run: H5 is run
once on the pilot's E runs (seeds 20, 21) and reported. 5b is not run.

  python3 h5_5a.py feasibility   BLIND. For each run, which matching works (mass on the actual window, else
                                 most probable cell, else not assessable). Uses only quantities about the
                                 actual window. Writes results/sprint5a_h5_feasibility.json.
  python3 h5_5a.py lock "<approval>"   ONLY after the product owner has approved the locked thresholds: writes
                                 results/h5_lock.json with SHA-256 of the code and the feasibility file, the
                                 fixed parameters and the approval text.
  python3 h5_5a.py measure       UNBLINDED. Refuses to run unless results/h5_lock.json exists (written
                                 only after the product owner has approved the locked thresholds). Writes
                                 results/sprint5a_h5.json.

Runs (final checkpoints, review point 9): levels by the mean reading of the reference (44.2 bits):
22.1 and 11.05 bits judged, 5.53 bits reported; the no-cost variant (cost 0.0001) and the reference level
44.2 bits reported. Test bank: the MVP bank, episode i has seed 2 000 000 + i, i < 4 000 (these agents
have never been tested on it; the pilot bank 4 000 000+ was used for the design decisions).
"""
import hashlib
import json
import os
import sys
import numpy as np
from bank import bank_seeds, BANK_SEED, N_BANK
from h5_bank import run_bank_module
from pilot_5a import load
import h5_eval as H

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
LOCK = os.path.join(RES, "h5_lock.json")
SEEDS = (20, 21)
JUDGED = {"22.1": "p5_ehalf22.1", "11.05": "p5_ehalf11.05"}
REPORTED = {"5.53": "p5_ehalf5.53", "44.2 (referens)": "p5_ehalf44.2", "utan kostnad (0,0001)": "p5_eref1e-4"}
LEVELS = {**JUDGED, **REPORTED}


def events_for(run, blind):
    ag, a, meta = load(os.path.join(RES, run + ".final.npz"))
    r = run_bank_module(ag, bank_seeds(N_BANK, BANK_SEED), blind=blind, detail=True)
    hit = float(r["found"].mean())
    bits = float((r["bits"] * r["bits_n"]).sum() / max(r["bits_n"].sum(), 1))
    return r["events"], dict(hit=hit, bits_per_step=bits, n_h5_events_all=len(r["events"]))


def feasibility():
    out = dict(bank=f"{BANK_SEED} + i, i < {N_BANK}", checkpoint="final", runs={}, levels={})
    for lvl, stem in LEVELS.items():
        per_seed = []
        for s in SEEDS:
            run = f"{stem}_s{s}"
            ev, info = events_for(run, blind=True)
            f = H.feasibility(H.build(ev))
            f.update(info)
            out["runs"][run] = f
            per_seed.append(f)
            print(run, json.dumps({k: (v if not isinstance(v, float) else round(v, 4)) for k, v in f.items()}),
                  flush=True)
        out["levels"][lvl] = dict(judged=lvl in JUDGED, method=H.level_method(per_seed))
        json.dump(out, open(os.path.join(RES, "sprint5a_h5_feasibility.json"), "w"), indent=1)
    print("LEVELS", json.dumps(out["levels"]), flush=True)


FILES = ("h5_eval.py", "h5_5a.py", "h5_bank.py", "h5.py", "env.py", "bank.py", "pilot_5a.py", "agent_rl.py",
         "schema_channel.py", "results/sprint5a_h5_feasibility.json") + tuple(
    f"results/{stem}_s{s}.final.npz" for stem in sorted(set(LEVELS.values())) for s in SEEDS)


def fingerprint():
    return dict(sha256={f: hashlib.sha256(open(os.path.join(HERE, f), "rb").read()).hexdigest() for f in FILES},
                params=dict(QS=len(H.QS), K_NULL=H.K_NULL, N_BOOT=H.N_BOOT, boot_seeds=[1000, 1001],
                            DROP_SHARED=H.DROP_SHARED, EQUALIZE_DIFFUSE=H.EQUALIZE_DIFFUSE, CLASS_DIFFUSE=H.CLASS_DIFFUSE,
                            PLATEAU=H.PLATEAU, PLATEAU_TOL=H.PLATEAU_TOL, seeds=list(SEEDS),
                            judged=list(JUDGED), reported=list(REPORTED), bank=[BANK_SEED, N_BANK],
                            checkpoint="final"),
                criterion="per seed: schema share > null share by more than 2 bootstrap SE; a judged level is "
                          "'klarad' if both seeds pass; no overall verdict across levels")


def lock(approval):
    if os.path.exists(LOCK):
        sys.exit("already locked")
    json.dump(dict(approval=approval, **fingerprint()), open(LOCK, "w"), indent=1)
    print("locked", LOCK)


def measure():
    if not os.path.exists(LOCK):
        sys.exit("h5_lock.json missing: thresholds not locked by the product owner. Refusing to unblind.")
    lock = json.load(open(LOCK))
    now = fingerprint()
    for k in ("sha256", "params", "criterion"):
        if lock[k] != now[k]:
            sys.exit(f"lock mismatch in {k}: code, parameters or feasibility changed after locking. Refusing.")
    feas = json.load(open(os.path.join(RES, "sprint5a_h5_feasibility.json")))
    out = dict(lock=lock, runs={}, levels={})
    for lvl, stem in LEVELS.items():
        method = feas["levels"][lvl]["method"]
        res, sens = [], []
        for i, s in enumerate(SEEDS):
            run = f"{stem}_s{s}"
            ev, info = events_for(run, blind=False)
            d = H.build(ev)
            r = H.evaluate(d, method, seed=1000 + i) if method else dict(method=None, n_events=int(d["n"]))
            r.update(info)
            # sensitivity (review R1): the other matching method, where it is identified
            other = "argmax" if method == "mass" else None
            if other and feas["runs"][run][other]["possible"]:
                r["sensitivity_argmax"] = H.evaluate(d, other, seed=2000 + i)
            out["runs"][run] = r
            res.append(r)
            sens.append(r.get("sensitivity_argmax"))
            print(run, json.dumps({k: (v if not isinstance(v, float) else round(v, 4)) for k, v in r.items()
                                   if k != "sensitivity_argmax"}), flush=True)
        if method is None:
            verdict = "ej bedömbar"
        else:
            verdict = "klarad" if all(r["passed"] for r in res) else "ej klarad"
            if all(x is not None for x in sens):
                v2 = "klarad" if all(x["passed"] for x in sens) else "ej klarad"
                if v2 != verdict:
                    verdict = "osäker (matchningsmetoderna ger olika svar)"
        out["levels"][lvl] = dict(judged=lvl in JUDGED, method=method,
                                  verdict=verdict if lvl in JUDGED else "redovisas (" + verdict + ")")
        json.dump(out, open(os.path.join(RES, "sprint5a_h5.json"), "w"), indent=1, default=float)
    print("LEVELS", json.dumps(out["levels"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "lock":
        lock(sys.argv[2])
    else:
        {"feasibility": feasibility, "measure": measure}[sys.argv[1]]()
