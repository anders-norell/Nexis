"""Reproduce the main tables of the report "A comparator, not a self-model" (sprints 2-6 and step 0).

Three levels, from fast to complete. Each writes nothing outside its work directory except the tables file.

  python3 reproduce.py tables
      Seconds. Builds the report's main tables, and the derived statistics (between-seed means, step 0
      interval), from the committed result files in results/. Writes results/reproduce_tables.md.

  python3 reproduce.py evaluate [--work DIR] [--all]
      About 1.5-2 h on one core. Copies v2/ to DIR (default ./repro_work), deletes the committed result
      files there, re-runs every evaluation on the committed checkpoints (the locked evaluators refuse to run
      if any locked code differs), compares each new result file with the committed one number by number,
      and builds the tables from the new files. --all also re-runs the two pilot evaluations (sprint 5a and
      sprint 6, about 2 h more).

  python3 reproduce.py train [--work DIR] [--jobs 2] [--only NAME[,NAME]]
      About 25-35 h with two cores. Retrains every run behind the main tables from scratch, with exactly
      the arguments stored in the committed checkpoint, compares the new final weights with the committed
      ones, then runs `evaluate` on the retrained checkpoints. --only retrains the named runs only (no
      evaluation), e.g. --only c6_mL-alarmdirect_s30 for a one-hour determinism check.

Requirements: Python 3, NumPy (the runs were made with NumPy 2.x on Linux x86-64; float results may differ
in the last digits on other platforms). Training uses one thread per run (OMP_NUM_THREADS=1).
"""
import argparse
import glob
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")

# Result files rebuilt by `evaluate`, in dependency order: (output, command). Later steps read earlier outputs.
EVAL_STEPS = [
    ("sprint2_gate.json", ["eval_gate2.py"]),
    ("sprint4_mvp.json", ["eval_mvp.py"]),
    ("sprint4_supp_h2a.json", ["supp_h2a.py"]),
    ("sprint5a_h5.json", ["h5_5a.py", "measure"]),
    ("sprint6_main.json", ["eval_main_1c.py", "run"]),
    ("sprint6_step0.json", ["eval_step0.py"]),
]
EVAL_PILOTS = [
    ("sprint5a_pilot.json", ["pilot_5a.py", "all"]),
    ("sprint6_pilot.json", ["pilot_1c.py", "eval"]),
]

# Runs behind the main tables (checkpoint name stems in results/).
RUNS = (
    [f"s2_base_s{s}" for s in (0, 1, 2)]
    + [f"s4_{c}_s{s}" for c in ("off", "fixed", "shuffled") for s in (10, 11, 12)]
    + [f"p5_{n}_s{s}" for n in ("eref1e-4", "ehalf44.2", "ehalf22.1", "ehalf11.05", "ehalf5.53") for s in (20, 21)]
    + [f"c6_mL-{c}_s{s}" for c in ("off", "e1e-4", "u1e-4", "cap1e-4", "1bE", "alarmdirect") for s in (30, 31, 32)]
)


# ----------------------------------------------------------------------------------------------- tables
def pp(x, d=1):
    return f"{100 * x:+.{d}f}" if x is not None else "-"


def pct(x, d=1):
    return f"{100 * x:.{d}f} %"


def seed_stats(vals):
    """Mean and SE over training seeds (sample SD / sqrt(n))."""
    v = np.asarray(vals, float)
    return float(v.mean()), float(v.std(ddof=1) / math.sqrt(len(v)))


def tables(res_dir):
    L = ["# Main tables, rebuilt from result files", "", f"Source: `{os.path.relpath(res_dir, HERE)}/`", ""]
    J = lambda f: json.load(open(os.path.join(res_dir, f)))

    # Sprint 2
    g = J("sprint2_gate.json")
    L += ["## Sprint 2: competence (seeds 0-2)", "", "| Seed | Baseline hit rate | Random agent |", "| --- | --- | --- |"]
    for s, r in sorted(g["seeds"].items(), key=lambda kv: int(kv[0])):
        L.append(f"| {s} | {pct(r['agent']['found'])} | {pct(r['random']['found'])} |")
    L += [""]

    # Sprint 4
    m = J("sprint4_mvp.json")
    R, V = m["runs"], m["verdict"]
    S4 = (10, 11, 12)
    row = lambda name, cells: L.append(f"| {name} | " + " | ".join(cells) + " |")
    L += ["## Sprint 4: main experiment (seeds 10-12, p_capture 0.25)", "",
          "| Measure | Seed 10 | Seed 11 | Seed 12 |", "| --- | --- | --- | --- |"]
    for c, lab in (("off", "Baseline"), ("fixed", "1a"), ("shuffled", "Shuffled")):
        row(f"Hit rate, {lab}", [pct(R[f"s4_{c}_s{s}"]["intact"]["found"]) for s in S4])
    for k, lab in (("vs_baseline", "F: 1a - Baseline"), ("vs_blandat", "F: 1a - Shuffled")):
        row(lab, [f"{pp(x['diff'])} pp (z {x['z']:.1f})" for x in V["F"][k]])
    row("H1: slope, 1a - Baseline", [f"{x['diff']:+.3f} (z {x['z']:.1f})" for x in V["H1"]["vs_baseline"]])
    for p, lab in (("a", "H2 (a) false unvisited"), ("b", "H2 (b) false visited")):
        row(lab, [f"{x['diff']:+.2f} (SE {x['se']:.3f})" for x in V["H2"][p]["fixed"]])
        row(f"{lab}, Shuffled", [f"{x['diff']:+.2f}" for x in V["H2"][p]["blandat"]])
    row("H3a: revisits, Baseline - 1a", [f"{pp(x['diff'])} pp (z {x['z']:.1f})" for x in V["H3a"]["revisits"]])
    row("H3b: acute lesion, hit change", [f"{pp(x['diff'])} pp" for x in V["H3b"]["hits"]])
    mu, se = seed_stats([x["diff"] for x in V["F"]["vs_baseline"]])
    L += ["", f"Verdicts: F {V['F']['passed']}, H1 {V['H1']['passed']}, H2 (a) {V['H2']['a']['effect_passed']}, "
          f"H2 (b) {V['H2']['b']['effect_passed']}, H3 {V['H3']['passed']}. "
          f"Between seeds, F vs Baseline {pp(mu)} +- {100 * se:.1f} pp (mean +- SE).", ""]

    # Sprint 5a H5
    h = J("sprint5a_h5.json")
    L += ["## Sprint 5a: H5 in 1b-E (seeds 20, 21)", "",
          "| Run | Bits per step | Hit rate | Schema share on intended | Null share | Difference (z) |",
          "| --- | --- | --- | --- | --- | --- |"]
    for n, r in h["runs"].items():
        L.append(f"| {n} | {r['bits_per_step']:.1f} | {pct(r['hit'])} | {r['schema_share']:.2f} | "
                 f"{r['null_share']:.2f} | {r['diff']:+.2f} ({r['z']:.0f}) |")
    L += ["", "Verdicts per level: " + "; ".join(f"{k}: {v['verdict']}" for k, v in h["levels"].items()), ""]

    # Sprint 6 main
    s6 = J("sprint6_main.json")
    R6, J6 = s6["runs"], s6["judgement"]
    S6 = (30, 31, 32)
    L += ["## Sprint 6: main run (seeds 30-32, p_capture 0.5)", "",
          "| Condition | Seed 30 | Seed 31 | Seed 32 | Bits per step |", "| --- | --- | --- | --- | --- |"]
    for c, lab in (("off", "Baseline"), ("e1e-4", "1c with alarm"), ("u1e-4", "1c without alarm"),
                   ("cap1e-4", "Capacity baseline"), ("1bE", "1b-E")):
        rs = [R6[f"c6_mL-{c}_s{s}"] for s in S6]
        bits = [r["bits"]["bits_per_step"] for r in rs if "bits" in r]
        b = f"{min(bits):.1f}-{max(bits):.1f}" if bits else "-"
        L.append(f"| {lab} | " + " | ".join(pct(r["found"]) for r in rs) + f" | {b} |")
    e = J6["e1e-4"]
    L += ["", "Prerequisites for 1c with alarm, per seed:", ""]
    for k, lab in (("a", "(a) vs Baseline"), ("b", "(b) vs capacity baseline")):
        L.append(f"- {lab}: " + ", ".join(f"{pp(x['diff'])} pp (z {x['z']:.2f}, {'pass' if x['passed'] else 'fail'})"
                                          for x in e[k]))
    L.append("- (c) probe above chance: " + ", ".join(f"z {x['z']:.0f} ({'pass' if x['passed'] else 'fail'})"
                                                    for x in e["c"]))
    mu, se = seed_stats([x["diff"] for x in e["a"]])
    pooled_d = float(np.mean([x["diff"] for x in e["a"]]))
    pooled_se = float(math.sqrt(sum(x["se"] ** 2 for x in e["a"])) / 3)
    L += [f"- Post hoc: pooled gain over Baseline {pp(pooled_d)} pp (z {pooled_d / pooled_se:.1f} from test noise); "
          f"between seeds {pp(mu)} +- {100 * se:.1f} pp.", ""]
    for s in S6:
        c = R6[f"c6_mL-e1e-4_s{s}"]["h5_primary"]["criterion"]
        sh = R6[f"c6_mL-e1e-4_s{s}"]["shadow_h5_primary"]["criterion"]
        L.append(f"- H5 probe (descriptive), seed {s}: intended - decoy {c['decoy']['value']:+.3f} "
                 f"(z {c['decoy']['z']:.0f}), intended - matched {c['matched']['value']:+.3f} (z {c['matched']['z']:.0f}); "
                 f"shadow module {sh['decoy']['value']:+.3f} / {sh['matched']['value']:+.3f}")
    L += [""]

    # Step 0
    z0 = J("sprint6_step0.json")
    C = z0["comparison"]
    L += ["## Step 0: comparator inputs to the policy, no module (seeds 30-32)", "",
          "| | Seed 30 | Seed 31 | Seed 32 |", "| --- | --- | --- | --- |",
          "| Hit rate, alarm direct | " + " | ".join(pct(z0["runs"][str(s)]["found"]) for s in S6) + " |",
          "| - 1c with alarm | " + " | ".join(f"{pp(C[str(s)]['vs_1c_with_alarm']['diff'], 2)} pp "
                                            f"(z {C[str(s)]['vs_1c_with_alarm']['z']:.2f})" for s in S6) + " |",
          "| - Baseline | " + " | ".join(f"{pp(C[str(s)]['vs_baseline']['diff'])} pp "
                                       f"(z {C[str(s)]['vs_baseline']['z']:.1f})" for s in S6) + " |"]
    d1 = [C[str(s)]["vs_1c_with_alarm"] for s in S6]
    d = float(np.mean([x["diff"] for x in d1]))
    se = float(math.sqrt(sum(x["se"] ** 2 for x in d1)) / 3)
    mu, sse = seed_stats([x["diff"] for x in d1])
    mub, sseb = seed_stats([C[str(s)]["vs_baseline"]["diff"] for s in S6])
    L += ["", f"Pre-registered rule met (module added nothing): {z0['module_added_nothing']}. "
          f"Post hoc: alarm direct - 1c, pooled {pp(d, 2)} pp, 95 % CI [{pp(d - 1.96 * se)}, {pp(d + 1.96 * se)}] pp "
          f"from test noise; between seeds {pp(mu, 2)} +- {100 * sse:.2f} pp. "
          f"Alarm direct - Baseline between seeds {pp(mub)} +- {100 * sseb:.1f} pp.", ""]
    return "\n".join(L)


def cmd_tables(args):
    out = os.path.join(RES, "reproduce_tables.md")
    open(out, "w").write(tables(RES) + "\n")
    print(open(out).read())
    print("wrote", out)


# ----------------------------------------------------------------------------------------------- compare
def flatten(x, prefix=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from flatten(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from flatten(v, f"{prefix}[{i}]")
    else:
        yield prefix, x


SKIP = ("minutes", "log.", "checkpoint_sha256", "approval", "sha256", "time", "date")


def compare(new_path, old_path):
    """Number-by-number comparison of two result files. Returns (n_compared, n_differing, max_abs_diff, examples)."""
    a = dict(flatten(json.load(open(new_path))))
    b = dict(flatten(json.load(open(old_path))))
    n = nd = 0
    mx, ex = 0.0, []
    for k in sorted(set(a) | set(b)):
        if any(s in k for s in SKIP):
            continue
        va, vb = a.get(k, "<missing>"), b.get(k, "<missing>")
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)) and not isinstance(va, bool):
            n += 1
            if (isinstance(va, float) and math.isnan(va)) and (isinstance(vb, float) and math.isnan(vb)):
                continue
            dif = abs(float(va) - float(vb))
            if dif > 1e-9 * max(1.0, abs(float(vb))):
                nd += 1
                mx = max(mx, dif)
                if len(ex) < 5:
                    ex.append(f"{k}: new {va} committed {vb}")
        elif va != vb:
            n += 1
            nd += 1
            if len(ex) < 5:
                ex.append(f"{k}: new {str(va)[:60]} committed {str(vb)[:60]}")
    return n, nd, mx, ex


# ----------------------------------------------------------------------------------------------- evaluate
def make_work(work, fresh_checkpoints=None):
    """Copy v2/ (code, committed checkpoints, locks, pre-registrations) to `work`, then remove the result files
    that `evaluate` rebuilds. With fresh_checkpoints (a directory), the checkpoints of RUNS are replaced by the
    retrained ones."""
    if os.path.exists(work):
        shutil.rmtree(work)
    shutil.copytree(HERE, work, ignore=shutil.ignore_patterns("repro_work*", "__pycache__", "*.ckpt.npz",
                                                              "*.log.jsonl", "*.stdout", "*.log"))
    for f, _ in EVAL_STEPS + EVAL_PILOTS:
        p = os.path.join(work, "results", f)
        if os.path.exists(p):
            os.remove(p)
    if fresh_checkpoints:
        for n in RUNS:
            for ext in (".best.npz", ".final.npz"):
                src = os.path.join(fresh_checkpoints, n + ext)
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(work, "results", n + ext))
        # the training logs are read by some evaluators (log summaries); use the retrained ones
        for n in RUNS:
            src = os.path.join(fresh_checkpoints, n + ".log.jsonl")
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(work, "results", n + ".log.jsonl"))
    else:
        for n in glob.glob(os.path.join(RES, "logs", "*.log.jsonl")):
            shutil.copy2(n, os.path.join(work, "results", os.path.basename(n)))


def run_steps(work, steps, logf):
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    for out, cmd in steps:
        t = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] {' '.join(cmd)}", flush=True)
        with open(logf, "a") as lf:
            r = subprocess.run([sys.executable] + cmd, cwd=work, stdout=lf, stderr=subprocess.STDOUT, env=env)
        ok = r.returncode == 0 and os.path.exists(os.path.join(work, "results", out))
        print(f"    -> {out} {'ok' if ok else 'FAILED (see ' + logf + ')'} in {(time.time() - t) / 60:.1f} min",
              flush=True)
        if not ok:
            return False
    return True


def cmd_evaluate(args, fresh=None):
    work = os.path.abspath(args.work)
    make_work(work, fresh)
    logf = os.path.join(work, "reproduce_evaluate.log")
    steps = EVAL_STEPS + (EVAL_PILOTS if args.all else [])
    ok = run_steps(work, steps, logf)
    report = ["# Comparison of re-evaluated result files with the committed ones", ""]
    for out, _ in steps:
        new, old = os.path.join(work, "results", out), os.path.join(RES, out)
        if not os.path.exists(new):
            report.append(f"- {out}: not produced")
            continue
        n, nd, mx, ex = compare(new, old)
        report.append(f"- {out}: {n} values compared, {nd} differ" + (f" (max abs diff {mx:.3g})" if nd else ""))
        report += [f"    - {e}" for e in ex]
    if ok:
        report += ["", tables(os.path.join(work, "results"))]
    out = os.path.join(work, "reproduce_report.md")
    open(out, "w").write("\n".join(report) + "\n")
    print("\n".join(report[: 2 + len(steps) * 6]))
    print("wrote", out)
    return ok


# ----------------------------------------------------------------------------------------------- train
def script_for(a):
    if a.get("direct_alarm"):
        return "train_alarm_direct.py"
    if a.get("schema") == "1c":
        return "train_1c.py"
    return "train_rl.py"


def valid_flags(script):
    h = subprocess.run([sys.executable, script, "--help"], cwd=HERE, capture_output=True, text=True).stdout
    return {tok.split("=")[0].strip("[],") for tok in h.split() if tok.startswith("--")}


def train_command(name, out_stem):
    z = np.load(os.path.join(RES, name + ".final.npz"))
    a = json.loads(str(z["m_args"]))
    script = script_for(a)
    flags = valid_flags(script)
    cmd = [script]
    for k, v in a.items():
        f = "--" + k.replace("_", "-")
        if k in ("out", "minutes", "direct_alarm") or f not in flags:
            continue
        if isinstance(v, bool):
            if v:
                cmd.append(f)
        else:
            cmd += [f, str(v)]
    return cmd + ["--out", out_stem, "--minutes", "100000"], int(z["m_episodes"])


def weights_equal(p, q):
    a, b = np.load(p), np.load(q)
    keys = [k for k in a.files if k.startswith(("p_", "m_p_", "mod_"))] or [k for k in a.files if not k.startswith("m_")]
    mx = 0.0
    for k in keys:
        if k not in b.files or a[k].shape != b[k].shape:
            return False, float("inf")
        mx = max(mx, float(np.max(np.abs(a[k].astype(float) - b[k].astype(float)))) if a[k].size else 0.0)
    return mx == 0.0, mx


def cmd_train(args):
    work = os.path.abspath(args.work)
    tdir = os.path.join(work, "retrained")
    os.makedirs(tdir, exist_ok=True)
    names = args.only.split(",") if args.only else RUNS
    jobs = []
    for n in names:
        if os.path.exists(os.path.join(tdir, n + ".final.npz")):
            continue
        cmd, ep = train_command(n, os.path.join(tdir, n))
        jobs.append((n, cmd))
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    running = []
    while jobs or running:
        while jobs and len(running) < args.jobs:
            n, cmd = jobs.pop(0)
            print(f"[{time.strftime('%H:%M:%S')}] start {n}: {' '.join(cmd)}", flush=True)
            lf = open(os.path.join(tdir, n + ".stdout"), "a")
            running.append((n, subprocess.Popen([sys.executable] + cmd, cwd=HERE, stdout=lf, stderr=subprocess.STDOUT,
                                                env=env)))
        time.sleep(20)
        for n, p in list(running):
            if p.poll() is not None:
                running.remove((n, p))
                print(f"[{time.strftime('%H:%M:%S')}] end {n} (exit {p.returncode})", flush=True)
    lines = ["# Retrained checkpoints compared with the committed ones", ""]
    for n in names:
        for ext in (".final.npz", ".best.npz"):
            p, q = os.path.join(tdir, n + ext), os.path.join(RES, n + ext)
            if os.path.exists(p) and os.path.exists(q):
                eq, mx = weights_equal(p, q)
                lines.append(f"- {n}{ext}: {'identical' if eq else f'differs, max abs weight diff {mx:.3g}'}")
            else:
                lines.append(f"- {n}{ext}: missing")
    out = os.path.join(tdir, "retrain_report.md")
    open(out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("wrote", out)
    if not args.only:
        args.work = os.path.join(work, "evaluate")
        cmd_evaluate(args, fresh=tdir)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["tables", "evaluate", "train"])
    ap.add_argument("--work", default=os.path.join(HERE, "repro_work"))
    ap.add_argument("--all", action="store_true", help="evaluate: also the two pilot evaluations")
    ap.add_argument("--jobs", type=int, default=2, help="train: runs in parallel")
    ap.add_argument("--only", default="", help="train: comma-separated run names, no evaluation")
    args = ap.parse_args()
    {"tables": cmd_tables, "evaluate": cmd_evaluate, "train": cmd_train}[args.what](args)


if __name__ == "__main__":
    main()
