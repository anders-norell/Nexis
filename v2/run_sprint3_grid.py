"""Sprint 3 pilot, step 2 (grid) and step 4 (confirm).

Step 2: grid of working points. Baseline and Fixed schema (1a), seed 0,
400 000 episodes each (entropy annealed over 300 000). Two runs at a time, one thread each.
Skips finished runs and resumes interrupted ones, so it can simply be started again.

  python3 run_sprint3_grid.py > results/p3_grid.stdout 2>&1

Step 4, at the chosen working point (standard settings, chosen 22 Oct by the pre-set rule):
three new seeds (1, 2, 3) for Baseline, Fixed schema (1a) and Shuffled schema, plus one seed of
Learned schema (1b, head loss weight 1.0) to settle the 1b design and check that H5 is measurable.

  python3 run_sprint3_grid.py confirm > results/p3_confirm.stdout 2>&1
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
POINTS = [  # (reveal, p_capture, windows), standard point first
    (1.0, 0.25, "1,2"), (1.0, 0.5, "1,2"), (1.0, 0.25, "1"),
    (0.5, 0.25, "1,2"), (0.5, 0.5, "1,2"),
    (0.25, 0.25, "1,2"), (0.25, 0.5, "1,2"),
    (0.0, 0.25, "1,2"), (0.0, 0.5, "1,2"),
]
SEED = 0
EPISODES = 400_000


def name(rev, pc, win, schema, seed=SEED):
    return f"p3_r{rev}_c{pc}_w{win.replace(',', '')}_{schema}_s{seed}"


def jobs():
    for rev, pc, win in POINTS:
        for schema in ("off", "fixed"):
            out = os.path.join("results", name(rev, pc, win, schema))
            if os.path.exists(os.path.join(HERE, out + ".final.npz")):
                continue
            yield out, ["python3", "train_rl.py", "--seed", str(SEED), "--out", out, "--schema", schema,
                        "--reveal", str(rev), "--p-capture", str(pc), "--windows", win,
                        "--max-episodes", str(EPISODES), "--ent-episodes", "300000", "--minutes", "90"]


def cmd(rev, pc, win, schema, seed, extra=()):
    out = os.path.join("results", name(rev, pc, win, schema, seed))
    return out, ["python3", "train_rl.py", "--seed", str(seed), "--out", out, "--schema", schema,
                 "--reveal", str(rev), "--p-capture", str(pc), "--windows", win,
                 "--max-episodes", str(EPISODES), "--ent-episodes", "300000", "--minutes", "90", *extra]


def confirm_jobs(point=(1.0, 0.25, "1,2")):
    todo = [("learned", 1, ("--sl", "1.0"))]
    todo += [(sch, s, ()) for s in (1, 2, 3) for sch in ("off", "fixed", "shuffled")]
    for sch, s, extra in todo:
        out, c = cmd(*point, sch, s, extra)
        if not os.path.exists(os.path.join(HERE, out + ".final.npz")):
            yield out, c


def main(par=2, mode="grid"):
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    queue, running = list(jobs() if mode == "grid" else confirm_jobs()), []
    while queue or running:
        while queue and len(running) < par:
            out, cmd = queue.pop(0)
            f = open(os.path.join(HERE, out + ".stdout"), "a")
            running.append((out, subprocess.Popen(cmd, cwd=HERE, stdout=f, stderr=subprocess.STDOUT, env=env)))
            print(time.strftime("%H:%M"), "start", out, flush=True)
        time.sleep(10)
        for item in list(running):
            if item[1].poll() is not None:
                print(time.strftime("%H:%M"), "done", item[0], "rc", item[1].returncode, flush=True)
                running.remove(item)


if __name__ == "__main__":
    main(2, sys.argv[1] if len(sys.argv) > 1 else "grid")
