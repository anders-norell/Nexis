"""Reference agents for the sprint 2 gate (run 21 Oct 2026, before training; this script
reproduces them). Standard settings, cue level weak, 3 x 4000 episodes (env seeds 900-902).

The best memoryless agent is the colour-guided agent without memory with a hot-quadrant prior;
its prior weight is the best of 1.5, 2, 3 and 5 (3), so the K2 threshold is set against the
strongest memoryless variant tried. Writes results/sprint2_references.json.
"""
import json
import os
import numpy as np
from env import EnvConfig, SearchEnv
from agents import RandomAgent, ItemScanAgent, GuidedAgent, MemorylessGuidedAgent
from metrics import run_metrics, summarise

AGENTS = {
    "random": lambda r: RandomAgent(r),
    "memoryless_guided": lambda r: MemorylessGuidedAgent(r),
    "memoryless_guided_prior1.5": lambda r: MemorylessGuidedAgent(r, 1.5),
    "memoryless_guided_prior2": lambda r: MemorylessGuidedAgent(r, 2.0),
    "memoryless_guided_prior3": lambda r: MemorylessGuidedAgent(r, 3.0),
    "memoryless_guided_prior5": lambda r: MemorylessGuidedAgent(r, 5.0),
    "item_scan": lambda r: ItemScanAgent(r),
    "guided_intended": lambda r: GuidedAgent(r, "intended"),
    "guided_inferred": lambda r: GuidedAgent(r, "inferred"),
    "guided_actual": lambda r: GuidedAgent(r, "actual"),
}

if __name__ == "__main__":
    out = {}
    for name, mk in AGENTS.items():
        out[name] = [summarise(run_metrics(SearchEnv(EnvConfig(), seed=900 + i), mk(np.random.default_rng(i)), 4000))
                     for i in range(3)]
        print(f"{name:28s} found {np.mean([x['found'] for x in out[name]]):.3f}", flush=True)
    with open(os.path.join(os.path.dirname(__file__), "results", "sprint2_references.json"), "w") as f:
        json.dump(out, f, indent=1)
