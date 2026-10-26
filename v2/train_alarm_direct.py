"""Sprint 6, step 0 (decision 26 Oct): the comparator signal straight to the policy, no module.

The policy is the baseline PriorityMapAgent with two extra per-cell inputs: the matching signal and the global
alarm, i.e. exactly the extra input that 1c "with alarm" gave its module (agent_rl.matching_signal, alarm_signal).
No schema channel, no module. Training is train_rl.py unchanged (same A2C, hyperparameters and seeding); this file
only swaps the agent factory and records direct_alarm=True in the checkpoint arguments. The locked files
(agent_rl.py, train_rl.py, ...) are not modified.

  python3 train_alarm_direct.py --seed 30 --p-capture 0.5 --max-episodes 1000000 --ent-episodes 300000 \
      --minutes 240 --out results/c6_mL-alarmdirect_s30
"""
import argparse
import sys
import train_rl
from agent_rl import PriorityMapAgent


def make_direct_alarm_agent(cfg, d=16, dh=32, seed=0, schema_mode="off"):
    return PriorityMapAgent(cfg, d=d, dh=dh, seed=seed, schema_mode="off", matching=True, alarm=True)


def main():
    train_rl.make_agent = make_direct_alarm_agent
    orig = argparse.ArgumentParser.parse_args

    def parse(self, *a, **k):
        ns = orig(self, *a, **k)
        ns.direct_alarm = True
        return ns
    argparse.ArgumentParser.parse_args = parse
    train_rl.main()


if __name__ == "__main__":
    main()
