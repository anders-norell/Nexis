"""Sprint 1 figure: learning curves (left) and attention capture (right)."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from diagnose import load, capture_profile

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8), facecolor="#fcfcfb")

curves = [json.load(open(f"results/s1_learned_s{s}.json"))["curve"] for s in range(3)]
it = [c["iter"] for c in curves[0]]
acc = np.array([[c["acc"] for c in cv] for cv in curves])
uni = [c["acc"] for c in json.load(open("results/s1_uniform_s0.json"))["curve"]]
for row in acc:
    ax1.plot(it, row, color=BLUE, lw=1, alpha=0.35)
ax1.plot(it, acc.mean(0), color=BLUE, lw=2)
ax1.plot(it, uni, color=ORANGE, lw=2)
ax1.axhline(1 / 64, color=MUTED, lw=1, ls="--")
ax1.text(it[-1], acc.mean(0)[-1] + 0.03, "Steered attention (3 seeds)", color=INK, ha="right")
ax1.text(it[-1], uni[-1] + 0.03, "Uniform attention", color=INK, ha="right")
ax1.text(it[-1], 1 / 64 + 0.02, "Chance (1 in 64)", color=MUTED, ha="right", fontsize=9)
ax1.set_ylim(0, 0.8); ax1.set_xlabel("Training iterations"); ax1.set_ylabel("Final-step pixel accuracy")
ax1.set_title("Steering attention almost doubles accuracy", loc="left", color=INK, fontsize=11)
ax1.grid(axis="y", color=GRID, lw=0.8)

prof = capture_profile(load("s1_learned_s0"))
x = prof["rel_steps"]
ax2.axvspan(-0.4, 1.4, color=GRID, alpha=0.6, lw=0)
ax2.plot(x, prof["on_flash"], color=ORANGE, lw=2, marker="o", ms=6)
ax2.plot(x, prof["on_target"], color=BLUE, lw=2, marker="o", ms=6)
ax2.text(0.5, 0.99, "flash", color=MUTED, ha="center", va="top", fontsize=9)
ax2.text(-0.12, prof["on_flash"][1], "On flash patch", color=INK, ha="right", va="center")
ax2.text(-1, prof["on_target"][0] + 0.04, "On target", color=INK, ha="center", va="bottom")
ax2.set_xticks(x); ax2.set_xticklabels(["before", "flash 1", "flash 2", "after"])
ax2.set_xlim(-1.3, 2.5); ax2.set_ylim(0, 1); ax2.set_ylabel("Attention weight")
ax2.set_title("A flash captures attention; it returns right after", loc="left", color=INK, fontsize=11)
ax2.grid(axis="y", color=GRID, lw=0.8)

fig.tight_layout()
fig.savefig("report/sprint1.png", dpi=160, facecolor=fig.get_facecolor())
print("saved")
