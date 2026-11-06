"""Poster figure: hip error and across-axis knee error per method, against expert disagreement (VSD, pairwise).
Values are the HFValid medians from the paper (Tables 2, 6, 8).
  python make_poster_figure.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
INK, INK2, GRID, BLUE, GREY = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#b8b7b2"
methods = ["True mask\n(floor)", "AttentionUNet", "SwinUNETR", "2D\nbaseline"]
panels = [("Hip centre error", [0.39, 2.46, 2.77, 3.12], 0.57),
          ("Knee error across the axis", [1.82, 2.04, 2.12, 2.68], 2.04)]

plt.rcParams.update({"font.family": "sans-serif", "font.size": 13, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, (title, vals, expert) in zip(axes, panels):
    bars = ax.bar(range(4), vals, width=0.6, color=[GREY, BLUE, BLUE, BLUE], edgecolor="white", linewidth=2)
    for b, v in zip(bars, vals):
        y = max(v, expert + 0.02) if abs(v - expert) < 0.25 else v      # keep value labels clear of the dashed line
        ax.text(b.get_x() + b.get_width() / 2, y + 0.06, f"{v:.2f}", ha="center", va="bottom", color=INK, fontsize=12)
    ax.axhline(expert, color=INK, ls="--", lw=1.6, label=f"two experts disagree (VSD, {expert:.2f} mm)")
    ax.legend(loc="upper left", frameon=False, fontsize=11)
    ax.set_xticks(range(4), methods)
    ax.set_title(title, color=INK, fontsize=15, loc="left", pad=10)
    ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
axes[0].set_ylabel("median error (mm)")
axes[0].set_ylim(0, 3.6)
fig.tight_layout()
for out in (HERE / "paper/figures/poster_error.png", HERE / "paper/poster_upload/figures/poster_error.png"):
    out.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
print("saved")
