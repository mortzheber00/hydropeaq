#!/usr/bin/env python3
"""Per-leg stroke benefit: cycle-mean drag while stroking vs held at the mean pose.

A submerged leg causes drag even when still, so zero is the wrong baseline
for judging its thrust. The frozen case keeps the base and the other legs as
solved. The comparison is drag only, so it is unaffected by the added-mass
artifact of the net force balance. --pose-scan N shows how much the choice
of held pose matters.

Usage:
  python stage3_visualization/thrust/plot_stroke_benefit.py
  python stage3_visualization/thrust/plot_stroke_benefit.py --solution task3_solution.npz --save benefit.pdf
  python stage3_visualization/thrust/plot_stroke_benefit.py --pose-scan 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "stage1_gait_optimization"))
from stage3_visualization.common.collocation import D_COLLOC, solution_states  # noqa: E402
from stage3_visualization.common.drag_model import leg_cycle_drag, stroke_benefit  # noqa: E402
from stage3_visualization.common.thesis_style import (  # noqa: E402
    PALETTE,
    TEXT_WIDTH_IN,
    full_width,
    legend_row,
)

from stage1_gait_optimization.hydro_model import load_robot  # noqa: E402
from stage1_gait_optimization.hydro_model.trajectory import load_solution  # noqa: E402

FROZEN_C = PALETTE[2]     # held still
SOLVED_C = PALETTE[1]     # stroking

full_width()


def pose_scan(robot, Xc_leg, nq, N, leg, n_poses, d=D_COLLOC):
    """Frozen-case drag with the leg held at ``n_poses`` of its own poses [N]."""
    names = list(robot.spec.actuated_joint_names)
    idx = [names.index(f"{leg}_{s}_joint") for s in ("Side", "Thigh", "Calf")]
    cols = np.linspace(0, N * d - 1, n_poses, dtype=int)
    return np.array([leg_cycle_drag(robot, Xc_leg, nq, N, leg,
                                    hold=[Xc_leg[7 + j, c] for j in idx])
                     for c in cols])


def plot_benefit(data):
    """Frozen vs solved drag per leg, with the gain as an arrow."""
    legs = list(data)
    y = np.arange(len(legs))
    frozen = np.array([data[lg][1] for lg in legs])
    solved = np.array([data[lg][0] for lg in legs])

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 2.7))
    h = 0.34
    ax.barh(y - h / 2, frozen, height=h, color=FROZEN_C, zorder=3,
            label="held still at mean pose")
    ax.barh(y + h / 2, solved, height=h, color=SOLVED_C, zorder=3,
            label="stroking, as solved")

    for i, (f, s) in enumerate(zip(frozen, solved)):
        ax.annotate("", xy=(s, i), xytext=(f, i),
                    arrowprops=dict(arrowstyle="->", lw=0.8, color="0.3",
                                    shrinkA=0, shrinkB=0), zorder=5)
        ax.annotate(f"+{s - f:.2f} N", xy=((f + s) / 2, i - 0.42),
                    ha="center", va="bottom", fontsize=7, color="0.25", zorder=5)

    ax.axvline(0.0, color="0.4", lw=0.7, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels([lg.replace("_", " ") for lg in legs], fontsize=8)
    # Inverted, with room for the gain label of the first row
    ax.set_ylim(len(legs) - 0.45, -0.85)
    ax.set_xlabel(r"cycle-mean forward force on the leg "
                  r"$\bar F_x^{\mathrm{world}}$ [N]")
    ax.grid(axis="x", alpha=0.3)
    ax.margins(x=0.12)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
              frameon=False, fontsize=8, borderaxespad=0.2, handlelength=1.4)
    legend_row(fig, ax)
    fig.tight_layout(pad=0.3)
    return fig


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--solution", type=Path, default=_ROOT / "task3_solution.npz")
    ap.add_argument("--save", type=Path, default=None)
    ap.add_argument("--pose-scan", type=int, default=0, metavar="N",
                    help="also report the frozen drag holding each leg at N of "
                         "its own poses, to bound the baseline's sensitivity")
    args = ap.parse_args()

    meta = load_solution(args.solution)
    robot = load_robot(meta["robot"])
    N, nq = meta["N"], meta["nq"]
    Xc_leg, _, _ = solution_states(robot, meta, args.solution.name)

    data = stroke_benefit(robot, Xc_leg, nq, N)

    print(f"{args.solution.name}:")
    print(f"  {'leg':14s}{'stroking':>10s}{'held still':>12s}{'recovered':>11s}")
    for lg, (s, f) in data.items():
        print(f"  {lg:14s}{s:10.4f}{f:12.4f}{s - f:11.4f}")
    tot_s = sum(s for s, _ in data.values())
    tot_f = sum(f for _, f in data.values())
    print(f"  {'all four legs':14s}{tot_s:10.4f}{tot_f:12.4f}{tot_s - tot_f:11.4f}")
    print("\n  (newtons of world-x force; 'recovered' is what the leg's own "
          "motion\n   buys against holding it still, and is the quantity to quote)")

    if args.pose_scan:
        print(f"\n  baseline sensitivity — frozen drag over {args.pose_scan} "
              f"held poses [N]:")
        for lg in data:
            v = pose_scan(robot, Xc_leg, nq, N, lg, args.pose_scan)
            print(f"    {lg:14s}{v.min():9.4f} … {v.max():.4f}"
                  f"   (mean-pose value {data[lg][1]:.4f})")

    fig = plot_benefit(data)
    if args.save:
        fig.savefig(args.save, dpi=300)
        print(f"\nSaved → {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
