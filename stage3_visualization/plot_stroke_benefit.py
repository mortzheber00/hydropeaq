#!/usr/bin/env python3
"""What each leg gains by stroking, against the drag it would suffer anyway.

The per-link attribution in ``plot_thrust_attribution.py`` says the hind legs
have a negative cycle-mean forward force, which reads as "the hind legs are dead
weight the optimiser should have left still".  That reading is wrong, and this
figure is the correction: **zero is the wrong baseline**.

A submerged limb costs drag whether or not it moves.  The legs are attached, the
hull is towed forward at 0.15 m/s, and the hind legs are the most deeply
submerged parts of the robot (calf alpha = 0.99, thigh alpha = 0.80), so they
pay a large rearward force before they do anything at all.  The question the
optimiser actually faces is not "does this leg produce thrust" but "is the robot
better off with this leg stroking than holding still", and that is what is drawn.

The counterfactual holds one leg's joints at their cycle-mean pose with zero
joint velocity, leaving the base motion and the other three legs exactly as
solved, and re-evaluates the drag on that leg's links.  Everything else about
the trajectory is untouched, so the difference is attributable to that leg's own
motion.

Two reasons this, rather than the raw cycle-mean force, is the number to quote:

  - It answers the question that determines whether the motion is worth its
    energy, which the raw force does not.
  - It is immune to the added-mass momentum defect.  Both sides of the
    comparison are pure quasi-steady drag — ``f_tau_added`` never enters — so
    the +-0.1 N artifact that makes the cycle-mean *net* force unquotable
    (see ``plot_thrust_budget.py``) is simply absent here.

The baseline is one choice among many: a leg held at a different pose suffers
different drag.  ``--pose-scan`` re-runs the frozen case holding the leg at each
of several poses taken from its own cycle and prints the spread, which bounds
how much the choice of held pose can move the answer.

Drag is ``drag_model.py``'s, which reproduces the model the OCP solved to
~5e-15; the counterfactual itself is ``drag_model.stroke_benefit``.  Samples are
the collocation points and the cycle mean uses the Radau weights, as everywhere
else here.

Usage:
  python plot_stroke_benefit.py
  python plot_stroke_benefit.py --solution ../task3_solution.npz --save benefit.pdf
  python plot_stroke_benefit.py --pose-scan 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "stage1_gait_optimization"))
from collocation import D_COLLOC, solution_states  # noqa: E402
from drag_model import leg_cycle_drag, stroke_benefit  # noqa: E402
from thesis_style import (  # noqa: E402
    PALETTE,
    TEXT_WIDTH_IN,
    full_width,
    legend_row,
)

from stage1_gait_optimization.hydro_model import load_robot  # noqa: E402
from stage1_gait_optimization.hydro_model.trajectory import load_solution  # noqa: E402

FROZEN_C = PALETTE[2]     # what the leg costs held still
SOLVED_C = PALETTE[1]     # what it costs stroking

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
    """Frozen against as-solved, one row per leg, with the gain drawn between."""
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

    # The gain is the distance between the two bars, so draw it as that distance
    # rather than as a third bar the reader has to relate back to them.
    for i, (f, s) in enumerate(zip(frozen, solved)):
        ax.annotate("", xy=(s, i), xytext=(f, i),
                    arrowprops=dict(arrowstyle="->", lw=0.8, color="0.3",
                                    shrinkA=0, shrinkB=0), zorder=5)
        ax.annotate(f"+{s - f:.2f} N", xy=((f + s) / 2, i - 0.42),
                    ha="center", va="bottom", fontsize=7, color="0.25", zorder=5)

    ax.axvline(0.0, color="0.4", lw=0.7, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels([lg.replace("_", " ") for lg in legs], fontsize=8)
    # Inverted with explicit padding: the gain label sits above each row, and on
    # the first row the default limits clip it against the frame.
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
