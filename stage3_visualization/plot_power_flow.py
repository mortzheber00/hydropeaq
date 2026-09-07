#!/usr/bin/env python3
"""Where the cycle's mechanical energy goes, and what the optimiser is charged for.

``plot_thrust_budget.py`` and ``plot_thrust_attribution.py`` say how the forward
force is made.  This says what it costs, and it is the figure that connects the
thrust mechanism to the cost of transport the co-design chapter reports.

Three figures on the full text-width canvas:

  ``*_power``   Instantaneous actuator power over the cycle: the signed total
      sum_j tau_j qdot_j, and the charged |power| sum_j |tau_j qdot_j|.  The gap
      between the two curves is negative work — joints being back-driven by the
      water — and the model pays full price for it.

  ``*_joints``  Positive and negative work per joint over the cycle.  Shows which
      joints deliver and which absorb, so the power cost can be read against the
      thrust attribution: a joint that absorbs is one the flow is driving.

  ``*_metric``  The objective the solver minimises against the metric the thesis
      reports, per interval.  They are different functionals (see below), and
      this is the figure that says how differently they rank the same cycle.

Two facts about the cost that this figure exists to make visible, both read off
the source rather than assumed:

  - The OCP minimises ``power_cost`` = sum_i B_i * ||tau .* qdot||^2
    (``ocp_common.build_collocation_nlp``, a *squared* per-joint power), while
    COT is reported from ``cycle_energy`` = integral of sum_j |tau_j qdot_j|
    (``ocp_common.cycle_energy``, an *absolute* one).  Minimising a sum of
    squares flattens power peaks; minimising absolute work reduces total work.
    They are not the same objective and they do not rank cycles the same way.
  - Both take |.| or (.)^2 per joint, so a joint absorbing power is charged
    exactly as much as one delivering it.  The model assumes no regeneration,
    which is the conservative assumption for a thesis to make, but it means the
    reported COT is an upper bound rather than a net energy balance.

Everything here — the traces, the per-joint split and the energy total — is
taken at the collocation points on the Radau weights, so the total is the same
number ``ocp_common.cycle_energy`` gives the co-design sweep rather than a
second, coarser estimate.  Sampling at the grid nodes instead is a
left-rectangle rule that reads 30-40% low, and it would also split the work at
points where Radau never enforced the dynamics (tau=0 is not a collocation
point).  That needs ``Xc``, which version-2 solution files carry and the
per-point ``codesign_results`` files do not; those are refused rather than
silently measured a second way.

Usage:
  python plot_power_flow.py
  python plot_power_flow.py --solution ../task3_solution.npz --save power.pdf
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
from collocation import D_COLLOC, cycle_integral, joint_power  # noqa: E402
from thesis_style import PALETTE, TEXT_WIDTH_IN, full_width, legend_row  # noqa: E402

from stage1_gait_optimization.hydro_model import load_robot  # noqa: E402
from stage1_gait_optimization.hydro_model.trajectory import load_solution  # noqa: E402
from stage1_gait_optimization.ocp_common import (  # noqa: E402
    collocation_coefficients,
    cost_of_transport,
    cycle_energy,
)

DELIVER_C = PALETTE[1]   # positive work — the actuator drives the joint
ABSORB_C = PALETTE[2]    # negative work — the water drives the joint
CHARGED_C = "0.25"

full_width()


def plot_power(P, T):
    """Figure 1: signed and charged actuator power over the cycle."""
    total = P.sum(axis=0)
    charged = np.abs(P).sum(axis=0)
    n = P.shape[1]

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 2.6))
    phase = (np.arange(n) + 0.5) / n
    two = np.concatenate([phase, 1 + phase])
    rep = lambda a: np.concatenate([a, a])                 # noqa: E731

    ax.axhline(0.0, color="0.75", lw=0.5, zorder=1)
    ax.axvline(1.0, color="0.6", lw=0.5, ls="--", zorder=1)

    ax.plot(two, rep(charged), color=CHARGED_C, lw=1.2, zorder=4,
            label=r"charged $\sum_j|\tau_j\dot q_j|$")
    ax.plot(two, rep(total), color=PALETTE[0], lw=1.0, zorder=3,
            label=r"net $\sum_j\tau_j\dot q_j$")
    # The area between them is the negative work, which is the whole point of
    # drawing both: it is what the robot spends and never gets back.
    ax.fill_between(two, rep(total), rep(charged), color=ABSORB_C, alpha=0.20,
                    lw=0, zorder=2, label="negative work (charged, not recovered)")

    ax.set_xlabel("cycle phase")
    ax.set_ylabel(r"actuator power [W]")
    ax.set_xlim(0, 2)
    ax.set_xticks([0, 0.5, 1, 1.5, 2])
    ax.grid(alpha=0.3)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
              handlelength=1.6, frameon=False, columnspacing=1.2,
              borderaxespad=0.2, fontsize=7.5)
    legend_row(fig, ax, rows=2)
    fig.tight_layout(pad=0.3)
    return fig


def plot_joints(P, T, labels, N):
    """Figure 2: work delivered and absorbed, per joint."""
    w_pos = np.array([cycle_integral(np.clip(row, 0.0, None), T, N) for row in P])
    w_neg = np.array([cycle_integral(np.clip(row, None, 0.0), T, N) for row in P])

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 3.0))
    y = np.arange(len(labels))
    ax.barh(y, w_pos, height=0.7, color=DELIVER_C, zorder=3, label="delivered")
    ax.barh(y, w_neg, height=0.7, color=ABSORB_C, zorder=3, label="absorbed")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.axvline(0.0, color="0.4", lw=0.7, zorder=4)
    ax.set_xlabel(r"work over one cycle [J]")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
              frameon=False, fontsize=8, borderaxespad=0.2, handlelength=1.4)
    legend_row(fig, ax)
    fig.tight_layout(pad=0.3)
    return fig


def plot_metric(P):
    """Figure 3: the minimised objective against the reported metric.

    Per interval, the absolute power the COT integrates and the squared power
    the NLP minimises, each normalised by its own cycle total so the two sit on
    one axis.  Where the two curves separate, the optimiser is being pushed by
    something the reported number does not measure.
    """
    absolute = np.abs(P).sum(axis=0)
    squared = (P ** 2).sum(axis=0)
    n = P.shape[1]

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 2.4))
    phase = (np.arange(n) + 0.5) / n
    ax.plot(phase, absolute / absolute.sum(), color=PALETTE[0], lw=1.1,
            label=r"reported: $\sum_j|\tau_j\dot q_j|$ (COT)")
    ax.plot(phase, squared / squared.sum(), color=PALETTE[2], lw=1.1,
            label=r"minimised: $\|\tau\odot\dot q\|^2$ (objective)")
    ax.set_xlabel("cycle phase")
    ax.set_ylabel("share of cycle total [-]")
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1,
              frameon=False, fontsize=7.5, borderaxespad=0.2, handlelength=1.6)
    legend_row(fig, ax, rows=2)
    fig.tight_layout(pad=0.3)
    return fig


def report(P, meta, robot, labels):
    """Energy totals, all on the Radau weights the sweep's own COT uses."""
    T, N = float(meta["T"]), int(meta["N"])
    w_pos = cycle_integral(np.clip(P, 0.0, None).sum(axis=0), T, N)
    w_neg = cycle_integral(np.clip(P, None, 0.0).sum(axis=0), T, N)
    charged = w_pos - w_neg

    print(f"  {'work delivered':<34s}{w_pos:>11.4f} J")
    print(f"  {'work absorbed':<34s}{w_neg:>11.4f} J")
    print(f"  {'net (signed) work':<34s}{w_pos + w_neg:>11.4f} J")
    print(f"  {'negative-work share of charged':<34s}"
          f"{-w_neg / charged * 100:>11.1f} %")

    # The number the sweep records, taken through its own function rather than
    # re-derived here.  It must equal `charged` above, which is the check that
    # this script's split and the sweep's total describe one trajectory.
    n_act = P.shape[0]
    nv = 6 + n_act
    _, _, _, B = collocation_coefficients(D_COLLOC)
    vc = np.asarray(meta["Xc"])[nv + 6:, :]
    E = cycle_energy(np.asarray(meta["U"]), vc, B, D_COLLOC, N, T)

    forward = float(meta["X"][0, -1] - meta["X"][0, 0])
    print(f"\n  {'cycle energy (cycle_energy)':<34s}{E:>11.4f} J")
    print(f"  {'  vs delivered + |absorbed|':<34s}{charged:>11.4f} J"
          f"   (diff {abs(E - charged):.2e})")
    print(f"  {'forward distance':<34s}{forward:>11.4f} m")
    print(f"  {'cost of transport':<34s}"
          f"{cost_of_transport(E, robot, forward):>11.4f}")

    worst = int(np.argmax(np.abs(P).sum(axis=1)))
    print(f"  {'most-worked joint':<34s}{labels[worst]:>11s}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--solution", type=Path, default=_ROOT / "task3_solution.npz")
    ap.add_argument("--save", type=Path, default=None,
                    help="output stem; each figure gets its own suffixed file")
    args = ap.parse_args()

    meta = load_solution(args.solution)
    robot = load_robot(meta["robot"])
    U, T = meta["U"], meta["T"]
    n_act = U.shape[0]
    if "Xc" not in meta:
        raise SystemExit(
            f"{args.solution.name} has no Xc block.  The grid-node fallback is a "
            f"left-rectangle rule that reads 30-40% low and splits the work at "
            f"points the dynamics were never enforced at, so it is refused "
            f"rather than reported.  Re-solve with a version-2 solution file.")

    P = joint_power(meta["Xc"], U, n_act, int(meta["N"]))
    labels = [n.replace("_", " ") for n in robot.spec.actuated_joint_names]

    print(f"{args.solution.name}:")
    report(P, meta, robot, labels)

    figs = {"power": plot_power(P, T),
            "joints": plot_joints(P, T, labels, int(meta["N"])),
            "metric": plot_metric(P)}
    if args.save:
        for tag, fig in figs.items():
            path = args.save.with_name(f"{args.save.stem}_{tag}{args.save.suffix}")
            fig.savefig(path, dpi=300)
            print(f"Saved → {path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
