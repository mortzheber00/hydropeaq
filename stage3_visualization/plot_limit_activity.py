#!/usr/bin/env python3
"""Which actuator limit shapes the optimised gait, and which one never binds.

Every other figure here shows what the solution *does*.  This one shows what it
was not allowed to do: the joint-angle box, the joint-rate box, the torque box
and the actuator-bandwidth filter, each drawn as the solver enforced it, with
the solution measured against it.

Two figures on the full text-width canvas, from two different inputs:

  ``*_traces``    The four constrained quantities over one cycle, every actuated
      joint on one axis, each normalised so that ``+-1`` is its own bound.
      Normalising is what makes twelve joints legible on one panel and puts the
      four limits — which live in rad, rad/s and Nm — on a common axis.  Note
      that ``0`` is the middle of each box, not the neutral pose: the angle box
      is asymmetric (amph's side joints run -135 to +21 deg).  One trajectory,
      from ``--solution``.

  ``*_activity``  Share of enforced samples sitting at each bound, against
      commanded speed along the Pareto front.  This is the panel that answers
      the question: a limit whose share grows with speed is what stops the robot
      going faster, and one that stays at zero is not part of the story at all.
      Speed is its x-axis, so it needs the whole sweep from ``--results``.

The two share ``measure``, which is why they live in one script: the shares are
then guaranteed to be the same computation, normalised the same way, over the
same enforced-sample set as the traces.  They do not share data, so a sweep that
is missing, incomplete or written before the ``Xc`` export costs the activity
figure alone — the traces are still drawn, and the reason is printed.

**Where the bounds are enforced, and why that is exactly the Xc columns.**
``ocp_common.build_collocation_nlp`` bounds the state at every grid node, and
again at the collocation points — but only ``j < d-1`` of them, because Radau's
last point is ``tau=1``, which *is* the next grid node, and constraining both
copies of one state makes LICQ fail.  So the enforced set is (grid nodes) plus
(interior collocation points), and the columns of ``Xc`` are precisely that
union: column ``k*d + (d-1)`` is grid node ``k+1``, and node 0 repeats node N
under the periodicity constraint.  Every enforced point therefore appears in
``Xc`` exactly once, which is why the shares below are taken over all of it and
need no de-duplication.

Sampling at the grid nodes alone understates the activity — it misses two of
every three constrained states — which is the reason this figure exists rather
than a histogram over ``X``.

Two things in the traces figure that look like bugs and are not.  amph's side
joints are pinned to zero by a pose constraint (``RobotSpec.pose_constraints``),
so they draw as flat lines — at zero on the rate panel, and at a fixed +-0.73 on
the angle panel, because their box is asymmetric and zero is not its middle.
And the torque-rate panel tracks the torque panel closely: at ``f_c`` = 20 Hz and
``dt`` ~ 18 ms the filter coefficient ``alpha`` is ~0.7, so each interval may
move the torque across 70 % of the full box and the window is a weak constraint
by construction.  ``alpha`` is printed with the report.

Torque is piecewise constant per interval, so it carries one value per interval
and is drawn as a step.  The bandwidth filter
(``ocp_common``: ``|U_k - (1-alpha) U_{k-1}| <= alpha * tau_max``, cyclic in k)
is the fourth actuator limit and the one nothing else in the pipeline looks at;
``alpha`` depends on the solved period, and ``f_c`` is not written to the
solution file, so it is read from the robot's spec — the same place the solve
read it.

Usage:
  python plot_limit_activity.py
  python plot_limit_activity.py --solution ../task3_solution.npz --save limits.pdf
  python plot_limit_activity.py --results /path/to/codesign_results
  python plot_limit_activity.py --results /nonexistent   # traces only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "stage1_gait_optimization"))
from thesis_style import (  # noqa: E402,F401  activates the shared style
    LEGEND_ROW_IN,
    PALETTE,
    TEXT_WIDTH_IN,
    full_width,
    legend_row,
)

import sweep_io  # noqa: E402
from collocation import D_COLLOC, collocation_states  # noqa: E402

from stage1_gait_optimization.hydro_model import load_robot  # noqa: E402
from stage1_gait_optimization.hydro_model.trajectory import load_solution  # noqa: E402
from stage1_gait_optimization.ocp_common import limits_for  # noqa: E402


# A sample counts as sitting on its bound when its normalised distance to that
# bound falls below this.  Relative, so one number covers all four limits.  It
# has to be loose enough to survive the solver: IPOPT converges the constraint
# to `tol` 1e-4 (`acceptable_tol` 1e-3), so an active bound is reached from
# either side by ~1e-3 in the constraint's own units — on the joint-angle box
# that is ~7e-4 normalised.  The report prints the peak utilisation next to
# every share so a reader can see the margin rather than take this on trust.
ACTIVE_TOL = 1e-3

# Colour by joint kind, not by leg: the question is whether *anything* reaches a
# bound, and twelve separate colours answer a question nobody asked.
KIND_COLOUR = {"Side": PALETTE[0], "Thigh": PALETTE[1], "Calf": PALETTE[2]}

# The constrained quantities, in the order they are drawn and tabulated.
#   key: (panel label, axis label)
LIMITS = {
    "angle": ("joint angle", r"$\hat q$"),
    "rate":  ("joint rate",  r"$\dot q / \dot q_{\max}$"),
    "torque": ("torque",     r"$\tau / \tau_{\max}$"),
    "bandwidth": ("torque rate", r"$\Delta\tau / (\alpha\,\tau_{\max})$"),
}

full_width()


def normalised(robot, Xc_leg, U, nq, N, T, d=D_COLLOC):
    """Each constrained quantity mapped so that its bound sits at ``+-1``.

    Returns ``({key: (n_joints, n_samples) array}, alpha)``.  The angle box is
    asymmetric, so it is mapped through its own midpoint and half-range; rate,
    torque and the bandwidth window are symmetric about zero and divide by the
    bound.  ``alpha`` comes back because how tight the bandwidth window is is
    not visible in its own normalised trace.
    """
    q_lb, q_ub, v_ub, tau_ub = limits_for(robot)
    mid, half = (q_ub + q_lb) / 2.0, (q_ub - q_lb) / 2.0

    # alpha is the solve's own first-order filter coefficient; it depends on the
    # solved period through dt, so it is rebuilt here rather than assumed.
    dt = float(T) / N
    f_c = robot.spec.ocp.f_c
    alpha = 2 * np.pi * dt * f_c / (2 * np.pi * dt * f_c + 1)
    # roll by one interval: U[:, 0]'s predecessor is U[:, N-1], which is the
    # wrap-around row the NLP adds after the k = 1..N-1 loop.
    prev = np.roll(U, 1, axis=1)

    return {
        "angle": (Xc_leg[7:nq, :] - mid[:, None]) / half[:, None],
        "rate": Xc_leg[nq + 6:, :] / v_ub[:, None],
        "torque": U / tau_ub[:, None],
        "bandwidth": (U - (1 - alpha) * prev) / (alpha * tau_ub[:, None]),
    }, alpha


def activity(norm: dict) -> dict:
    """Share of (joint, sample) pairs sitting on the bound, per limit."""
    return {k: float((np.abs(v) >= 1.0 - ACTIVE_TOL).mean())
            for k, v in norm.items()}


def plot_traces(norm, phase, N, kinds):
    """Figure 1: the four constrained quantities over one cycle."""
    fig, axes = plt.subplots(len(LIMITS), 1, sharex=True,
                             figsize=(TEXT_WIDTH_IN, 1.45 * len(LIMITS) + 0.5))

    # Torque and its rate are piecewise constant per interval, so they step at
    # the interval edges rather than following the collocation phases.
    step_phase = np.arange(N + 1) / N

    for ax, (key, (_, ylabel)) in zip(axes, LIMITS.items()):
        a = norm[key]
        stepped = a.shape[1] == N
        x = step_phase if stepped else phase
        for j in range(a.shape[0]):
            row = np.concatenate([a[j], a[j, :1]]) if stepped else a[j]
            ax.plot(x, row, color=KIND_COLOUR.get(kinds[j], "0.5"), lw=0.7,
                    alpha=0.75, zorder=3,
                    drawstyle="steps-post" if stepped else "default")
        for s in (-1.0, 1.0):
            ax.axhline(s, color="k", lw=0.9, zorder=4)
        ax.axhline(0.0, color="0.75", lw=0.5, zorder=1)
        ax.set_ylabel(ylabel)
        ax.set_ylim(-1.18, 1.18)
        ax.set_yticks([-1, 0, 1])
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("cycle phase")
    axes[-1].set_xlim(0, 1)
    axes[-1].set_xticks([0, 0.25, 0.5, 0.75, 1.0])

    handles = [Line2D([], [], color=c, lw=1.4, label=k.lower())
               for k, c in KIND_COLOUR.items()]
    handles.append(Line2D([], [], color="k", lw=0.9, label="limit"))
    axes[0].legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0),
                   ncol=4, fontsize=7.5, frameon=False, handlelength=1.6,
                   columnspacing=1.4, borderaxespad=0.2)
    fig.set_size_inches(fig.get_size_inches() + [0.0, LEGEND_ROW_IN])
    fig.tight_layout(pad=0.3)
    return fig


def plot_activity(points, band):
    """Figure 2: share of samples at each bound, against commanded speed."""
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 2.8))

    v = np.array([p["speed"] for p in points])
    pinned = np.array([p["pinned"] for p in points])
    for i, (key, (label, _)) in enumerate(LIMITS.items()):
        share = np.array([p["activity"][key] for p in points]) * 100.0
        colour = PALETTE[i % len(PALETTE)]
        ax.plot(v, share, color=colour, lw=1.2, alpha=0.85, zorder=2, label=label)
        # Same open/filled convention as plot_cot_vs_period: a point whose T sat
        # at a window edge had its cadence chosen by the grid, not the solver.
        ax.plot(v[~pinned], share[~pinned], color=colour, linestyle="none",
                marker="o", markersize=4.5, zorder=3)
        ax.plot(v[pinned], share[pinned], color=colour, linestyle="none",
                marker="o", markersize=4.5, markerfacecolor="none",
                markeredgewidth=0.9, zorder=3)

    ax.set_xlabel(r"commanded forward speed $v$ [m\,s$^{-1}$]")
    ax.set_ylabel(r"samples on the bound [\%]")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.margins(x=0.05)

    marks = [Line2D([], [], color="0.35", linestyle="none", marker="o",
                    markersize=4.5, markerfacecolor="none", markeredgewidth=0.9,
                    label=rf"$T$ pinned at edge ($\pm{band:.2f}$ s)")]
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles + marks, loc="lower center",
              bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=7,
              columnspacing=1.2, handlelength=1.6, frameon=False,
              borderaxespad=0.2)
    legend_row(fig, ax, rows=2)
    fig.tight_layout(pad=0.3)
    return fig


def measure(robot, meta):
    """``(norm, phase, alpha)`` for one solution, Xc frame checked first."""
    X, U, T, N, nq = meta["X"], meta["U"], meta["T"], meta["N"], meta["nq"]
    if "Xc" not in meta:
        raise SystemExit(
            "solution has no Xc block, so the bounds could only be checked at "
            "the grid nodes — two of every three enforced states would be "
            "missed.  Re-solve with a version-2 solution file.")
    Xc_leg, phase, _ = collocation_states(robot, X, meta["Xc"], nq, N)
    norm, alpha = normalised(robot, Xc_leg, U, nq, N, T)
    return norm, phase, alpha


def report(norm, labels, alpha):
    """Peak utilisation and share on the bound, per limit and per joint."""
    print(f"  bandwidth filter coefficient alpha = {alpha:.3f} "
          f"(the window is {alpha:.0%} of the full torque box per interval)")
    print(f"  {'limit':<14s}{'peak |n|':>10s}{'on bound':>10s}   {'worst joint':<24s}")
    for key, (label, _) in LIMITS.items():
        a = np.abs(norm[key])
        worst = int(np.argmax(a.max(axis=1)))
        share = (a >= 1.0 - ACTIVE_TOL).mean()
        print(f"  {label:<14s}{a.max():>10.4f}{share:>10.1%}   {labels[worst]:<24s}")

    over = {k: np.abs(v).max() - 1.0 for k, v in norm.items()}
    worst = max(over, key=over.get)
    if over[worst] > ACTIVE_TOL:
        # Past a bound by more than the solver's own tolerance is not an active
        # constraint, it is a violated one, and the figure would be drawing a
        # box the trajectory does not respect.
        raise SystemExit(
            f"\n{LIMITS[worst][0]} exceeds its bound by {over[worst]:.2e} "
            f"(normalised), past the {ACTIVE_TOL:.0e} tolerance — this solution "
            f"does not satisfy the limits it is being measured against.")


def sweep_activity(robot, meta, results):
    """``(points, band)`` for the activity figure, or ``None`` if the sweep cannot
    drive it.

    The traces figure needs nothing from the sweep, so a results directory that
    is missing, incomplete, or written before the ``Xc`` export should cost the
    activity panel and not the run.  ``sweep_io`` refuses each of those with a
    ``SystemExit`` naming the cause, and that message is what gets reported
    before the skip — a silent single-figure run would look like the sweep had
    been read and found empty.
    """
    try:
        band = sweep_io.detect_band(sweep_io.load_rows(results))
        pairs = sweep_io.load_sweep(results)
    except SystemExit as exc:
        print(f"\nNo activity figure — {exc}")
        return None

    # One robot build for the whole front — it dominates the runtime, and it is
    # the same robot as the traces solution unless --solution points elsewhere.
    names = {m["robot"] for _, m in pairs}
    if len(names) > 1:
        raise SystemExit(f"sweep mixes robots ({', '.join(sorted(names))})")
    sweep_robot = robot if names == {meta["robot"]} else load_robot(names.pop())

    print(f"\nPareto front from {results} "
          f"({len(pairs)} solves, free-T window +/- {band:.3f} s):")
    points = []
    for row, m in pairs:
        n_row, _, _ = measure(sweep_robot, m)
        points.append({"speed": row["speed"],
                       "pinned": sweep_io.is_pinned(row, band),
                       "activity": activity(n_row)})
        shares = "  ".join(f"{k} {points[-1]['activity'][k]:5.1%}" for k in LIMITS)
        print(f"  v = {row['speed']:.3f} m/s   {shares}")
    return points, band


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--solution", type=Path, default=_ROOT / "task3_solution.npz",
                    help="solution drawn in the traces figure")
    ap.add_argument("--results", type=Path, default=sweep_io.DEFAULT_RESULTS,
                    help="sweep directory the activity figure walks; without a "
                         "usable one only the traces figure is drawn")
    ap.add_argument("--save", type=Path, default=None,
                    help="output stem; each figure gets its own suffixed file")
    args = ap.parse_args()

    meta = load_solution(args.solution)
    robot = load_robot(meta["robot"])
    labels = [n.replace("_", " ") for n in robot.spec.actuated_joint_names]
    kinds = [next((k for k in KIND_COLOUR if k in n), "")
             for n in robot.spec.actuated_joint_names]

    print(f"{args.solution.name}:")
    norm, phase, alpha = measure(robot, meta)
    report(norm, labels, alpha)

    figs = {"traces": plot_traces(norm, phase, int(meta["N"]), kinds)}
    front = sweep_activity(robot, meta, args.results)
    if front is not None:
        figs["activity"] = plot_activity(*front)
    if args.save:
        for tag, fig in figs.items():
            path = args.save.with_name(f"{args.save.stem}_{tag}{args.save.suffix}")
            fig.savefig(path, dpi=300)
            print(f"Saved → {path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
