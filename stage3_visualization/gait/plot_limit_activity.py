#!/usr/bin/env python3
"""Actuator limit usage: which bounds shape the optimised gait.

``*_traces``   joint angle, joint rate, torque and torque rate (bandwidth
               filter) over one cycle, each normalised so its bound is +-1
               (--solution)
``*_activity`` share of the cycle in which any joint dwells on each bound, vs
               speed along the Pareto front (--results); skipped if the sweep
               is unusable

Traces are sampled on ``Xc``, which contains every point where the OCP enforces
the bounds exactly once. Flat lines in the traces are amph's pinned side joints.
f_c is read from the robot spec, since it is not stored in the solution.

Usage:
  python stage3_visualization/gait/plot_limit_activity.py
  python stage3_visualization/gait/plot_limit_activity.py --solution task3_solution.npz --save limits.pdf
  python stage3_visualization/gait/plot_limit_activity.py --results /path/to/codesign_results
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "stage1_gait_optimization"))
from stage3_visualization.common.thesis_style import (  # noqa: E402,F401  activates the shared style
    HALF,
    LEGEND_ROW_IN,
    PALETTE,
    TEXT_WIDTH_IN,
    full_width,
    half_width,
    legend_row,
)

from stage3_visualization.common import sweep_io  # noqa: E402
from stage3_visualization.common.collocation import D_COLLOC, collocation_states  # noqa: E402

from stage1_gait_optimization.hydro_model import load_robot  # noqa: E402
from stage1_gait_optimization.hydro_model.trajectory import load_solution  # noqa: E402
from stage1_gait_optimization.ocp_common import limits_for  # noqa: E402


# Normalised distance below which a bound counts as active (IPOPT tolerance level)
ACTIVE_TOL = 1e-3

# Dwell tolerance (thesis metric): within 1 % of the admissible span. In the
# half-range normalisation that is 2 * DWELL_FRAC. Looser than ACTIVE_TOL so a
# joint holding at a stop counts for the whole plateau. Also used by
# nominal_metrics.py.
DWELL_FRAC = 1e-2

# Quantities in the activity figure (the bandwidth window is not a state)
DWELL_LIMITS = ("angle", "rate", "torque")

# Colour by joint type, not by leg
KIND_COLOUR = {"Side": PALETTE[0], "Thigh": PALETTE[1], "Calf": PALETTE[2]}

# key: (panel label, axis label), in plotting order
LIMITS = {
    "angle": ("joint angle", r"$\hat q$"),
    "rate":  ("joint rate",  r"$\dot q / \dot q_{\max}$"),
    "torque": ("torque",     r"$\tau / \tau_{\max}$"),
    "bandwidth": ("torque rate", r"$\Delta\tau / (\alpha\,\tau_{\max})$"),
}

full_width()


def normalised(robot, Xc_leg, U, nq, N, T, d=D_COLLOC):
    """Constrained quantities scaled so their bounds are +-1: ``({key: array}, alpha)``.

    The asymmetric angle box is mapped via its midpoint and half-range.
    """
    q_lb, q_ub, v_ub, tau_ub = limits_for(robot)
    mid, half = (q_ub + q_lb) / 2.0, (q_ub - q_lb) / 2.0

    # Bandwidth filter coefficient, as in the OCP (depends on the solved T)
    dt = float(T) / N
    f_c = robot.spec.ocp.f_c
    alpha = 2 * np.pi * dt * f_c / (2 * np.pi * dt * f_c + 1)
    # Cyclic predecessor, as in the OCP
    prev = np.roll(U, 1, axis=1)

    return {
        "angle": (Xc_leg[7:nq, :] - mid[:, None]) / half[:, None],
        "rate": Xc_leg[nq + 6:, :] / v_ub[:, None],
        "torque": U / tau_ub[:, None],
        "bandwidth": (U - (1 - alpha) * prev) / (alpha * tau_ub[:, None]),
    }, alpha


def dwell_masks(robot, X, U, nq) -> dict:
    """Per-joint masks of dwelling on each bound at the grid nodes.

    Uses the N equally spaced nodes (last node omitted, it repeats node 0), so
    the shares are shares of time. Lower and upper angle stops are separate.
    """
    q_lb, q_ub, v_ub, tau_ub = limits_for(robot)
    mid, half = (q_ub + q_lb) / 2.0, (q_ub - q_lb) / 2.0
    band = 2.0 * DWELL_FRAC

    qn = (X[7:nq, :-1] - mid[:, None]) / half[:, None]
    rn = X[nq + 6:, :-1] / v_ub[:, None]
    return {
        "angle_lo": qn <= -1.0 + band,
        "angle_hi": qn >= 1.0 - band,
        "rate": np.abs(rn) >= 1.0 - band,
        "torque": np.abs(U / tau_ub[:, None]) >= 1.0 - band,
    }


def activity(masks: dict) -> dict:
    """Share of the cycle in which any joint dwells on each bound."""
    return {
        "angle": float((masks["angle_lo"] | masks["angle_hi"]).any(axis=0).mean()),
        "rate": float(masks["rate"].any(axis=0).mean()),
        "torque": float(masks["torque"].any(axis=0).mean()),
    }


def plot_traces(norm, phase, N, kinds):
    """Normalised constrained quantities over one cycle."""
    fig, axes = plt.subplots(len(LIMITS), 1, sharex=True,
                             figsize=(TEXT_WIDTH_IN, 1.45 * len(LIMITS) + 0.5))

    # Torque quantities are piecewise constant per interval, drawn as steps
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
    """Dwell share per bound vs speed (call within a ``half_width()`` context)."""
    fig, ax = plt.subplots(figsize=HALF)

    v = np.array([p["speed"] for p in points])
    pinned = np.array([p["pinned"] for p in points])
    for i, key in enumerate(DWELL_LIMITS):
        label = LIMITS[key][0]
        share = np.array([p["activity"][key] for p in points]) * 100.0
        colour = PALETTE[i % len(PALETTE)]
        ax.plot(v, share, color=colour, lw=1.2, alpha=0.85, zorder=2, label=label)
        # Open markers: T pinned at the window edge
        ax.plot(v[~pinned], share[~pinned], color=colour, linestyle="none",
                marker="o", markersize=4.5, zorder=3)
        ax.plot(v[pinned], share[pinned], color=colour, linestyle="none",
                marker="o", markersize=4.5, markerfacecolor="none",
                markeredgewidth=0.9, zorder=3)

    ax.set_xlabel(r"forward speed [m\,s$^{-1}$]")
    ax.set_ylabel(r"share of the cycle on the bound [\%]")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.margins(x=0.05)

    marks = [Line2D([], [], color="0.35", linestyle="none", marker="o",
                    markersize=4.5, markerfacecolor="none", markeredgewidth=0.9,
                    label=rf"$T$ pinned ($\pm{band:.2f}$ s)")]
    handles, _ = ax.get_legend_handles_labels()
    # Two legend columns fit the half-width canvas
    ax.legend(handles=handles + marks, loc="lower center",
              bbox_to_anchor=(0.5, 1.0), ncol=2,
              columnspacing=1.2, handlelength=1.6, frameon=False,
              borderaxespad=0.2)
    legend_row(fig, ax, rows=2)
    fig.tight_layout(pad=0.3)
    return fig


def measure(robot, meta):
    """``(norm, phase, alpha)`` for one solution."""
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
    """Print peak utilisation and active share per limit; exit if a bound is violated."""
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
        raise SystemExit(
            f"\n{LIMITS[worst][0]} exceeds its bound by {over[worst]:.2e} "
            f"(normalised), past the {ACTIVE_TOL:.0e} tolerance — this solution "
            f"does not satisfy the limits it is being measured against.")


def sweep_activity(robot, meta, results):
    """``(points, band)`` for the activity figure, or None (with the reason printed)."""
    try:
        band = sweep_io.detect_band(sweep_io.load_rows(results))
        pairs = sweep_io.load_sweep(results)
    except SystemExit as exc:
        print(f"\nNo activity figure — {exc}")
        return None

    # Build the robot once (reuse the traces robot if it matches)
    names = {m["robot"] for _, m in pairs}
    if len(names) > 1:
        raise SystemExit(f"sweep mixes robots ({', '.join(sorted(names))})")
    sweep_robot = robot if names == {meta["robot"]} else load_robot(names.pop())

    print(f"\nPareto front from {results} "
          f"({len(pairs)} solves, free-T window +/- {band:.3f} s):")
    points = []
    for row, m in pairs:
        masks = dwell_masks(sweep_robot, m["X"], m["U"], m["nq"])
        points.append({"speed": row["speed"],
                       "pinned": sweep_io.is_pinned(row, band),
                       "activity": activity(masks)})
        shares = "  ".join(f"{k} {points[-1]['activity'][k]:5.1%}"
                           for k in DWELL_LIMITS)
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
        # Scope the half-width rcParams to this figure
        with plt.rc_context():
            half_width()
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
