#!/usr/bin/env python3
"""Mechanism metrics vs speed along the Pareto front: why COT rises with speed.

Three panels, one point per solve:
  - peak-to-peak forward force from drag and from added mass
  - negative work as a share of the charged work
  - stroke benefit (drag gain vs holding the leg at its mean pose), front and hind

Force amplitudes are used instead of cycle means, which are not meaningful in
this model (see plot_thrust_budget.py). The stroke benefit is a difference of
two drag evaluations and is unaffected.

Usage:
  python stage3_visualization/speed_sweep/plot_mechanism_vs_speed.py
  python stage3_visualization/speed_sweep/plot_mechanism_vs_speed.py --results /path/to/codesign_results --save mechanism.pdf
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
    PALETTE,
    TEXT_WIDTH_IN,
    full_width,
)

from stage3_visualization.common import sweep_io  # noqa: E402
from stage3_visualization.common.collocation import (  # noqa: E402
    collocation_states,
    cycle_integral,
    joint_power,
)
from stage3_visualization.common.drag_model import stroke_benefit  # noqa: E402
from stage3_visualization.common.force_budget import MOMENTUM_TOL, force_terms, momentum_residual  # noqa: E402

from stage1_gait_optimization.hydro_model import SymbolicDynamics, load_robot  # noqa: E402

full_width()


def measure(robot, dyn, meta):
    """Mechanism metrics of one solution (Xc presence is checked by sweep_io)."""
    X, U, T, N, nq = meta["X"], meta["U"], meta["T"], meta["N"], meta["nq"]
    Xc_leg, _, _ = collocation_states(robot, X, meta["Xc"], nq, N)

    terms, resid = force_terms(robot, dyn, Xc_leg, U, N, nq)
    # Reject solutions that are not periodic in momentum.
    dp = momentum_residual(dyn, X, nq)
    if abs(dp) > MOMENTUM_TOL:
        raise SystemExit(
            f"\nrigid-body forward momentum changes by {dp:.3e} kg m/s over the "
            f"cycle, past {MOMENTUM_TOL:.0e} — this solution is not periodic.")

    P = joint_power(meta["Xc"], U, robot.n_actuated, N)
    charged = cycle_integral(np.abs(P).sum(axis=0), T, N)
    negative = -cycle_integral(np.clip(P, None, 0.0).sum(axis=0), T, N)

    benefit = stroke_benefit(robot, Xc_leg, nq, N)
    front = [s - f for leg, (s, f) in benefit.items() if leg.startswith("Front")]
    hind = [s - f for leg, (s, f) in benefit.items() if leg.startswith("Hind")]

    return {
        "drag_pp": float(terms["drag"].ptp()),
        "added_pp": float((terms["MA_a"] + terms["CAv"]).ptp()),
        "charged": charged,
        "neg_share": negative / charged if charged else np.nan,
        "benefit_front": float(np.mean(front)) if front else np.nan,
        "benefit_hind": float(np.mean(hind)) if hind else np.nan,
        "resid": float(np.abs(resid).max()),
    }


def plot_mechanism(points, band):
    """Three mechanism panels vs speed."""
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH_IN, 2.3))
    v = np.array([p["speed"] for p in points])
    pinned = np.array([p["pinned"] for p in points])

    def series(ax, y, colour, label):
        ax.plot(v, y, color=colour, lw=1.2, alpha=0.85, zorder=2, label=label)
        ax.plot(v[~pinned], y[~pinned], color=colour, linestyle="none",
                marker="o", markersize=4.0, zorder=3)
        ax.plot(v[pinned], y[pinned], color=colour, linestyle="none", marker="o",
                markersize=4.0, markerfacecolor="none", markeredgewidth=0.9,
                zorder=3)

    ax = axes[0]
    series(ax, np.array([p["drag_pp"] for p in points]), PALETTE[2], "drag")
    series(ax, np.array([p["added_pp"] for p in points]), PALETTE[0], "added mass")
    ax.set_ylabel(r"force amplitude [N]")
    ax.legend(loc="best", fontsize=6.5, frameon=True, framealpha=0.85,
              handlelength=1.4)

    ax = axes[1]
    series(ax, np.array([p["neg_share"] for p in points]) * 100.0, PALETTE[4],
           "negative work")
    ax.set_ylabel(r"negative work [\% of charged]")

    ax = axes[2]
    ax.axhline(0.0, color="0.45", lw=0.8, ls="--", zorder=1)
    series(ax, np.array([p["benefit_front"] for p in points]), PALETTE[1], "front")
    series(ax, np.array([p["benefit_hind"] for p in points]), PALETTE[3], "hind")
    ax.set_ylabel(r"stroke benefit [N]")
    ax.legend(loc="best", fontsize=6.5, frameon=True, framealpha=0.85,
              handlelength=1.4)

    for ax in axes:
        ax.set_xlabel(r"forward speed [m\,s$^{-1}$]")
        ax.grid(alpha=0.3)
        ax.margins(x=0.08)

    marks = [Line2D([], [], color="0.35", linestyle="none", marker="o",
                    markersize=4.0, markerfacecolor="none", markeredgewidth=0.9,
                    label=rf"$T$ pinned at edge ($\pm{band:.2f}$ s)")]
    fig.legend(handles=marks, loc="lower center", ncol=1, fontsize=7,
               frameon=False, handlelength=1.8)
    fig.tight_layout(pad=0.4, rect=(0, 0.10, 1, 1))
    return fig


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=sweep_io.DEFAULT_RESULTS,
                    help="sweep directory holding codesign_summary.json")
    ap.add_argument("--gaits", nargs="+", default=None, help="keep only these gaits")
    ap.add_argument("--save", type=Path, default=None,
                    help="write the figure here (format from the extension)")
    args = ap.parse_args()

    band = sweep_io.detect_band(sweep_io.load_rows(args.results))
    pairs = sweep_io.load_sweep(args.results, gaits=args.gaits)
    print(f"{len(pairs)} Pareto-front solves from {args.results}")

    # Build robot and dynamics once for the whole front
    robots = {m["robot"] for _, m in pairs}
    if len(robots) > 1:
        raise SystemExit(f"sweep mixes robots ({', '.join(sorted(robots))})")
    robot = load_robot(robots.pop())
    dyn = SymbolicDynamics(robot)

    print(f"  {'v [m/s]':>8s}{'COT':>7s}{'drag pp':>9s}{'added pp':>9s}"
          f"{'neg work':>10s}{'benefit F':>11s}{'benefit H':>11s}")
    points = []
    for row, meta in pairs:
        m = measure(robot, dyn, meta)
        m["speed"] = row["speed"]
        m["pinned"] = sweep_io.is_pinned(row, band)
        points.append(m)
        print(f"  {row['speed']:>8.3f}{row['cot']:>7.2f}{m['drag_pp']:>9.3f}"
              f"{m['added_pp']:>9.3f}{m['neg_share']:>9.1%}"
              f"{m['benefit_front']:>11.3f}{m['benefit_hind']:>11.3f}")

    fig = plot_mechanism(points, band)
    if args.save:
        fig.savefig(args.save, dpi=300, bbox_inches="tight", pad_inches=0.15)
        print(f"Saved → {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
