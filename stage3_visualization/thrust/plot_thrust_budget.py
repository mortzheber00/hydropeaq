#!/usr/bin/env python3
"""Whole-robot forward-force budget over one cycle, split into the EoM terms.

``*_traces``  world-frame forward force of each term over two cycles
``*_means``   cycle mean of each term

Buoyancy and gravity vanish in the world frame, which serves as a sanity check.

Caveat: over a periodic cycle the net external force should average to zero,
but the mean drag balances a momentum defect of the added-mass model instead.
``C_A v`` omits the d(alpha)/dq term, which matters because the robot swims at
the surface (with alpha == 1 the defect disappears). The per-cycle shape and
the per-link attribution are valid; the cycle-mean drag is not a physical
resistance. The defect is printed.

Usage:
  python stage3_visualization/thrust/plot_thrust_budget.py
  python stage3_visualization/thrust/plot_thrust_budget.py --solution task3_solution.npz --save budget.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "stage1_gait_optimization"))
from stage3_visualization.common.collocation import D_COLLOC, cycle_mean, solution_states  # noqa: E402
from stage3_visualization.common.force_budget import (  # noqa: E402
    EOM_TOL,
    EXT_FORCE_TOL,
    MOMENTUM_TOL,
    TERMS,
    force_terms,
    momentum_residual,
)
from stage3_visualization.common.thesis_style import TEXT_WIDTH_IN, full_width, legend_row  # noqa: E402

from stage1_gait_optimization.hydro_model import SymbolicDynamics, load_robot  # noqa: E402
from stage1_gait_optimization.hydro_model.trajectory import load_solution  # noqa: E402

full_width()


def plot_traces(terms, phase):
    """All forward-force terms over two cycles."""
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 2.9))

    two = np.concatenate([phase, 1 + phase])
    rep = lambda a: np.concatenate([a, a])                 # noqa: E731

    ax.axhline(0.0, color="0.75", lw=0.5, zorder=1)
    ax.axvline(1.0, color="0.6", lw=0.5, ls="--", zorder=1)

    n_flat = 0
    for key, (label, colour, external) in TERMS.items():
        # Zero terms (buoyancy, gravity): thick dashes with staggered phase so both stay visible
        flat = np.allclose(terms[key], 0.0, atol=1e-12)
        if flat:
            style = dict(lw=1.8, ls=(4 * n_flat, (4, 4)), zorder=2)
            n_flat += 1
        else:
            style = dict(lw=1.0, ls="-", zorder=3)
        ax.plot(two, rep(terms[key]), color=colour, alpha=0.9, label=label,
                **style)

    ax.set_xlabel("cycle phase")
    ax.set_ylabel(r"forward force $F_x^{\mathrm{world}}$ [N]")
    ax.set_xlim(0, 2)
    ax.set_xticks([0, 0.5, 1, 1.5, 2])
    ax.grid(alpha=0.3)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
              handlelength=1.8, frameon=False, columnspacing=1.2,
              borderaxespad=0.2, fontsize=7)

    legend_row(fig, ax, rows=3)
    fig.tight_layout(pad=0.3)
    return fig


def plot_means(terms, N):
    """Cycle mean of each term, largest first."""
    keys = list(TERMS)
    means = np.array([cycle_mean(terms[k], N) for k in keys])
    order = np.argsort(-np.abs(means))

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 2.4))
    y = np.arange(len(keys))
    ax.barh(y, means[order], color=[TERMS[keys[i]][1] for i in order],
            height=0.68, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([TERMS[keys[i]][0] for i in order], fontsize=7.5)
    ax.invert_yaxis()
    ax.axvline(0.0, color="0.4", lw=0.7, zorder=4)
    ax.set_xlabel(r"cycle-mean forward force $\bar F_x^{\mathrm{world}}$ [N]")
    ax.grid(axis="x", alpha=0.3)

    # Value labels (zero-length bars would otherwise be invisible)
    span = max(np.abs(means).max(), 1e-12)
    for i, m in enumerate(means[order]):
        # Print machine-zero as "0" rather than "-0.000"
        txt = "0" if abs(m) < 1e-9 else f"{m:+.3f}"
        ax.annotate(txt, xy=(m, i), fontsize=6.5,
                    xytext=(3 if m >= 0 else -3, 0), textcoords="offset points",
                    ha="left" if m >= 0 else "right", va="center", color="0.25")
    ax.set_xlim(-1.35 * span, 1.35 * span)
    fig.tight_layout(pad=0.3)
    return fig


def report(terms, resid, T, N, inertia, v_mean):
    """Print term statistics and the momentum closure.

    The rigid-body mean is only quadrature error and sets the scale; the
    added-mass defect is the artifact that the mean drag balances (see module
    docstring).
    """
    print(f"  {'term':<34s}{'mean [N]':>11s}{'pk-pk [N]':>11s}")
    for key, (label, _, _) in TERMS.items():
        a = terms[key]
        print(f"  {key:<34s}{cycle_mean(a, N):>11.4f}{a.ptp():>11.4f}")

    ext = sum(cycle_mean(terms[k], N) for k, (_, _, e) in TERMS.items() if e)
    rb = cycle_mean(terms["Mrb_a"], N) + cycle_mean(terms["Crb_v"], N)
    am = cycle_mean(terms["MA_a"], N) + cycle_mean(terms["CAv"], N)
    print(f"\n  net external forward force      {ext:>11.4f} N")
    print(f"  rigid-body momentum (quadrature){rb:>11.2e} N")
    print(f"  added-mass momentum defect      {am:>11.4f} N")
    print(f"  max |EoM identity residual|     {np.abs(resid).max():>11.3e} N")
    print(f"  cycle period T                  {float(T):>11.4f} s")
    if abs(ext) > EXT_FORCE_TOL:
        # Speed loss this force would cause, which a periodic solution cannot have
        lost = ext / inertia * float(T)
        print(f"\n  NOTE: the net external force should average to zero over a "
              f"periodic\n  cycle and does not.  It sits within "
              f"{abs(ext - am):.1e} N of the added-mass defect\n  above, so it is "
              f"that modelling artifact, not resistance.  On {inertia:.2f} kg it\n"
              f"  would cost {abs(lost):.3f} m/s per cycle "
              f"({abs(lost) / v_mean * 100:.0f}% of the {v_mean:.3f} m/s mean\n"
              f"  speed) -- and the solution is periodic, so no such deceleration "
              f"occurs.\n  Do not quote the cycle-mean drag as a physical number; "
              f"the per-cycle\n  shape and the per-link attribution are "
              f"unaffected.")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--solution", type=Path, default=_ROOT / "task3_solution.npz")
    ap.add_argument("--save", type=Path, default=None,
                    help="output stem; each figure gets its own suffixed file, "
                         "and the extension picks the format")
    args = ap.parse_args()

    meta = load_solution(args.solution)
    robot = load_robot(meta["robot"])
    dyn = SymbolicDynamics(robot)
    X, U, T, N, nq = meta["X"], meta["U"], meta["T"], meta["N"], meta["nq"]
    Xc_leg, phase, xc_err = solution_states(robot, meta, args.solution.name)

    terms, resid = force_terms(robot, dyn, Xc_leg, U, N, nq)

    print(f"{args.solution.name}:")
    print(f"  (sampled at {N * D_COLLOC} collocation points; "
          f"Xc reconstruction error {xc_err:.1e})")
    # Surge inertia and mean speed for the note in report()
    inertia = float(pin.computeTotalMass(robot.model)
                    + np.array(dyn.f_M_added(X[:nq, 0]))[0, 0])
    v_mean = float(X[0, -1] - X[0, 0]) / float(T)
    report(terms, resid, T, N, inertia, v_mean)

    # EoM identity (catches frame/sign errors) and momentum periodicity
    worst = np.abs(resid).max()
    if worst > EOM_TOL:
        raise SystemExit(
            f"\nEoM residual {worst:.3e} N exceeds {EOM_TOL:.0e} — the force "
            f"decomposition does not close, so the figure would be wrong.")
    dp = momentum_residual(dyn, X, nq)
    if abs(dp) > MOMENTUM_TOL:
        raise SystemExit(
            f"\nrigid-body forward momentum changes by {dp:.3e} kg m/s over the "
            f"cycle, past {MOMENTUM_TOL:.0e}.  That term is a momentum "
            f"derivative and the state is meant to be periodic, so this solve "
            f"never closed its periodicity constraints.")

    figs = {"traces": plot_traces(terms, phase), "means": plot_means(terms, N)}
    if args.save:
        for tag, fig in figs.items():
            path = args.save.with_name(f"{args.save.stem}_{tag}{args.save.suffix}")
            fig.savefig(path, dpi=300)
            print(f"Saved → {path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
