#!/usr/bin/env python3
"""Whole-robot forward-force budget over one optimised swim cycle.

Every other thrust figure here is quasi-steady *drag* on one leg:
``thrust_heatmap.py`` maps drag-thrust capability per pose, and
``gait_diagnostics.py`` evaluates the same drag model along the solution for the
three links of a single leg.  Neither includes added mass, neither covers the
whole robot, and neither closes a force balance.  This figure is the budget
those two are slices of.  The decomposition itself, and the frame it is taken
in, are ``force_budget.py``; where it is sampled and how it is averaged are
``collocation.py``.

Two figures, both on the full text-width canvas so they go in unscaled:

  ``*_traces``  The world-frame forward force on the whole robot over the cycle,
      split into the terms of the equation of motion.  Drawn twice, because the
      solution is periodic and the repeat is what makes that visible.

  ``*_means``   The cycle mean of each term.  This is where the result reads:
      which mechanism actually carries the forward force, and by how much.

Buoyancy and gravity collapsing to zero (1e-17) in the world frame is not
decoration: a vertical force cannot push the robot forward, so it is a free
correctness check on the whole pipeline, and it is drawn.

**Why the external forces do not sum to zero, and why the cycle-mean drag
should not be read as net resistance.**

Over a periodic cycle the state returns to itself, so the momentum does too, so
the net external force *must* average to zero.  It does not here: the
cycle-mean drag is -0.107 N.  That number is not a measurement of resistance.
It is the exact mirror of a momentum defect in the added-mass model, and the
equation of motion has no choice but to balance one against the other:

    mean(M_rb a + C_rb v)  =  -2e-4 N     <- rigid body conserves momentum
                                             (quadrature error; the exact
                                              statement is p_x(T) - p_x(0))
    mean(M_A  a + C_A  v)  =  -0.107 N    <- added mass does not
    mean(drag)             =  -0.107 N    <- forced equal to their sum

The cause is the submersion ratio.  ``dynamics.py`` builds ``C_A v`` as the
per-link Kirchhoff force at zero acceleration rather than from the Christoffel
symbols of ``M_A(q)``, which drops the d(alpha)/dq Coriolis contribution — its
own comment says so.  That was verified here rather than taken on trust:
rebuilding the identical model with ``z_surface`` far above the robot, so
alpha == 1 and d(alpha)/dq == 0, moves the added-mass mean from **-0.1067 N to
+0.0055 N** — a factor of twenty, down to quadrature noise.  Everything else is
unchanged, so the defect is the free-surface term and nothing else.

This robot swims at the surface: the hull averages 13% submerged and the front
legs leave the water every cycle, so alpha swings hard and the dropped term is
large.  Consistently, ``plot_thrust_attribution.py`` finds the hull — the link
with the most extreme alpha variation — carries -0.086 of the -0.107.

For the thesis this means the *shape* of the budget over the cycle is sound,
and so is the per-link attribution of who thrusts and who resists; but the
cycle-mean net force is dominated by a modelling artifact of the same size, and
should not be quoted as a physical result until ``C_A`` carries the d(alpha)/dq
term.  The console prints the defect on its own line so the figure cannot be
read without it.

Usage:
  python plot_thrust_budget.py
  python plot_thrust_budget.py --solution ../task3_solution.npz --save budget.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "stage1_gait_optimization"))
from collocation import D_COLLOC, cycle_mean, solution_states  # noqa: E402
from force_budget import (  # noqa: E402
    EOM_TOL,
    EXT_FORCE_TOL,
    MOMENTUM_TOL,
    TERMS,
    force_terms,
    momentum_residual,
)
from thesis_style import TEXT_WIDTH_IN, full_width, legend_row  # noqa: E402

from stage1_gait_optimization.hydro_model import SymbolicDynamics, load_robot  # noqa: E402
from stage1_gait_optimization.hydro_model.trajectory import load_solution  # noqa: E402

full_width()


def plot_traces(terms, phase):
    """Figure 1: every term of the forward-force balance over the cycle."""
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 2.9))

    # Collocation points are unevenly spaced inside each interval, so the phase
    # axis comes from the Radau roots rather than a linspace.
    two = np.concatenate([phase, 1 + phase])
    rep = lambda a: np.concatenate([a, a])                 # noqa: E731

    ax.axhline(0.0, color="0.75", lw=0.5, zorder=1)
    ax.axvline(1.0, color="0.6", lw=0.5, ls="--", zorder=1)

    n_flat = 0
    for key, (label, colour, external) in TERMS.items():
        # Buoyancy and gravity are flat on zero here, which is the point: they
        # are drawn thick and dashed so a reader sees two lines on the axis
        # rather than wondering where they went.  Both lie on exactly the same
        # pixels, so the dash phases are staggered — otherwise whichever is
        # drawn second hides the other completely and the check reads as one
        # term, not two.
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
    """Figure 2: the cycle mean of each term, which is what nets out."""
    keys = list(TERMS)
    means = np.array([cycle_mean(terms[k], N) for k in keys])
    # Largest first, so the mechanism that carries the force is at the top.
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

    # Value at the end of each bar: the two static terms are zero to 1e-17 and
    # a bar of that length is invisible, so the number has to carry them.
    span = max(np.abs(means).max(), 1e-12)
    for i, m in enumerate(means[order]):
        # The static terms land at ~1e-17.  "-0.000" reads as a rounded-away
        # small number; they are zero to machine precision, so say so.
        txt = "0" if abs(m) < 1e-9 else f"{m:+.3f}"
        ax.annotate(txt, xy=(m, i), fontsize=6.5,
                    xytext=(3 if m >= 0 else -3, 0), textcoords="offset points",
                    ha="left" if m >= 0 else "right", va="center", color="0.25")
    ax.set_xlim(-1.35 * span, 1.35 * span)
    fig.tight_layout(pad=0.3)
    return fig


def report(terms, resid, T, N, inertia, v_mean):
    """The numbers a caption needs, printed rather than drawn.

    The two closure lines at the end are the ones worth reading.  ``rigid-body
    momentum`` is the scale to read the next line against: ``M_rb a + C_rb v``
    is a true momentum derivative and the state is periodic, so its cycle mean
    is nothing but the quadrature's own truncation error (order 1e-3 N; the
    exact statement is checked in ``main`` on the momentum itself).
    ``added-mass momentum defect`` is the amount by which the added-mass terms
    fail the same test, and it is two orders larger.  It is the reason the external forces do not sum to zero,
    and it comes from the dropped d(alpha)/dq term — see the module docstring for
    the free-surface test that pins it down.  Because the equation of motion has
    to balance, the cycle-mean drag is forced equal to this defect: the two lines
    should print as near-equal numbers, and when they do, the mean drag is
    reporting the artifact rather than any resistance the robot actually feels.
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
        # Steady swimming means zero mean force, not positive: the robot holds
        # its speed rather than accelerating.  Quoting what this force *would*
        # do makes the claim falsifiable instead of a bare assertion -- the
        # trajectory is periodic, so a deceleration this large plainly is not
        # happening, which is what proves the number is bookkeeping.
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
    # Effective surge inertia and mean speed, for the diagnostic note.
    inertia = float(pin.computeTotalMass(robot.model)
                    + np.array(dyn.f_M_added(X[:nq, 0]))[0, 0])
    v_mean = float(X[0, -1] - X[0, 0]) / float(T)
    report(terms, resid, T, N, inertia, v_mean)

    # Two different checks.  The first is an identity and only catches frame or
    # sign slips; the second has physical content — the rigid body conserves
    # momentum, so a cycle that does not give it back is not a periodic solve.
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
