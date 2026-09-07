#!/usr/bin/env python3
"""How the optimised gait's structure changes with the commanded speed.

``plot_solution_legs.py`` draws the stroke geometry of one solution.  This
measures four numbers off *every* solution on the sweep's Pareto front and puts
them against the speed that solution was solved for, which is the question H3
actually asks: does the phase timing and the stride frequency shift with the
commanded speed, or stay fixed?

Four panels on one text-width canvas:

  duty factor      Fraction of the cycle the foot sweeps backwards, averaged
      over the four legs, with the per-leg spread as a bar.  The 0.5 line is
      drawn because the hypothesis is stated against it — below the line the
      power stroke is the shorter half of the cycle.
  stride frequency 1/T of the solved period.
  inter-limb phase Lag of the right front leg behind the left (a left-right
      pair), and of the left hind behind the left front (a front-hind pair).
  depth separation Mean foot height over the recovery stroke minus the mean over
      the power stroke.  Positive means the power stroke runs deeper, which is
      the arrangement that presents more area to the flow when pushing than when
      recovering.

**The phase is a circular quantity and is treated as one.**  Taking the start of
the first power segment and subtracting instead makes the number jump by a whole
cycle whenever the stroke straddles phase 0: on the shipped sweep that metric
read 0.99 at one speed and 0.26 at the next, which is wrap-around, not a change
in coordination.  Each leg's stroke here is reduced to a circular mean of its
power spans, weighted by span width, and differences are wrapped into [0, 1).

**Sampled at the collocation points**, so a stroke reversal is placed on ~3x as
many samples as the grid nodes carry.  ``power_spans`` therefore interpolates
crossings in *phase*, not in sample index — the Radau roots are unevenly spaced
inside each interval and an index-based interpolation would put every transition
in the wrong place.  Given the grid nodes it reproduces
``plot_solution_legs.power_segments`` to ~1e-12, which ``--check`` asserts.

The duty factor is a property of the *hypothesis*, not of the biology: no
literature value is baked in.  ``--duty-ref LO HI`` draws a reference band where
the source says it should be, and without it no band is drawn.

Usage:
  python plot_structure_vs_speed.py
  python plot_structure_vs_speed.py --results /path/to/codesign_results
  python plot_structure_vs_speed.py --duty-ref 0.35 0.45 --save structure.pdf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin
from matplotlib.lines import Line2D

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "stage1_gait_optimization"))
from thesis_style import (  # noqa: E402,F401  activates the shared style
    PALETTE,
    TEXT_WIDTH_IN,
    full_width,
)

import sweep_io  # noqa: E402
from collocation import D_COLLOC, collocation_states, power_spans  # noqa: E402
from drag_model import require_supported  # noqa: E402

from stage1_gait_optimization.hydro_model import load_robot  # noqa: E402
from stage1_gait_optimization.ocp_common import collocation_coefficients  # noqa: E402


# The two limb pairs the phase panel reports, as (lagging leg, reference leg).
PAIRS = {
    "left--right (front)": ("Front_Right", "Front_Left"),
    "front--hind (left)": ("Hind_Left", "Front_Left"),
}

full_width()


def circular_centre(spans) -> float:
    """Width-weighted circular mean of the span midpoints, in [0, 1).

    A single contiguous span — the usual case — gives exactly its midpoint.
    """
    if not spans:
        return float("nan")
    z = sum(w * np.exp(2j * np.pi * (s + w / 2.0)) for s, w in spans)
    return float(np.angle(z) / (2 * np.pi) % 1.0)


def leg_metrics(robot, Xc_leg, phase, nq, N, leg, d=D_COLLOC):
    """``(duty, centre, depth_sep)`` for one leg of one solution.

    ``depth_sep`` is the Radau-weighted mean foot height over the recovery
    stroke minus the same over the power stroke, in metres, in the base frame:
    positive when the power stroke runs deeper.  The foot height is taken
    relative to the hip, as ``plot_solution_legs`` draws it, so the base's own
    heave over the cycle does not enter.
    """
    _, _, _, B = collocation_coefficients(d)
    foot_fid = robot.foot_frame_ids[leg]
    n_col = Xc_leg.shape[1]

    vx = np.zeros(n_col)
    z = np.zeros(n_col)
    for col in range(n_col):
        q, v = Xc_leg[:nq, col], Xc_leg[nq:, col]
        robot.forward_kinematics(q)
        R_b = np.array(robot.data.oMi[1].rotation)
        pos = robot.leg_centerline_positions(leg)
        z[col] = (R_b.T @ (pos["foot"] - pos["side"]))[2]
        # Foot velocity relative to the hull, in the base frame — the same
        # definition gait_diagnostics.py and plot_solution_legs.py split the
        # stroke on.  Zeroing the base twist is what makes it hull-relative.
        J = pin.computeFrameJacobian(robot.model, robot.data, q, foot_fid,
                                     pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        v_rel = np.asarray(v, dtype=float).copy()
        v_rel[:6] = 0.0
        vx[col] = (R_b.T @ (J[:3, :] @ v_rel))[0]

    spans = power_spans(phase, vx)
    w = np.array([B[col % d] for col in range(n_col)])
    power = vx < 0
    # Conditional Radau-weighted means: a plain .mean() would weight the three
    # roots of every interval equally, which the quadrature does not.
    z_pow = float((w * z)[power].sum() / w[power].sum()) if power.any() else np.nan
    z_rec = float((w * z)[~power].sum() / w[~power].sum()) if (~power).any() else np.nan
    return sum(width for _, width in spans), circular_centre(spans), z_rec - z_pow


def measure(robot, meta):
    """Per-solution structure metrics, averaged or paired over the legs."""
    X, T, N, nq = meta["X"], meta["T"], meta["N"], meta["nq"]
    # sweep_io has already refused any solve without an Xc block.
    Xc_leg, phase, _ = collocation_states(robot, X, meta["Xc"], nq, N)

    per_leg = {leg: leg_metrics(robot, Xc_leg, phase, nq, N, leg)
               for leg in robot.spec.leg_names}
    duty = np.array([m[0] for m in per_leg.values()])
    depth = np.array([m[2] for m in per_leg.values()])
    phases = {}
    for name, (lag, ref) in PAIRS.items():
        if lag in per_leg and ref in per_leg:
            phases[name] = (per_leg[lag][1] - per_leg[ref][1]) % 1.0
    return {
        "duty": float(duty.mean()), "duty_lo": float(duty.min()),
        "duty_hi": float(duty.max()), "freq": 1.0 / float(T),
        "depth": float(np.nanmean(depth)) * 1e3,   # mm
        "phases": phases,
    }


def plot_structure(points, band, duty_ref):
    """The four structure panels against commanded speed."""
    fig, axes = plt.subplots(2, 2, figsize=(TEXT_WIDTH_IN, 4.2))
    v = np.array([p["speed"] for p in points])
    pinned = np.array([p["pinned"] for p in points])

    def series(ax, y, colour, label=None):
        ax.plot(v, y, color=colour, lw=1.2, alpha=0.85, zorder=2, label=label)
        ax.plot(v[~pinned], y[~pinned], color=colour, linestyle="none",
                marker="o", markersize=4.5, zorder=3)
        ax.plot(v[pinned], y[pinned], color=colour, linestyle="none", marker="o",
                markersize=4.5, markerfacecolor="none", markeredgewidth=0.9,
                zorder=3)

    # ── duty factor ─────────────────────────────────────────────────────────
    ax = axes[0, 0]
    duty = np.array([p["duty"] for p in points])
    lo = np.array([p["duty_lo"] for p in points])
    hi = np.array([p["duty_hi"] for p in points])
    if duty_ref:
        ax.axhspan(min(duty_ref), max(duty_ref), color="0.85", zorder=0)
    ax.axhline(0.5, color="0.45", lw=0.8, ls="--", zorder=1)
    ax.errorbar(v, duty, yerr=[duty - lo, hi - duty], fmt="none", ecolor=PALETTE[0],
                elinewidth=0.8, capsize=2, alpha=0.6, zorder=2)
    series(ax, duty, PALETTE[0])
    ax.annotate("half the cycle", xy=(0.02, 0.5), xycoords=("axes fraction", "data"),
                fontsize=6.5, color="0.45", va="top")
    # The hypothesis is stated against the 0.5 line, so it has to be on the
    # panel with room above it even when every solution sits well below.
    ax.set_ylim(top=max(ax.get_ylim()[1], 0.53))
    ax.set_ylabel("duty factor [-]")

    # ── stride frequency ────────────────────────────────────────────────────
    ax = axes[0, 1]
    series(ax, np.array([p["freq"] for p in points]), PALETTE[1])
    # Without this, a sweep whose periods happen to agree to float precision
    # (an N-continuation, say) gets an axis zoomed onto 1e-12 of solver noise
    # and an offset label, which reads as structure that is not there.
    ax.ticklabel_format(axis="y", useOffset=False, style="plain")
    ax.set_ylabel(r"stride frequency $1/T$ [Hz]")

    # ── inter-limb phase ────────────────────────────────────────────────────
    ax = axes[1, 0]
    for i, name in enumerate(PAIRS):
        y = np.array([p["phases"].get(name, np.nan) for p in points])
        if np.isfinite(y).any():
            series(ax, y, PALETTE[2 + i], label=name)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("phase lag [cycles]")
    ax.legend(loc="best", fontsize=6.5, frameon=True, framealpha=0.85,
              handlelength=1.4)

    # ── depth separation ────────────────────────────────────────────────────
    ax = axes[1, 1]
    ax.axhline(0.0, color="0.45", lw=0.8, ls="--", zorder=1)
    series(ax, np.array([p["depth"] for p in points]), PALETTE[4])
    ax.set_ylabel("power stroke deeper by [mm]")

    for ax in axes.flat:
        ax.grid(alpha=0.3)
        ax.margins(x=0.06)
    for ax in axes[1, :]:
        ax.set_xlabel(r"commanded speed $v$ [m\,s$^{-1}$]")

    marks = [Line2D([], [], color="0.35", linestyle="none", marker="o",
                    markersize=4.5, markerfacecolor="none", markeredgewidth=0.9,
                    label=rf"$T$ pinned at edge ($\pm{band:.2f}$ s)")]
    if duty_ref:
        marks.append(Line2D([], [], color="0.85", lw=6,
                            label="reported for the dog paddle"))
    fig.legend(handles=marks, loc="lower center", ncol=2, fontsize=7,
               frameon=False, columnspacing=1.4, handlelength=1.8)
    fig.tight_layout(pad=0.4, rect=(0, 0.06, 1, 1))
    return fig


def check_against_grid(robot, meta):
    """``power_spans`` on the grid nodes must equal ``power_segments`` there.

    The two split the stroke on the same rule; this one interpolates in phase
    and the other in sample index, which agree exactly when the samples are the
    uniformly spaced grid nodes.  It is the check that the phase-aware version
    did not quietly change the definition of a power stroke.
    """
    from plot_solution_legs import leg_traces, power_segments

    X, nq, N = meta["X"], meta["nq"], meta["N"]
    worst = 0.0
    for leg in robot.spec.leg_names:
        _, _, vx = leg_traces(robot, X, nq, leg)
        ref = sum(w for _, w in power_segments(vx))
        # leg_traces' last column repeats the first; drop it and put the
        # remaining N nodes on their own phases.
        phase = np.arange(N) / N
        got = sum(w for _, w in power_spans(phase, vx[:-1]))
        worst = max(worst, abs(ref - got))
    return worst


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=sweep_io.DEFAULT_RESULTS,
                    help="sweep directory holding codesign_summary.json")
    ap.add_argument("--gaits", nargs="+", default=None,
                    help="keep only these gaits")
    ap.add_argument("--all-points", action="store_true",
                    help="every feasible solve, not just the Pareto front")
    ap.add_argument("--duty-ref", type=float, nargs=2, default=None,
                    metavar=("LO", "HI"),
                    help="duty-factor band reported for the biological gait; "
                         "no default, because the number belongs to its source")
    ap.add_argument("--check", action="store_true",
                    help="assert power_spans matches plot_solution_legs on the "
                         "grid nodes, then continue")
    ap.add_argument("--save", type=Path, default=None,
                    help="write the figure here (format from the extension)")
    args = ap.parse_args()

    band = sweep_io.detect_band(sweep_io.load_rows(args.results))
    pairs = sweep_io.load_sweep(args.results, pareto_only=not args.all_points,
                                gaits=args.gaits)
    print(f"{len(pairs)} solves from {args.results}, "
          f"free-T window +/- {band:.3f} s")

    # One robot build for the whole sweep; it dominates the runtime here.
    names = {m["robot"] for _, m in pairs}
    if len(names) > 1:
        raise SystemExit(f"sweep mixes robots ({', '.join(sorted(names))})")
    name = names.pop()
    require_supported(name, "plot_structure_vs_speed")
    robot = load_robot(name)

    points = []
    for row, meta in pairs:
        if args.check:
            err = check_against_grid(robot, meta)
            if err > 1e-9:
                raise SystemExit(
                    f"power_spans disagrees with plot_solution_legs by {err:.2e} "
                    f"of a cycle on the grid nodes — the two are no longer "
                    f"splitting the stroke the same way.")
        m = measure(robot, meta)
        m["speed"] = row["speed"]
        m["pinned"] = sweep_io.is_pinned(row, band)
        points.append(m)
        lag = "  ".join(f"{k.split()[0]} {v:.3f}" for k, v in m["phases"].items())
        print(f"  v = {row['speed']:.3f} m/s  T = {row['T']:.3f} s   "
              f"duty {m['duty']:.3f} [{m['duty_lo']:.3f}, {m['duty_hi']:.3f}]  "
              f"f {m['freq']:.3f} Hz  depth {m['depth']:+.1f} mm  {lag}")
    if args.check:
        print("  (power_spans matches plot_solution_legs on the grid nodes)")

    fig = plot_structure(points, band, args.duty_ref)
    if args.save:
        fig.savefig(args.save, dpi=300, bbox_inches="tight", pad_inches=0.15)
        print(f"Saved → {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
