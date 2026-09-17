#!/usr/bin/env python3
"""Gait structure vs speed along the Pareto front.

Writes four half-width figures (``<name>_{duty,frequency,phase,depth}``):
  - duty factor: power-stroke share averaged over the legs, with per-leg range
  - stride frequency 1/T
  - inter-limb phase lag (front right vs front left, hind left vs front left),
    from circular means of the power spans so wrap-around does not cause jumps
  - depth separation: foot height in recovery minus in power (positive = power
    stroke deeper)

Sampled at the collocation points. ``--duty-ref LO HI`` adds a literature band.

Usage:
  python stage3_visualization/speed_sweep/plot_structure_vs_speed.py
  python stage3_visualization/speed_sweep/plot_structure_vs_speed.py --results /path/to/codesign_results
  python stage3_visualization/speed_sweep/plot_structure_vs_speed.py --duty-ref 0.35 0.45 --save structure.pdf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin
from matplotlib.lines import Line2D

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "stage1_gait_optimization"))
from stage3_visualization.common.thesis_style import (  # noqa: E402,F401  activates the shared style
    HALF,
    PALETTE,
    half_width,
    legend_row,
)

from stage3_visualization.common import sweep_io  # noqa: E402
from stage3_visualization.common.collocation import D_COLLOC, collocation_states, power_spans  # noqa: E402
from stage3_visualization.common.drag_model import require_supported  # noqa: E402

from stage1_gait_optimization.hydro_model import load_robot  # noqa: E402
from stage1_gait_optimization.ocp_common import collocation_coefficients  # noqa: E402


# Phase panel pairs: (lagging leg, reference leg)
PAIRS = {
    "left--right (front)": ("Front_Right", "Front_Left"),
    "front--hind (left)": ("Hind_Left", "Front_Left"),
}

half_width()


def circular_centre(spans) -> float:
    """Width-weighted circular mean of the span midpoints, in [0, 1)."""
    if not spans:
        return float("nan")
    z = sum(w * np.exp(2j * np.pi * (s + w / 2.0)) for s, w in spans)
    return float(np.angle(z) / (2 * np.pi) % 1.0)


def leg_metrics(robot, Xc_leg, phase, nq, N, leg, d=D_COLLOC):
    """``(duty, centre, depth_sep)`` for one leg.

    ``depth_sep`` [m] is the weighted mean hip-relative foot height (base frame)
    in recovery minus that in power.
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
        # Hull-relative foot velocity (base twist zeroed), in the base frame
        J = pin.computeFrameJacobian(robot.model, robot.data, q, foot_fid,
                                     pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        v_rel = np.asarray(v, dtype=float).copy()
        v_rel[:6] = 0.0
        vx[col] = (R_b.T @ (J[:3, :] @ v_rel))[0]

    spans = power_spans(phase, vx)
    w = np.array([B[col % d] for col in range(n_col)])
    power = vx < 0
    # Radau-weighted phase means
    z_pow = float((w * z)[power].sum() / w[power].sum()) if power.any() else np.nan
    z_rec = float((w * z)[~power].sum() / w[~power].sum()) if (~power).any() else np.nan
    return sum(width for _, width in spans), circular_centre(spans), z_rec - z_pow


def measure(robot, meta):
    """Structure metrics of one solution (Xc presence is checked by sweep_io)."""
    X, T, N, nq = meta["X"], meta["T"], meta["N"], meta["nq"]
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
    """One half-width figure per metric vs speed, as ``{name: fig}``."""
    v = np.array([p["speed"] for p in points])
    pinned = np.array([p["pinned"] for p in points])
    figs = {}

    def panel(name):
        fig, ax = plt.subplots(figsize=HALF)
        figs[name] = fig
        ax.grid(alpha=0.3)
        ax.margins(x=0.06)
        ax.set_xlabel(r"forward speed [m\,s$^{-1}$]")
        return ax

    def series(ax, y, colour, label=None):
        ax.plot(v, y, color=colour, lw=1.2, alpha=0.85, zorder=2, label=label)
        ax.plot(v[~pinned], y[~pinned], color=colour, linestyle="none",
                marker="o", markersize=4.5, zorder=3)
        ax.plot(v[pinned], y[pinned], color=colour, linestyle="none", marker="o",
                markersize=4.5, markerfacecolor="none", markeredgewidth=0.9,
                zorder=3)

    def pinned_mark():
        return Line2D([], [], color="0.35", linestyle="none", marker="o",
                      markersize=4.5, markerfacecolor="none",
                      markeredgewidth=0.9,
                      label=rf"$T$ pinned ($\pm{band:.2f}$ s)")

    def legend(ax, handles, **kw):
        # Framed by default so the legend masks the data beneath it
        opts = dict(frameon=True, framealpha=0.92, edgecolor="0.8")
        opts.update(kw)
        return ax.legend(handles=handles, **opts)

    # --- Duty factor ---
    ax = panel("duty")
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
                fontsize=7, color="0.45", va="top")
    # Keep the 0.5 reference line in view
    ax.set_ylim(top=max(ax.get_ylim()[1], 0.53))
    ax.set_ylabel("duty factor [-]")
    handles = [pinned_mark()]
    if duty_ref:
        handles.append(Line2D([], [], color="0.85", lw=6,
                              label="reported for the dog paddle"))
    # Add headroom for the wide legend
    y0, y1 = ax.get_ylim()
    ax.set_ylim(top=y1 + 0.45 * (y1 - y0))
    legend(ax, handles, loc="upper center")

    # --- Stride frequency ---
    ax = panel("frequency")
    series(ax, np.array([p["freq"] for p in points]), PALETTE[1])
    # No offset notation, which would amplify solver noise for near-equal periods
    ax.ticklabel_format(axis="y", useOffset=False, style="plain")
    ax.set_ylabel(r"stride frequency $1/T$ [Hz]")
    legend(ax, [pinned_mark()], loc="best")

    # --- Inter-limb phase ---
    ax = panel("phase")
    for i, name in enumerate(PAIRS):
        y = np.array([p["phases"].get(name, np.nan) for p in points])
        if np.isfinite(y).any():
            series(ax, y, PALETTE[2 + i], label=name)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("phase lag [cycles]")
    # Legend above the axes (the y range is fixed to one cycle)
    legend(ax, ax.get_legend_handles_labels()[0] + [pinned_mark()],
           loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
           frameon=False, handlelength=1.4, columnspacing=1.2,
           borderaxespad=0.2)
    legend_row(ax.figure, ax, rows=2)

    # --- Depth separation ---
    ax = panel("depth")
    ax.axhline(0.0, color="0.45", lw=0.8, ls="--", zorder=1)
    series(ax, np.array([p["depth"] for p in points]), PALETTE[4])
    ax.set_ylabel("power stroke deeper by [mm]")
    legend(ax, [pinned_mark()], loc="best")

    for fig in figs.values():
        fig.tight_layout(pad=0.3)
    return figs


def check_against_grid(robot, meta):
    """Max duty difference between ``power_spans`` and ``plot_solution_legs.power_segments`` on the grid nodes."""
    from stage3_visualization.gait.plot_solution_legs import leg_traces, power_segments

    X, nq, N = meta["X"], meta["nq"], meta["N"]
    worst = 0.0
    for leg in robot.spec.leg_names:
        _, _, vx = leg_traces(robot, X, nq, leg)
        ref = sum(w for _, w in power_segments(vx))
        # Drop the repeated last node
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
                    help="write the figures here, one per metric, with the "
                         "metric appended to the stem (format from the "
                         "extension)")
    args = ap.parse_args()

    band = sweep_io.detect_band(sweep_io.load_rows(args.results))
    pairs = sweep_io.load_sweep(args.results, pareto_only=not args.all_points,
                                gaits=args.gaits)
    print(f"{len(pairs)} solves from {args.results}, "
          f"free-T window +/- {band:.3f} s")

    # Build the robot once
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

    figs = plot_structure(points, band, args.duty_ref)
    if args.save:
        for name, fig in figs.items():
            path = args.save.with_name(f"{args.save.stem}_{name}{args.save.suffix}")
            # Uncropped half-width canvas (see half_width())
            fig.savefig(path, dpi=300)
            print(f"Saved → {path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
