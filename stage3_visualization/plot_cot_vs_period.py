#!/usr/bin/env python3
"""
Cost of transport vs. cycle period at a fixed forward speed, one curve per gait.

Reads ``codesign_summary.json`` — the sweep summary written by
``codesign/run_codesign.py`` — and plots a single speed slice of it.  The inner
OCP minimises energy subject to an average-speed floor (ε-constraint), and the
floor binds to ~1e-7, so every point at one ``v_target`` really is at the same
forward speed and the only thing varying along the curve is the cadence.  The
figure is therefore the answer to "at this speed, what cycle period swims most
efficiently, and does the answer depend on the gait?".

COT here is the sweep's own definition (``codesign/solver.py``):

    COT = ∫|τ·q̇| dt / (m g d)

— mechanical work over one cycle per unit weight per unit distance travelled,
so it is dimensionless and comparable across gaits and speeds.

Two features of the sweep the figure makes explicit:

  - The period grid is over the *centre* T of a free-T refine window
    (``FREE_T_BAND``, ±0.15 s by default), and the solver may move T inside it.
    x is the solved period, so the samples are unevenly spaced.  Points whose T
    ended up pinned at a window edge are drawn open: their cadence is set by the
    window, not by an efficiency optimum, so they are not stationary points of
    the curve.  In the shipped sweep that is 57% of all solves.
  - Each point is an independent local NLP solution, so the connecting line is a
    guide to the eye, not an interpolation of one continuous branch.

Usage:
  python plot_cot_vs_period.py
  python plot_cot_vs_period.py --speed 0.15 --save ../docs/figures/cot_vs_period.pdf
  python plot_cot_vs_period.py --gaits LSPG25 LSPG33 --speed 0.25
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from thesis_style import style_for, tex  # also activates the shared plot style

DEFAULT_SUMMARY = (Path(__file__).resolve().parents[1] / "stage1_gait_optimization" /
                   "codesign" / "codesign_results" / "codesign_summary.json")
DEFAULT_SPEED = 0.2      # m/s — the pipeline's nominal design speed (V_TARGET)

# Tolerance on |T - t_center| for calling the free-T window active.
BAND_TOL = 1e-3


def load_slice(path: Path, speed: float, gaits=None):
    """Feasible rows at the ``v_target`` nearest ``speed``, grouped by gait.

    Returns ``(v_target, {gait: rows sorted by solved T}, n_dropped)``.
    """
    rows = json.loads(path.read_text())
    targets = sorted({r["v_target"] for r in rows})
    v = min(targets, key=lambda t: abs(t - speed))

    at_v = [r for r in rows if abs(r["v_target"] - v) < 1e-9]
    if gaits:
        at_v = [r for r in at_v if r["gait"] in gaits]
    good = [r for r in at_v
            if r["feasible"] and np.isfinite(r["cot"]) and r["cot"] > 0]

    order = gaits if gaits else sorted({r["gait"] for r in good})
    by_gait = {g: sorted((r for r in good if r["gait"] == g), key=lambda r: r["T"])
               for g in order}
    by_gait = {g: rs for g, rs in by_gait.items() if rs}
    if not by_gait:
        raise SystemExit(f"no feasible points at v_target = {v:.4f} m/s")
    return v, by_gait, len(at_v) - len(good)


def detect_band(path: Path) -> float:
    """Free-T half-window used by the sweep, read off the data.

    ``FREE_T_BAND`` lives in run_codesign.py and is not written to the summary,
    but every solve that hit the window sits exactly at ``t_center ± band``, so
    the largest observed excursion is the band.
    """
    rows = json.loads(path.read_text())
    devs = [abs(r["T"] - r["t_center"]) for r in rows if r["feasible"]]
    return max(devs) if devs else 0.0


def plot_cot_vs_period(by_gait: dict, v_target: float, band: float,
                       title: str | None):
    fig, ax = plt.subplots(figsize=(8.8, 4.2))

    for gait, rows in by_gait.items():
        colour, marker = style_for(gait)
        T = np.array([r["T"] for r in rows])
        cot = np.array([r["cot"] for r in rows])
        # Pinned at a window edge -> the cadence is the constraint's, not the
        # solver's choice.  Drawn open; interior optima drawn filled.
        pinned = np.array([abs(r["T"] - r["t_center"]) >= band - BAND_TOL
                           for r in rows])

        best = int(np.argmin(cot))
        ax.plot(T, cot, color=colour, lw=1.2, alpha=0.85, zorder=2,
                label=rf"{tex(gait)}: {cot[best]:.2f} at $T={T[best]:.2f}$ s")
        ax.plot(T[~pinned], cot[~pinned], color=colour, linestyle="none",
                marker=marker, markersize=5.5, zorder=3)
        ax.plot(T[pinned], cot[pinned], color=colour, linestyle="none",
                marker=marker, markersize=5.5, markerfacecolor="none",
                markeredgewidth=0.9, zorder=3)
        # Most efficient cadence for this gait.
        ax.plot(T[best], cot[best], color=colour, linestyle="none", marker="*",
                markersize=13, markeredgecolor="k", markeredgewidth=0.4, zorder=4)

    ax.set_xlabel(r"cycle period $T$ [s]")
    ax.set_ylabel(r"cost of transport $\mathrm{COT} = \int|\tau\dot{q}|\,"
                  r"\mathrm{d}t \,/\, (m g d)$ [-]")
    if title is None:
        title = (r"Efficiency vs.\ cadence at fixed forward speed "
                 rf"$v = {v_target:.3f}$ m/s")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.margins(y=0.08)

    # Both legends sit in a column to the right: the curves cross each other
    # freely, so an in-axes box lands on data.
    gait_legend = ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
                            fontsize=8, frameon=True, framealpha=0.9,
                            title="initial gait (lowest COT)", title_fontsize=8)
    ax.add_artist(gait_legend)

    # tight_layout ignores artists anchored outside the axes, so reserve the
    # right column explicitly.  Do it before measuring the legend below: it
    # resizes the axes, and the second legend is anchored in axes coordinates.
    fig.tight_layout(rect=(0, 0, 0.70, 1))

    marks = [
        Line2D([], [], color="0.35", linestyle="none", marker="o",
               markersize=5.5, label=r"$T$ interior to refine window"),
        Line2D([], [], color="0.35", linestyle="none", marker="o",
               markersize=5.5, markerfacecolor="none",
               label=rf"$T$ pinned at edge ($\pm{band:.2f}$ s)"),
        Line2D([], [], color="0.35", linestyle="none", marker="*",
               markersize=11, label="lowest COT of the gait"),
    ]
    # Butt the mark key directly under the gait legend rather than pinning it to
    # the bottom of the axes, which leaves a tall empty gap between the two.
    fig.canvas.draw()
    bb = gait_legend.get_window_extent(fig.canvas.get_renderer())
    y = bb.transformed(ax.transAxes.inverted()).y0 - 0.04
    ax.legend(handles=marks, loc="upper left", bbox_to_anchor=(1.02, y),
              fontsize=7.5, frameon=True, framealpha=0.9)
    return fig


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY,
                        help="codesign_summary.json from a sweep")
    parser.add_argument("--speed", type=float, default=DEFAULT_SPEED,
                        help="forward speed to slice at; snapped to the nearest "
                             "swept v_target")
    parser.add_argument("--gaits", nargs="+", default=None,
                        help="keep only these gaits, in this plot order")
    parser.add_argument("--free-t-band", type=float, default=None,
                        help="free-T half-window; default: read off the data")
    parser.add_argument("--title", default=None)
    parser.add_argument("--save", type=Path, default=None,
                        help="write the figure here (format from the extension)")
    args = parser.parse_args()

    if not args.summary.exists():
        raise SystemExit(f"no sweep summary at {args.summary}")
    v, by_gait, n_dropped = load_slice(args.summary, args.speed, args.gaits)
    band = args.free_t_band if args.free_t_band is not None else detect_band(args.summary)

    if abs(v - args.speed) > 1e-9:
        print(f"Requested v = {args.speed:.4f} m/s -> nearest swept target "
              f"{v:.4f} m/s")
    print(f"Slice at v_target = {v:.4f} m/s, free-T window +/- {band:.3f} s:")
    for gait, rows in by_gait.items():
        best = min(rows, key=lambda r: r["cot"])
        pinned = sum(abs(r["T"] - r["t_center"]) >= band - BAND_TOL for r in rows)
        print(f"  {gait:<11s} {len(rows):>2d} pts ({pinned} pinned)  "
              f"T in [{rows[0]['T']:.3f}, {rows[-1]['T']:.3f}] s  "
              f"min COT {best['cot']:.3f} at T = {best['T']:.3f} s")
    if n_dropped:
        print(f"  ({n_dropped} infeasible point(s) at this speed dropped)")

    fig = plot_cot_vs_period(by_gait, v, band, args.title)
    if args.save:
        # pad_inches above the default: the tight bbox under-measures usetex
        # text, which shaves the last glyph off the widest legend entry.
        fig.savefig(args.save, dpi=300, bbox_inches="tight", pad_inches=0.15)
        print(f"Saved → {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
