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
  python plot_cot_vs_period.py --speed 0.15 --save ../../docs/figures/cot_vs_period.pdf
  python plot_cot_vs_period.py --gaits LSPG25 LSPG33 --speed 0.25
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from stage3_visualization.common.thesis_style import (  # also activates the shared plot style
    TEXT_WIDTH_IN,
    full_width,
    legend_row,
    style_for,
    tex,
)

from stage3_visualization.common import sweep_io

DEFAULT_SUMMARY = (Path(__file__).resolve().parents[2] / "stage1_gait_optimization" /
                   "codesign" / "codesign_results" / "codesign_summary.json")
DEFAULT_SPEED = 0.2      # m/s — the pipeline's nominal design speed (V_TARGET)

full_width()


def load_slice(rows: list, speed: float, gaits=None):
    """Feasible rows at the ``v_target`` nearest ``speed``, grouped by gait.

    Returns ``(v_target, {gait: rows sorted by solved T}, n_dropped)``.
    """
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


def plot_cot_vs_period(by_gait: dict, v_target: float, band: float):
    # The text-block canvas, so the figure goes in unscaled and its type matches
    # the rest of the thesis.  It used to be 8.8 in wide against a 5.98 in block,
    # which LaTeX shrank to 68% and took the tick labels with it.
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 3.0))

    for gait, rows in by_gait.items():
        colour, marker = style_for(gait)
        T = np.array([r["T"] for r in rows])
        cot = np.array([r["cot"] for r in rows])
        # Pinned at a window edge -> the cadence is the constraint's, not the
        # solver's choice.  Drawn open; interior optima drawn filled.
        pinned = np.array([sweep_io.is_pinned(r, band) for r in rows])
        # On the sweep's own Pareto front: this point is the one the front
        # figure carries at this speed, so a reader can find it on the curve
        # rather than inferring it from the minimum.
        front = np.array([bool(r.get("pareto")) for r in rows])

        best = int(np.argmin(cot))
        # Name only.  The per-gait minimum used to ride along in the label, but
        # at text width that column is 1.8 in and the numbers no longer fit;
        # they are printed to stdout, which is where a caption takes them from.
        ax.plot(T, cot, color=colour, lw=1.2, alpha=0.85, zorder=2,
                label=tex(gait))
        # Ring under the marker, wide enough to stay visible around it.
        ax.plot(T[front], cot[front], color="0.25", linestyle="none", marker="o",
                markersize=10, markerfacecolor="none", markeredgewidth=0.8,
                zorder=2.5)
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
    # No title: this goes into the thesis via \includegraphics and the LaTeX
    # caption describes it, as it does for every other figure here.  The speed
    # the slice is taken at is the one thing the axes do not carry, so it is
    # annotated inside them instead.
    ax.annotate(rf"$v = {v_target:.3f}$ m\,s$^{{-1}}$", xy=(0.99, 0.97),
                xycoords="axes fraction", ha="right", va="top", fontsize=8,
                color="0.35")
    ax.grid(alpha=0.3)
    ax.margins(y=0.08)

    # Both legends go above the axes, in one block: at text width there is no
    # room for a right-hand column, and the curves cross freely so an in-axes
    # box lands on data.
    marks = [
        Line2D([], [], color="0.35", linestyle="none", marker="o",
               markersize=5.5, label=r"$T$ interior to refine window"),
        Line2D([], [], color="0.35", linestyle="none", marker="o",
               markersize=5.5, markerfacecolor="none",
               label=rf"$T$ pinned at edge ($\pm{band:.2f}$ s)"),
        Line2D([], [], color="0.35", linestyle="none", marker="*",
               markersize=11, label="lowest COT of the gait"),
        Line2D([], [], color="0.25", linestyle="none", marker="o",
               markersize=9, markerfacecolor="none", markeredgewidth=0.8,
               label="on the sweep's Pareto front"),
    ]
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles + marks, loc="lower center",
              bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=7,
              columnspacing=1.2, handlelength=1.6, frameon=False,
              borderaxespad=0.2)

    # An outside legend is invisible to tight_layout, so buy its rows here.
    legend_row(fig, ax, rows=2)
    fig.tight_layout(pad=0.3)
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
    parser.add_argument("--save", type=Path, default=None,
                        help="write the figure here (format from the extension)")
    args = parser.parse_args()

    if not args.summary.exists():
        raise SystemExit(f"no sweep summary at {args.summary}")
    rows = json.loads(args.summary.read_text())
    v, by_gait, n_dropped = load_slice(rows, args.speed, args.gaits)
    band = (args.free_t_band if args.free_t_band is not None
            else sweep_io.detect_band(rows))

    if abs(v - args.speed) > 1e-9:
        print(f"Requested v = {args.speed:.4f} m/s -> nearest swept target "
              f"{v:.4f} m/s")
    print(f"Slice at v_target = {v:.4f} m/s, free-T window +/- {band:.3f} s:")
    for gait, rows in by_gait.items():
        best = min(rows, key=lambda r: r["cot"])
        pinned = sum(sweep_io.is_pinned(r, band) for r in rows)
        front = sum(bool(r.get("pareto")) for r in rows)
        print(f"  {gait:<11s} {len(rows):>2d} pts ({pinned} pinned, "
              f"{front} on front)  "
              f"T in [{rows[0]['T']:.3f}, {rows[-1]['T']:.3f}] s  "
              f"min COT {best['cot']:.3f} at T = {best['T']:.3f} s")
    if n_dropped:
        print(f"  ({n_dropped} infeasible point(s) at this speed dropped)")

    fig = plot_cot_vs_period(by_gait, v, band)
    if args.save:
        # pad_inches above the default: the tight bbox under-measures usetex
        # text, which shaves the last glyph off the widest legend entry.
        fig.savefig(args.save, dpi=300, bbox_inches="tight", pad_inches=0.15)
        print(f"Saved → {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
