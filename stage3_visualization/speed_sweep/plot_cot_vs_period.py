#!/usr/bin/env python3
"""Cost of transport vs. solved cycle period at one speed of a co-design sweep.

One curve per gait; COT = ∫|τ·q̇| dt / (m g d) as in codesign/solver.py. Open
markers: T pinned at the edge of its free-T window (not a true optimum).
Stars: lowest COT per gait. Rings: Pareto-front points. Lines only guide the
eye; each point is an independent solve.

Usage:
  python stage3_visualization/speed_sweep/plot_cot_vs_period.py
  python stage3_visualization/speed_sweep/plot_cot_vs_period.py --speed 0.15 --save cot_vs_period.pdf
  python stage3_visualization/speed_sweep/plot_cot_vs_period.py --gaits LSPG25 LSPG33 --speed 0.25
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
DEFAULT_SPEED = 0.2      # [m/s]

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
    """COT vs T curves for one speed slice."""
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 3.0))

    for gait, rows in by_gait.items():
        colour, marker = style_for(gait)
        T = np.array([r["T"] for r in rows])
        cot = np.array([r["cot"] for r in rows])
        pinned = np.array([sweep_io.is_pinned(r, band) for r in rows])
        front = np.array([bool(r.get("pareto")) for r in rows])

        best = int(np.argmin(cot))
        ax.plot(T, cot, color=colour, lw=1.2, alpha=0.85, zorder=2,
                label=tex(gait))
        # Ring around Pareto-front points
        ax.plot(T[front], cot[front], color="0.25", linestyle="none", marker="o",
                markersize=10, markerfacecolor="none", markeredgewidth=0.8,
                zorder=2.5)
        ax.plot(T[~pinned], cot[~pinned], color=colour, linestyle="none",
                marker=marker, markersize=5.5, zorder=3)
        ax.plot(T[pinned], cot[pinned], color=colour, linestyle="none",
                marker=marker, markersize=5.5, markerfacecolor="none",
                markeredgewidth=0.9, zorder=3)
        # Lowest COT of this gait
        ax.plot(T[best], cot[best], color=colour, linestyle="none", marker="*",
                markersize=13, markeredgecolor="k", markeredgewidth=0.4, zorder=4)

    ax.set_xlabel(r"cycle period $T$ [s]")
    ax.set_ylabel(r"cost of transport $\mathrm{COT} = \int|\tau\dot{q}|\,"
                  r"\mathrm{d}t \,/\, (m g d)$ [-]")
    # No title (LaTeX caption); annotate the slice speed instead.
    ax.annotate(rf"$v = {v_target:.3f}$ m\,s$^{{-1}}$", xy=(0.99, 0.97),
                xycoords="axes fraction", ha="right", va="top", fontsize=8,
                color="0.35")
    ax.grid(alpha=0.3)
    ax.margins(y=0.08)

    # Combined legend above the axes
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
        # Extra padding: the tight bbox under-measures usetex text.
        fig.savefig(args.save, dpi=300, bbox_inches="tight", pad_inches=0.15)
        print(f"Saved → {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
