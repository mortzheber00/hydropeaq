#!/usr/bin/env python3
"""Leg stick figures of one or more OCP solutions, plus a gait-timing diagram.

Each panel shows a leg's sagittal skeleton at evenly spaced instants (coloured
by cycle phase) and its foot path, in the base frame, with shared axis limits.
Power stroke (``v_foot,x < 0`` in the base frame) is drawn bold, recovery faded.
Closed-chain legs are drawn as their separate sub-chains. The timing figure is
saved as ``<name>_timing.<ext>``.

Usage:
  python stage3_visualization/gait/plot_solution_legs.py
  python stage3_visualization/gait/plot_solution_legs.py --solutions a.npz b.npz --labels baseline symmetric
  python stage3_visualization/gait/plot_solution_legs.py --legs Front_Left Hind_Left --frames 16 --save legs.pdf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "stage1_gait_optimization"))
from stage3_visualization.common.thesis_style import PALETTE, tex  # noqa: E402  also activates the shared plot style
from stage1_gait_optimization.hydro_model import load_robot
from stage1_gait_optimization.hydro_model.trajectory import (
    expand_to_tree,
    load_solution,
)

DEFAULT_SOLUTION = Path(__file__).resolve().parents[2] / "task3_solution.npz"

# Not cyclic, but cyclic maps pass through white and would hide frames.
CYCLE_CMAP = "viridis"

# Same colours as the initial-gait timing diagram
POWER_COLOR, RECOVERY_COLOR = PALETTE[0], "#D9EAF3"


def leg_traces(robot, X: np.ndarray, nq: int, leg: str):
    """Base-frame ``(x, z)`` leg skeleton per grid node.

    Returns ``(segs, foot_xz, vx, hips)``: segments ``(K, S, 2, 2)``, foot path
    ``(K, 2)``, foot x-velocity ``(K,)`` (negative in the power stroke) and
    base-fixed mounting points ``(H, 2)``.
    """
    X_tree = expand_to_tree(robot, X, nq)
    nq_tree = robot.nq
    K = X.shape[1]

    segs = None
    for k in range(K):
        robot.forward_kinematics(X_tree[:nq_tree, k])
        # Base rotation (a view into robot.data, so read it per step)
        R_b = np.array(robot.data.oMi[1].rotation)
        pts = np.array(robot.leg_skeleton(leg))          # (S, 2, 3), world frame
        if segs is None:
            segs = np.zeros((K,) + pts.shape[:2] + (2,))
            # Skeleton endpoint closest to the foot
            foot_ij = np.unravel_index(
                np.linalg.norm(pts - robot.foot_positions()[leg], axis=-1).argmin(),
                pts.shape[:2])
        # Relative to the leg mount, rotated into the base frame
        segs[k] = ((pts - pts[0, 0]) @ R_b)[:, :, [0, 2]]

    foot_xz = segs[:, foot_ij[0], foot_ij[1], :]
    # Periodic central difference (the last column repeats the first)
    x = foot_xz[:, 0]
    vx = np.gradient(np.concatenate([x[:-1], x[:-1], x[:-1]]))[K - 1: 2 * (K - 1) + 1]

    # Hips: segment starts that are fixed in the base frame and not the end of
    # another segment (amph's fixed thigh joint is excluded by the latter).
    starts, ends = segs[:, :, 0, :], segs[0, :, 1, :]
    fixed = np.ptp(starts, axis=0).max(axis=1) < 1e-9
    free = (np.linalg.norm(starts[0][:, None] - ends[None], axis=-1) > 1e-9).all(axis=1)
    hips = np.unique(starts[0][fixed & free], axis=0)
    return segs, foot_xz, vx, hips


def load_all(paths, labels):
    """Load every solution, checking they share a robot."""
    out = []
    for i, p in enumerate(paths):
        meta = load_solution(str(p))
        out.append({
            "path": Path(p),
            "label": labels[i] if labels else Path(p).stem,
            "T": meta["T"],
            "X": meta["X"],
            "nq": meta["nq"],
            "N": meta["N"],
            "robot": meta["robot"],
        })
    robots = {s["robot"] for s in out}
    if len(robots) > 1:
        raise SystemExit(
            f"solutions are for different robots ({', '.join(sorted(robots))}); "
            f"leg geometries are not comparable"
        )
    return out


def all_traces(robot, sols, legs):
    """``leg_traces`` for every (solution, leg), keyed by ``(label, leg)``."""
    traces = {}
    for s in sols:
        for leg in legs:
            try:
                traces[s["label"], leg] = leg_traces(robot, s["X"], s["nq"], leg)
            except Exception as e:
                raise SystemExit(
                    f"{s['robot']}: cannot build a sagittal skeleton for "
                    f"{leg!r} ({e})"
                )
    return traces


def power_segments(vx: np.ndarray):
    """``(start, width)`` spans of the cycle with ``vx < 0``.

    ``vx`` has one value per grid node (last repeats first). Zero crossings are
    interpolated linearly so phase-shifted legs line up exactly. Wrapping spans
    are split in two for ``broken_barh``.
    """
    v = vx[:-1]
    n = len(v)
    if (v < 0).all():
        return [(0.0, 1.0)]
    if (v >= 0).all():
        return []

    def turn(k):
        """Where vx crosses zero between point ``k`` and the next."""
        a, b = v[k], v[(k + 1) % n]
        return (k + a / (a - b)) / n

    # Pair each start with the next end; a missing end wraps past 1.
    starts = [turn(k) for k in range(n) if v[k] >= 0 > v[(k + 1) % n]]
    ends = [turn(k) for k in range(n) if v[k] < 0 <= v[(k + 1) % n]]
    segs = []
    for a in starts:
        later = [e for e in ends if e > a]
        b = min(later) if later else min(ends) + 1.0
        segs += [(a, 1.0 - a), (0.0, b - 1.0)] if b > 1.0 else [(a, b - a)]
    return segs


def plot_gait_timing(sols, legs, traces):
    """Power/recovery bars per leg over one cycle, one panel per solution."""
    n = len(legs)
    fig, axes = plt.subplots(len(sols), 1, squeeze=False, sharex=True,
                             figsize=(7.0, (0.42 * n + 0.5) * len(sols) + 1.0))
    for ax, s in zip(axes[:, 0], sols):
        for i, leg in enumerate(legs):
            y = n - 1 - i                     # first leg drawn on top
            ax.broken_barh([(0.0, 1.0)], (y - 0.4, 0.8),
                           facecolors=RECOVERY_COLOR, edgecolors="0.6", lw=0.8)
            ax.broken_barh(power_segments(traces[s["label"], leg][2]),
                           (y - 0.4, 0.8),
                           facecolors=POWER_COLOR, edgecolors="0.3", lw=0.8)
        ax.set_yticks(range(n))
        ax.set_yticklabels([tex(l.replace("_", " ")) for l in legs[::-1]])
        ax.set_ylim(-0.6, n - 0.4)
        ax.set_xlim(0.0, 1.0)
        ax.set_xticks(np.linspace(0.0, 1.0, 5))
        if len(sols) > 1:
            ax.set_title(rf"{tex(s['label'])} ($T={s['T']:.3f}$ s)")
    axes[-1, 0].set_xlabel("cycle fraction")

    handles = [
        Patch(facecolor=POWER_COLOR, edgecolor="0.3",
              label=r"power stroke ($v_{\mathrm{foot},x} < 0$)"),
        Patch(facecolor=RECOVERY_COLOR, edgecolor="0.6", label="recovery stroke"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8,
               frameon=True, framealpha=0.9)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    return fig


def plot_solution_legs(sols, legs, n_frames: int, traces):
    """Stick-figure grid: legs as a grid for one solution, legs x solutions otherwise."""
    stacked = np.vstack([segs.reshape(-1, 2) for segs, *_ in traces.values()])
    (x0, z0), (x1, z1) = stacked.min(0), stacked.max(0)
    mx, mz = 0.08 * (x1 - x0), 0.08 * (z1 - z0)

    single = len(sols) == 1
    if single:
        n_cols = min(2, len(legs))
        n_rows = -(-len(legs) // n_cols)
        cells = [(i // n_cols, i % n_cols, leg, sols[0])
                 for i, leg in enumerate(legs)]
    else:
        n_rows, n_cols = len(legs), len(sols)
        cells = [(r, c, leg, s)
                 for r, leg in enumerate(legs) for c, s in enumerate(sols)]

    # Cell size follows the data aspect (equal axes), avoiding dead space
    cell_w = 3.7
    cell_h = cell_w * (z1 - z0 + 2 * mz) / (x1 - x0 + 2 * mx) + 0.55
    fig, axes = plt.subplots(n_rows, n_cols, squeeze=False,
                             figsize=(cell_w * n_cols + 1.0, cell_h * n_rows + 0.9))
    norm = Normalize(0.0, 1.0)
    cmap = plt.get_cmap(CYCLE_CMAP)

    for ax in axes.flat:          # hide unused cells
        ax.set_visible(False)
    for row, col, leg, s in cells:
        ax = axes[row, col]
        ax.set_visible(True)
        segs, foot_xz, vx, hips = traces[s["label"], leg]
        power = vx < 0
        K = segs.shape[0]

        ax.plot(foot_xz[:, 0], foot_xz[:, 1], "-", color="0.45", lw=1.2, zorder=1)

        # Evenly spaced frames, excluding the duplicated last node
        idx = np.unique(np.linspace(0, K - 2, n_frames).round().astype(int))
        for k in idx:
            colour = cmap(norm(k / (K - 1)))
            lw, alpha = (2.4, 1.0) if power[k] else (1.1, 0.55)
            zo = 3 if power[k] else 2
            # Per segment, since closed-chain legs are not one polyline
            for a, b in segs[k]:
                ax.plot([a[0], b[0]], [a[1], b[1]], "-", color=colour, lw=lw,
                        alpha=alpha, marker="o", ms=2.5, zorder=zo)
            ax.plot(foot_xz[k, 0], foot_xz[k, 1], "o", color=colour,
                    ms=6 if power[k] else 4, alpha=alpha, zorder=zo)

        ax.plot(hips[:, 0], hips[:, 1], "ks", ms=6, zorder=4)
        ax.set_xlim(x0 - mx, x1 + mx)
        ax.set_ylim(z0 - mz, z1 + mz)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)

        leg_name = tex(leg.replace("_", " "))
        if single:
            ax.set_title(leg_name)
        else:
            if row == 0:
                ax.set_title(rf"{tex(s['label'])} ($T={s['T']:.3f}$ s)")
            if col == 0:
                ax.set_ylabel(f"{leg_name}\n" r"$z_{\mathrm{body}}$ [m]")
        if row == n_rows - 1:
            ax.set_xlabel(r"$x_{\mathrm{body}}$ [m]")
        if single and col == 0:
            ax.set_ylabel(r"$z_{\mathrm{body}}$ [m]")

    handles = [
        Line2D([], [], color="0.35", lw=2.4, marker="o", ms=2.5,
               label=r"power stroke ($v_{\mathrm{foot},x} < 0$)"),
        Line2D([], [], color="0.35", lw=1.1, alpha=0.55, marker="o", ms=2.5,
               label="recovery stroke"),
        Line2D([], [], color="0.45", lw=1.2, label="foot path over the cycle"),
        Line2D([], [], color="k", linestyle="none", marker="s", ms=6,
               label="hip"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8,
               frameon=True, framealpha=0.9, bbox_to_anchor=(0.45, 0.0))

    # Leave room for the legend (bottom) and colourbar (right)
    fig.tight_layout(rect=(0, 0.05, 0.90, 1.0))

    cax = fig.add_axes((0.92, 0.18, 0.015, 0.66))
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label("cycle fraction")
    cb.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    return fig


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--solutions", type=Path, nargs="+",
                        default=[DEFAULT_SOLUTION],
                        help="one or more solution .npz files, plotted as columns")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="column titles; default: the file stems")
    parser.add_argument("--legs", nargs="+", default=None,
                        help="legs to draw as rows; default: all of them")
    parser.add_argument("--frames", type=int, default=12,
                        help="stick figures drawn per cycle")
    parser.add_argument("--save", type=Path, default=None,
                        help="write the figure here (format from the extension); "
                             "the timing figure goes to *_timing.<ext>")
    args = parser.parse_args()

    for p in args.solutions:
        if not p.exists():
            raise SystemExit(f"no solution at {p}")
    if args.labels and len(args.labels) != len(args.solutions):
        raise SystemExit(f"got {len(args.labels)} labels for "
                         f"{len(args.solutions)} solutions")

    sols = load_all(args.solutions, args.labels)
    robot = load_robot(sols[0]["robot"])
    legs = args.legs or list(robot.spec.leg_names)
    unknown = [l for l in legs if l not in robot.spec.leg_names]
    if unknown:
        raise SystemExit(f"unknown leg(s) {unknown}; have "
                         f"{list(robot.spec.leg_names)}")

    print(f"Robot: {sols[0]['robot']}   legs: {', '.join(legs)}")
    for s in sols:
        print(f"  {s['label']:<28s} T = {s['T']:.4f} s, N = {s['N']}, "
              f"{s['path']}")

    traces = all_traces(robot, sols, legs)

    fig = plot_solution_legs(sols, legs, args.frames, traces)
    fig_timing = plot_gait_timing(sols, legs, traces)
    if args.save:
        # Extra padding: the tight bbox under-measures usetex text.
        fig.savefig(args.save, dpi=300, bbox_inches="tight", pad_inches=0.15)
        print(f"Saved → {args.save}")
        timing_path = args.save.with_name(f"{args.save.stem}_timing{args.save.suffix}")
        fig_timing.savefig(timing_path, dpi=300, bbox_inches="tight", pad_inches=0.15)
        print(f"Saved → {timing_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
