#!/usr/bin/env python3
"""
Leg-configuration stick figures of an OCP *solution*, one column per solution.

The solved analogue of ``initial_guess/visualization/leg_configurations.py``,
which draws the same stick figures for the paper keyframes that seed the OCP.
Here the configurations come from a solved trajectory, so the figure shows what
the optimiser actually made the leg do, and several solutions can be placed
side by side.

Each panel draws the leg's sagittal-plane skeleton at evenly spaced instants of
the cycle, plus the closed foot path over the whole cycle.  Everything is
expressed in the **base frame** (base translation *and* rotation removed),
because base heave and pitch over a swim cycle are comparable to the stroke
itself — a world-frame plot would mostly show the body moving.

The skeleton is the robot's cylinder chain, so a serial leg (amph) comes out as
the connected run ``[hip, thigh, calf, foot]`` while a closed-chain leg (body2)
comes out as its two sub-chains — the loop pins that join them are not in the
tree model, so they are not drawn.  Every point that stays put in the base
frame over the cycle is marked as a hip.

Encoding:
  - colour   = cycle fraction, on a cyclic colormap (fraction 0 and 1 are the
               same instant, so a linear ramp would put a false seam there)
  - emphasis = power stroke (foot sweeping backwards in the base frame,
               ``v_foot,x < 0``) drawn solid and thick; recovery drawn thin and
               faded.  Same power/recovery split ``gait_diagnostics.py`` uses.

All panels share one set of axis limits and an equal aspect ratio, so leg
lengths and stroke envelopes are directly comparable across legs and solutions.

A second figure shows the gait timing — the power stroke marked over the whole
cycle, one bar per leg — as ``leg_configurations.py`` does for the initial
gaits.  There the power phase is a prescribed fraction at a prescribed offset;
here it is read back off the solved foot velocity, so the bars show what the
optimiser chose.  ``--save`` writes it next to the main figure as
``*_timing.<ext>``.

Usage:
  python plot_solution_legs.py
  python plot_solution_legs.py --solutions a.npz b.npz --labels "baseline" "symmetric"
  python plot_solution_legs.py --legs Front_Left Hind_Left --frames 16 --save legs.pdf
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
from thesis_style import PALETTE, tex  # also activates the shared plot style

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stage1_gait_optimization"))
from stage1_gait_optimization.hydro_model import load_robot
from stage1_gait_optimization.hydro_model.trajectory import (
    expand_to_tree,
    load_solution,
)

DEFAULT_SOLUTION = Path(__file__).resolve().parents[1] / "task3_solution.npz"

# Perceptually uniform and CVD-safe, and — unlike the cyclic maps — never passes
# through white, so no frame disappears against the page.  The cost is a colour
# seam between the last and first frame of the cycle, which the closed foot path
# already makes obvious.
CYCLE_CMAP = "viridis"

# Same power/recovery blues the initial-gait timing diagram uses: PALETTE[0],
# and PALETTE[0] at 15 % on white.
POWER_COLOR, RECOVERY_COLOR = PALETTE[0], "#D9EAF3"


def leg_traces(robot, X: np.ndarray, nq: int, leg: str):
    """Base-frame ``(x, z)`` leg skeleton per grid point, and the foot path.

    Returns ``(segs, foot_xz, vx, hips)`` where ``segs`` is ``(K, S, 2, 2)``
    over the leg's ``S`` skeleton segments and their two endpoints, ``foot_xz``
    is ``(K, 2)``, ``vx`` is the ``(K,)`` foot velocity along x — negative on
    the power stroke — and ``hips`` is ``(H, 2)``.  The velocity rather than
    the ``vx < 0`` flag, so ``power_segments`` can interpolate where the stroke
    actually turns instead of rounding it to a grid point.

    ``leg_skeleton`` rather than ``leg_centerline_positions``: the latter reads
    a leg's Side/Thigh/Calf joints by name and so exists only for a serial leg,
    while the cylinder chain is defined for every robot.  For a serial leg the
    two agree — the chain *is* the centreline, minus the zero-length foot cap.
    """
    X_tree = expand_to_tree(robot, X, nq)
    nq_tree = robot.nq
    K = X.shape[1]

    segs = None
    for k in range(K):
        robot.forward_kinematics(X_tree[:nq_tree, k])
        # oMi[1] is the free-flyer placement and is a view into robot.data, so
        # it must be read inside the loop.
        R_b = np.array(robot.data.oMi[1].rotation)
        pts = np.array(robot.leg_skeleton(leg))          # (S, 2, 3), world frame
        if segs is None:
            segs = np.zeros((K,) + pts.shape[:2] + (2,))
            # Which endpoint is the foot, rather than assuming the chain ends
            # on it.  Taken off the skeleton so it carries the same sagittal
            # projection as the drawn segments.
            foot_ij = np.unravel_index(
                np.linalg.norm(pts - robot.foot_positions()[leg], axis=-1).argmin(),
                pts.shape[:2])
        # Rotate into the base frame after removing the leg's mounting point
        # (the first link's start): subtracting world positions alone would
        # leave the base's own pitch in the drawing.
        segs[k] = ((pts - pts[0, 0]) @ R_b)[:, :, [0, 2]]

    foot_xz = segs[:, foot_ij[0], foot_ij[1], :]
    # Power stroke = foot travelling backwards relative to the body.  Central
    # difference on the closed cycle; X's last column repeats the first.
    x = foot_xz[:, 0]
    vx = np.gradient(np.concatenate([x[:-1], x[:-1], x[:-1]]))[K - 1: 2 * (K - 1) + 1]

    # A link start that neither moves in the base frame nor meets another drawn
    # link is mounted on the base: amph's one hip, body2's two hip servos.
    # Marking them keeps a sub-chain that starts away from the origin from
    # appearing to float.  Both halves of the test are needed — amph's thigh
    # joint is fixed too (the OCP pins the side joints to zero) but continues
    # the side link, while body2's parallelogram pin continues nothing drawn
    # but moves.
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
    """``leg_traces`` for every (solution, leg) pair, keyed by ``(label, leg)``.

    Computed up front so the shared axis limits can be set before drawing, and
    so the timing figure reuses the same power/recovery split.
    """
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
    """``(start, width)`` cycle-fraction spans where the foot sweeps backwards.

    ``vx`` carries one value per grid point; its last entry repeats the first
    and is dropped.  Each sign change is placed by linear interpolation between
    the two points bracketing it rather than snapped to a point, for two
    reasons: the bar edges land where the stroke actually turns, to within the
    interpolation, instead of up to a grid step away; and a left-right pair
    constrained to one stroke at a free phase offset (``add_symmetry_constraints``)
    comes out as an exact shift.  Rounding to grid points does not — the offset
    is continuous, so the two legs sample the same stroke at different points
    and disagree by a step wherever a turn falls between their samples.

    A span that wraps past fraction 1 comes back as two, which is what
    ``broken_barh`` wants anyway.
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

    # Equal counts, alternating around the cycle, so each start pairs with the
    # next end; the one start with no end after it closes past fraction 1.
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
    stacked = np.vstack([segs.reshape(-1, 2) for segs, *_ in traces.values()])
    (x0, z0), (x1, z1) = stacked.min(0), stacked.max(0)
    mx, mz = 0.08 * (x1 - x0), 0.08 * (z1 - z0)

    # One solution: lay the legs out as a grid, like the initial-guess figure
    # this mirrors — a single column of four would be a metre-tall strip.
    # Several solutions: legs down the rows, solutions across the columns, so
    # any row compares the same leg between designs.
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

    # Panels are aspect-equal on shared limits, so their drawn shape is fixed by
    # the data.  Size the cells to match it, or every row gets padded with a
    # band of dead space.
    cell_w = 3.7
    cell_h = cell_w * (z1 - z0 + 2 * mz) / (x1 - x0 + 2 * mx) + 0.55
    fig, axes = plt.subplots(n_rows, n_cols, squeeze=False,
                             figsize=(cell_w * n_cols + 1.0, cell_h * n_rows + 0.9))
    norm = Normalize(0.0, 1.0)
    cmap = plt.get_cmap(CYCLE_CMAP)

    for ax in axes.flat:          # blank any unused cell in the leg grid
        ax.set_visible(False)
    for row, col, leg, s in cells:
        ax = axes[row, col]
        ax.set_visible(True)
        segs, foot_xz, vx, hips = traces[s["label"], leg]
        power = vx < 0
        K = segs.shape[0]

        # Closed foot path for the whole cycle, under the stick figures.
        ax.plot(foot_xz[:, 0], foot_xz[:, 1], "-", color="0.45", lw=1.2, zorder=1)

        # Evenly spaced instants; drop the duplicated final grid point so the
        # first and last drawn frame are not the same configuration.
        idx = np.unique(np.linspace(0, K - 2, n_frames).round().astype(int))
        for k in idx:
            colour = cmap(norm(k / (K - 1)))
            lw, alpha = (2.4, 1.0) if power[k] else (1.1, 0.55)
            zo = 3 if power[k] else 2
            # Segment by segment rather than as one polyline: a closed-chain
            # leg's links do not form a single connected run.
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
    # Figure-level so it never lands on a panel's stick figures.
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8,
               frameon=True, framealpha=0.9, bbox_to_anchor=(0.45, 0.0))

    # Reserve the right column for the colourbar and a bottom strip for the
    # legend; tight_layout accounts for neither.
    fig.tight_layout(rect=(0, 0.05, 0.90, 1.0))

    # One shared colourbar: the phase encoding is identical in every panel.
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
        # pad above the default: the tight bbox under-measures usetex text.
        fig.savefig(args.save, dpi=300, bbox_inches="tight", pad_inches=0.15)
        print(f"Saved → {args.save}")
        timing_path = args.save.with_name(f"{args.save.stem}_timing{args.save.suffix}")
        fig_timing.savefig(timing_path, dpi=300, bbox_inches="tight", pad_inches=0.15)
        print(f"Saved → {timing_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
