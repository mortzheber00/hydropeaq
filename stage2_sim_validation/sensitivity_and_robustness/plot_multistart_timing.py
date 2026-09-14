#!/usr/bin/env python3
"""
Gait diagram for the multi-start solutions: when does each leg push?

The distance matrix says no two starts converged to the same solution.  This
says what that means for coordination — whether the optimiser kept the
inter-leg phasing it was seeded with (lateral sequence for LSPG, trot-like for
TLPG50) or reorganised the gait entirely.

Power stroke is ``v_foot,x < 0`` in the **base frame** — the foot sweeping
backwards relative to the hull.  This is the kinematic definition of the stroke,
and it matches ``plot_solution_legs.py`` and ``gait_diagnostics.py``.

The alternative is the world-frame velocity, which asks instead whether the foot
is pushing water backwards; because the hull is itself moving forward at
0.15 m/s the foot has to beat that first, so that window is shorter -- 31-37% of
the cycle against 37-49% here.  Both are defensible and they answer different
questions; this figure is about coordination, so it uses the kinematic one.

Usage:
  python plot_multistart_timing.py --sweep 20260828_105014
  python plot_multistart_timing.py --sweep TAG --save ../docs/figures/gait_diagram.pdf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "stage1_gait_optimization"))
sys.path.insert(0, str(REPO_ROOT / "stage3_visualization"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hydro_model import load_robot                                # noqa: E402
from plot_multistart import fetch                                 # noqa: E402
from thesis_style import style_for                                # noqa: E402


def power_mask(robot, X, leg: str) -> np.ndarray:
    """``v_foot,x < 0`` in the base frame, one entry per cycle sample.

    The final column of ``X`` repeats the first (closed cycle), so it is dropped
    — keeping it would double-count one instant and put a false seam in the run
    that wraps phase 0.
    """
    nq, nv = robot.nq_reduced, robot.nv_reduced
    fid = robot.foot_frame_ids[leg]
    K = X.shape[1] - 1
    vx = np.empty(K)
    for t in range(K):
        q, v = X[:nq, t], X[nq:nq + nv, t]
        robot.forward_kinematics(q)
        J = pin.computeFrameJacobian(robot.model, robot.data, q, fid,
                                     pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        # Hull-relative: zero the base twist, then rotate into the base frame.
        v_rel = np.asarray(v, dtype=float).copy()
        v_rel[:6] = 0.0
        vx[t] = (np.array(robot.data.oMi[1].rotation).T @ (J[:3, :] @ v_rel))[0]
    return vx < 0


def runs(mask: np.ndarray):
    """Contiguous True intervals as ``(start, width)`` in cycle fractions.

    Circular: a stroke straddling phase 0 is emitted as two pieces so it draws
    at both ends of the axis rather than as one bar spanning the whole cycle.
    """
    K = len(mask)
    out, i = [], 0
    while i < K:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < K and mask[j]:
            j += 1
        out.append((i / K, (j - i) / K))
        i = j
    # Stitch a run that wraps: last sample and first sample both in power.
    if len(out) > 1 and mask[0] and mask[-1]:
        s0, w0 = out[0]
        s1, w1 = out[-1]
        out = out[1:-1] + [(s1, w1 + w0)]     # the tail piece runs past 1.0
    return out


def onset(mask: np.ndarray) -> float:
    """Cycle fraction where the power stroke begins (first False->True)."""
    K = len(mask)
    for i in range(K):
        if mask[i] and not mask[i - 1]:
            return i / K
    return 0.0


def build_figure(gaits, legs, masks):
    fig, axes = plt.subplots(len(gaits), 1, figsize=(6.4, 5.0), sharex=True)
    for ax, gait in zip(np.atleast_1d(axes), gaits):
        colour, _ = style_for(gait)
        for row, leg in enumerate(legs):
            y = len(legs) - 1 - row
            for start, width in runs(masks[gait, leg]):
                ax.broken_barh([(start, width)], (y - 0.34, 0.68),
                               facecolor=colour, edgecolor="none")
        ax.set_yticks(range(len(legs)))
        ax.set_yticklabels([l.replace("_", " ") for l in reversed(legs)],
                           fontsize=8)
        ax.set_ylim(-0.6, len(legs) - 0.4)
        ax.set_xlim(0, 1)
        ax.tick_params(length=0)
        ax.grid(axis="x", alpha=0.3)
        ax.set_axisbelow(True)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.set_ylabel(gait, fontsize=9, rotation=0, ha="right", va="center",
                      labelpad=42)
    np.atleast_1d(axes)[-1].set_xlabel("cycle phase [-]")
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep", required=True, help="sweep_tag to plot")
    parser.add_argument("--robot", default="amph", help="registered robot name")
    parser.add_argument("--save", type=Path, default=None,
                        help="write the figure here; a .pdf also writes a "
                             "300 dpi .png beside it")
    args = parser.parse_args()

    robot = load_robot(args.robot)
    legs = list(robot.spec.leg_names)
    data = fetch(args.sweep)
    gaits = list(data)
    masks = {(g, leg): power_mask(robot, data[g][0]["X"], leg)
             for g in gaits for leg in legs}

    print(f"sweep {args.sweep}: power stroke = v_foot,x < 0 (base frame)\n")
    print(f"{'':<14s}" + "".join(f"{l.replace('_', ' '):>14s}" for l in legs))
    print("duty fraction of the cycle")
    for g in gaits:
        print(f"  {g:<12s}" + "".join(f"{masks[g, l].mean():>13.1%}" for l in legs))
    print("\nonset phase, and offset from Front_Left — the coordination pattern")
    for g in gaits:
        o = {l: onset(masks[g, l]) for l in legs}
        ref = o[legs[0]]
        print(f"  {g:<12s}" + "".join(
            f"{(o[l] - ref) % 1.0:>13.2f}" for l in legs))

    fig = build_figure(gaits, legs, masks)
    if args.save:
        # pad_inches above the default: the tight bbox under-measures usetex.
        fig.savefig(args.save, dpi=300, bbox_inches="tight", pad_inches=0.15)
        print(f"\nSaved → {args.save}")
        if args.save.suffix == ".pdf":
            png = args.save.with_suffix(".png")
            fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.15)
            print(f"Saved → {png}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
