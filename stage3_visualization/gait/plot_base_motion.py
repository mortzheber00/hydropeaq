#!/usr/bin/env python3
"""Base motion of one gait cycle: stroboscopic side view and phase traces.

``*_side``   the robot at several phases over two cycles, with base and foot paths
``*_traces`` world-frame forward velocity, position deviation and orientation
             over two cycles of normalised phase

The base twist in the solution is in the body frame and is rotated into the
world frame before plotting. Traces use the grid nodes.

Usage:
  python stage3_visualization/gait/plot_base_motion.py
  python stage3_visualization/gait/plot_base_motion.py \\
      --solution stage1_gait_optimization/codesign/codesign_results/TLPG50_v0p17_T1p200.npz --save base_motion.pdf
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
from stage1_gait_optimization.hydro_model import load_robot
from stage1_gait_optimization.hydro_model.trajectory import expand_to_tree, load_solution

from stage3_visualization.common.thesis_style import (  # noqa: E402,F401  activates the shared style
    LEGEND_ROW_IN,
    PALETTE,
    TEXT_WIDTH_IN,
    full_width,
    legend_row,
)

BODY_C = PALETTE[0]
# x/y/z and roll/pitch/yaw share these colours (separate legends)
TRACE_C = PALETTE[:3]
FOOT_C = (PALETTE[1], PALETTE[2])  # front, hind
# Side view: STROBE_PHASES poses spread over STROBE_CYCLES cycles. Poses repeat
# the same phases unless PHASES / CYCLES is non-integer (e.g. 10 over 3).
STROBE_CYCLES = 2
STROBE_PHASES = 8     # poses in total, not per cycle
full_width()


def base_pose(X, nq):
    """World position, roll/pitch/yaw [deg] and world-frame linear velocity."""
    pos = X[0:3, :]
    quat = X[3:7, :]                      # (x, y, z, w)
    twist_body = X[nq : nq + 3, :]

    rpy = np.empty_like(pos)
    vel_world = np.empty_like(pos)
    for k in range(X.shape[1]):
        R = pin.Quaternion(*np.roll(quat[:, k], 1)).matrix()   # pin expects (w, x, y, z)
        rpy[:, k] = pin.rpy.matrixToRpy(R)
        vel_world[:, k] = R @ twist_body[:, k]
    return pos, np.degrees(rpy), vel_world


def one_side_front_to_hind(robot, X, nq):
    """One leg per left-right pair (one body side), sorted front to hind."""
    legs = [pair[0] for pair in robot.spec.lr_leg_pairs]
    X_tree = expand_to_tree(robot, X, nq)
    robot.forward_kinematics(X_tree[: robot.nq, 0])
    oMb = robot.data.oMi[1]
    body_x = {leg: oMb.actInv(np.asarray(robot.leg_skeleton(leg)[0][0]))[0]
              for leg in legs}
    return sorted(legs, key=lambda leg: -body_x[leg])


def leg_geometry(robot, X, nq, legs):
    """Per-node leg segments, spine and foot positions (world frame).

    The spine connects the hips of ``legs``, so body and legs are drawn on the
    same side.
    """
    X_tree = expand_to_tree(robot, X, nq)
    frames, feet = [], {leg: [] for leg in legs}

    for k in range(X.shape[1]):
        robot.forward_kinematics(X_tree[: robot.nq, k])
        skel = {leg: robot.leg_skeleton(leg) for leg in legs}
        frames.append({
            "spine": (np.asarray(skel[legs[0]][0][0]),
                      np.asarray(skel[legs[-1]][0][0])),
            "legs": skel,
        })
        for leg in legs:
            feet[leg].append(np.asarray(robot.foot_positions()[leg]))
    return frames, {leg: np.array(p).T for leg, p in feet.items()}


def _position_deviation(pos, N):
    """Base position minus the linear forward advance and the mean."""
    out = pos.copy()
    out[0] -= np.linspace(pos[0, 0], pos[0, -1], N + 1)
    return out - out.mean(axis=1, keepdims=True)


def plot_side_view(robot, X, nq):
    """Stroboscopic side view."""
    pos, _, _ = base_pose(X, nq)
    N = X.shape[1] - 1
    legs = one_side_front_to_hind(robot, X, nq)
    frames, feet = leg_geometry(robot, X, nq, legs)
    dx_cycle = pos[0, -1] - pos[0, 0]

    fig, ax_side = plt.subplots(figsize=(TEXT_WIDTH_IN, 2.5))

    # Spread the poses over several cycles; one cycle advances less than a body length.
    for i, ph in enumerate(np.linspace(0, STROBE_CYCLES, STROBE_PHASES,
                                       endpoint=False)):
        cycle, frac = divmod(ph, 1.0)
        f = frames[int(round(frac * N))]
        dx = cycle * dx_cycle
        alpha = 0.3 + 0.7 * i / (STROBE_PHASES - 1)
        ax_side.plot([f["spine"][0][0] + dx, f["spine"][1][0] + dx],
                     [f["spine"][0][2], f["spine"][1][2]],
                     color=BODY_C, lw=1.6, alpha=alpha, zorder=4,
                     solid_capstyle="round", label="body" if i == 0 else None)
        for leg in legs:
            for p, q in f["legs"][leg]:
                ax_side.plot([p[0] + dx, q[0] + dx], [p[2], q[2]],
                             color="0.25", lw=0.8, alpha=alpha, zorder=4,
                             solid_capstyle="round")

    for cycle in range(STROBE_CYCLES):
        dx = cycle * dx_cycle
        # Dashed black, to distinguish it from the grey legs
        ax_side.plot(pos[0] + dx, pos[2], color="k", lw=0.7, ls=(0, (4, 2)),
                     zorder=5, label="base path" if cycle == 0 else None)
        for leg, colour, name in zip(legs, FOOT_C, ("front foot", "hind foot")):
            ax_side.plot(feet[leg][0] + dx, feet[leg][2], color=colour, lw=0.5,
                         alpha=0.65, zorder=2,
                         label=name if cycle == 0 else None)

    ax_side.set_aspect("equal")
    ax_side.set_xlabel(r"$x$ [m]")
    ax_side.set_ylabel(r"$z$ [m]")
    ax_side.grid(alpha=0.3)
    # Legend above the axes, since the poses fill the frame
    ax_side.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4,
                   handlelength=1.6, frameon=False, columnspacing=1.4,
                   borderaxespad=0.2)
    legend_row(fig, ax_side)
    fig.tight_layout(pad=0.3)
    return fig


def plot_phase_traces(X, T, N, nq):
    """Base velocity, position deviation and orientation over two cycles."""
    pos, rpy, vel_world = base_pose(X, nq)
    v_mean = (pos[0, -1] - pos[0, 0]) / T
    dev = _position_deviation(pos, N)

    # Extra height for the two legend rows
    fig, (ax_v, ax_p, ax_a) = plt.subplots(
        3, 1, figsize=(TEXT_WIDTH_IN, 3.6 + 2 * LEGEND_ROW_IN), sharex=True)

    phase = np.linspace(0, 1, N + 1)
    two = np.concatenate([phase[:-1], 1 + phase])          # cycle drawn twice
    rep = lambda a: np.concatenate([a[:-1], a])            # noqa: E731

    for ax in (ax_v, ax_p, ax_a):
        ax.minorticks_on()
        ax.grid(which="major", alpha=0.3)
        ax.grid(which="minor", alpha=0.12, lw=0.5)
        ax.axvline(1.0, color="0.6", lw=0.5, ls="--")

    ax_v.plot(two, rep(vel_world[0]), color=TRACE_C[0], lw=1.0)
    ax_v.axhline(v_mean, color="0.35", lw=0.6, ls="--")
    ax_v.axhline(0.0, color="0.75", lw=0.5)
    ax_v.annotate(rf"$\bar v = {v_mean:.2f}$\,m\,s$^{{-1}}$", xy=(0.99, 0.97),
                  xycoords="axes fraction", fontsize=8,
                  ha="right", va="top", color="0.35")
    # Show T, since the x-axis is normalised
    ax_v.margins(y=0.18)
    ax_v.annotate(rf"$T = {T:.2f}$\,s", xy=(0.01, 0.97), xycoords="axes fraction",
                  fontsize=8, ha="left", va="top", color="0.35")
    ax_v.set_ylabel(r"$v_x^{\mathrm{world}}$" "\n" r"[m\,s$^{-1}$]")

    for row, c, lab in zip(range(3), TRACE_C, ("$x$", "$y$", "$z$")):
        ax_p.plot(two, rep(dev[row]) * 1e3, color=c, lw=1.0, label=lab)
    ax_p.set_ylabel("translation\ndeviation [mm]")
    # Separate legends: rows reuse the colours
    ax_p.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0),
                columnspacing=1.4, handlelength=1.4, frameon=False,
                borderaxespad=0.2)

    for row, c, lab in zip(range(3), TRACE_C, ("roll", "pitch", "yaw")):
        ax_a.plot(two, rep(rpy[row]), color=c, lw=1.0, label=lab)
    ax_a.set_ylabel("rotation\n[deg]")
    ax_a.set_xlabel("cycle phase")
    ax_a.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0),
                columnspacing=1.4, handlelength=1.4, frameon=False,
                borderaxespad=0.2)

    ax_a.set_xlim(0, 2)
    ax_a.set_xticks([0, 0.5, 1, 1.5, 2])
    # Set hspace after tight_layout; gridspec_kw would make tight_layout fall back
    # to default margins.
    fig.tight_layout(pad=0.3)
    fig.subplots_adjust(hspace=0.45)
    return fig


def motion_stats(X, T, N, nq):
    """Summary numbers printed to the console."""
    pos, rpy, vel_world = base_pose(X, nq)
    dev = _position_deviation(pos, N)
    return {
        "mean speed [m/s]": (pos[0, -1] - pos[0, 0]) / T,
        "base vx world min/max [m/s]": (vel_world[0].min(), vel_world[0].max()),
        "base x deviation pk-pk [mm]": np.ptp(dev[0]) * 1e3,
        "base y deviation pk-pk [mm]": np.ptp(dev[1]) * 1e3,
        "base z deviation pk-pk [mm]": np.ptp(dev[2]) * 1e3,
        "roll/pitch/yaw pk-pk [deg]": tuple(np.round(np.ptp(rpy, axis=1), 1)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--solution", type=Path,
                    default=_ROOT / "task3_solution.npz")
    ap.add_argument("--save", type=Path, default=None,
                    help="output stem; each figure gets its own suffixed file, "
                         "and the extension picks the format")
    args = ap.parse_args()

    meta = load_solution(args.solution)
    robot = load_robot(meta["robot"])
    X, T, N, nq = meta["X"], meta["T"], meta["N"], meta["nq"]
    figs = {
        "side": plot_side_view(robot, X, nq),
        "traces": plot_phase_traces(X, T, N, nq),
    }

    print(f"{args.solution.name}:")
    for k, v in motion_stats(X, T, N, nq).items():
        print(f"  {k:<30s} {v}")

    if args.save:
        for tag, fig in figs.items():
            path = args.save.with_name(f"{args.save.stem}_{tag}{args.save.suffix}")
            fig.savefig(path, dpi=300)
            print(f"Saved → {path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
