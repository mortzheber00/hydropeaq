#!/usr/bin/env python3
"""Base-motion figures for one optimised gait cycle.

Two standalone figures of the same motion, each on the full text-width canvas so
they go into the thesis unscaled and their type matches:

  ``*_side``    Stroboscopic side view — the body drawn at a few phases along
      the path it actually travels, so forward travel, vertical motion and pitch
      appear in one plane — the one a reader pictures the robot swimming in.
      Several cycles are drawn: the solution is periodic, so each repeat is the
      first cycle shifted by one cycle's advance, which spaces the poses to read.

  ``*_traces``  Phase-normalised traces — world-frame base velocity, the
      base position about the steady advance, and the base orientation.  The
      x-axis is cycle phase rather than seconds so that solutions with different
      periods (the co-design sweep spans T = 0.6-1.4 s) lie on the same axis, and
      the cycle is drawn twice so the periodicity is visible rather than asserted.

Frames matter here and the solution file mixes them: ``X[0:3]`` is world-frame
position but ``X[nq:nq+6]`` is the base twist in the *body* frame (see
``hydro_model/trajectory.py``).  With the base pitching ~20 deg per stroke the
two differ by a large fraction of the mean speed, so the velocity plotted is
the body twist rotated into the world frame, not the raw state.

Traces are the N+1 grid nodes.  The saved collocation block ``Xc`` would give
denser samples, but it is stored in tangent coordinates about a reference
quaternion the file does not keep, so the base orientation cannot be recovered
from it.  The node-to-node ripple in the velocity trace is the solution's, not
an artifact of drawing straight lines between samples.

Usage:
  python plot_base_motion.py
  python plot_base_motion.py --solution ../stage1_gait_optimization/codesign/\\
      codesign_results/TLPG50_v0p17_T1p200.npz --save base_motion.pdf
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
# One colour per base component, in index order: x/y/z and roll/pitch/yaw.
# The two rows reuse the three colours for different quantities, which is why
# each keeps its own legend.
TRACE_C = PALETTE[:3]
# Front and hind foot paths, in the drawn legs' order (front first).
FOOT_C = (PALETTE[1], PALETTE[2])
# Panel (a) spans this many cycles; the poses are spread evenly over all of
# them.  More cycles is what buys resolution: the panel is a fixed width, so a
# longer travel puts more x in it, each pose is drawn smaller, and the extra
# room takes the extra poses.
#
# Keep the two coprime.  The poses land on phase (i * CYCLES / PHASES) mod 1, so
# a common factor makes them repeat: 12 over 3 cycles is four phases drawn three
# times each, while 10 over 3 is ten distinct phases of the stroke.
STROBE_CYCLES = 2
STROBE_PHASES = 8     # poses in total, not per cycle
full_width()


def base_pose(X, nq):
    """World position, Euler angles [deg] and world-frame linear velocity."""
    pos = X[0:3, :]
    quat = X[3:7, :]                      # scalar-last (x, y, z, w)
    twist_body = X[nq : nq + 3, :]

    rpy = np.empty_like(pos)
    vel_world = np.empty_like(pos)
    for k in range(X.shape[1]):
        R = pin.Quaternion(*np.roll(quat[:, k], 1)).matrix()   # pin wants (w,x,y,z)
        rpy[:, k] = pin.rpy.matrixToRpy(R)
        vel_world[:, k] = R @ twist_body[:, k]
    return pos, np.degrees(rpy), vel_world


def one_side_front_to_hind(robot, X, nq):
    """One leg per left-right pair, ordered front to hind in the body frame.

    A sagittal view of all four legs draws each pair twice, and the pairs are
    phase-shifted, so one side is what reads.  The order is resolved from the
    model rather than assumed of ``lr_leg_pairs``, because it fixes which end of
    the body the spine starts at.
    """
    legs = [pair[0] for pair in robot.spec.lr_leg_pairs]
    X_tree = expand_to_tree(robot, X, nq)
    robot.forward_kinematics(X_tree[: robot.nq, 0])
    oMb = robot.data.oMi[1]
    body_x = {leg: oMb.actInv(np.asarray(robot.leg_skeleton(leg)[0][0]))[0]
              for leg in legs}
    return sorted(legs, key=lambda leg: -body_x[leg])


def leg_geometry(robot, X, nq, legs):
    """Per-node world-frame skeleton segments, spine endpoints and foot points.

    ``legs`` is one leg per left-right pair, and the spine runs between *those*
    legs' attachments (Side joints) — so the drawing is one consistent side of
    the robot.  Taking the spine from whichever legs sit front-most and
    hind-most across all four instead puts it on the other side, and the base
    yaws enough (~20 deg) that the legs then visibly hang off nothing.  The body
    outline is the model's own joint positions either way, not geometry invented
    for the figure.
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
    """Base position about the steady advance, mean removed.

    The linear ramp in x is 3x the oscillation and would otherwise be the only
    thing that trace shows.
    """
    out = pos.copy()
    out[0] -= np.linspace(pos[0, 0], pos[0, -1], N + 1)
    return out - out.mean(axis=1, keepdims=True)


def plot_side_view(robot, X, nq):
    """Figure 1: stroboscopic side view, full text width."""
    pos, _, _ = base_pose(X, nq)
    N = X.shape[1] - 1
    legs = one_side_front_to_hind(robot, X, nq)
    frames, feet = leg_geometry(robot, X, nq, legs)
    dx_cycle = pos[0, -1] - pos[0, 0]

    # Equal aspect, so the height follows the travel drawn: the panel is as tall
    # as the motion is, and picking anything else just letterboxes it.
    fig, ax_side = plt.subplots(figsize=(TEXT_WIDTH_IN, 2.5))

    # ── stroboscopic side view ──────────────────────────────────────────────
    # One cycle advances less than the body is long, so the poses are spread
    # over every drawn cycle rather than packed into one: the same phases get
    # STROBE_CYCLES times the spacing, and the skeletons stay separable.
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
        # Dashed and black: the leg segments are grey too, and a solid grey
        # path reads as one more limb.
        ax_side.plot(pos[0] + dx, pos[2], color="k", lw=0.7, ls=(0, (4, 2)),
                     zorder=5, label="base path" if cycle == 0 else None)
        # The two feet sweep nearly the same amplitude but in different height
        # bands, so both paths fit without tangling.
        for leg, colour, name in zip(legs, FOOT_C, ("front foot", "hind foot")):
            ax_side.plot(feet[leg][0] + dx, feet[leg][2], color=colour, lw=0.5,
                         alpha=0.65, zorder=2,
                         label=name if cycle == 0 else None)

    ax_side.set_aspect("equal")
    ax_side.set_xlabel(r"$x$ [m]")
    ax_side.set_ylabel(r"$z$ [m]")
    ax_side.grid(alpha=0.3)
    # Above the axes, not in them: the poses reach the top of the frame, and an
    # inside legend either covers them or pushes the ylim past the motion.
    ax_side.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4,
                   handlelength=1.6, frameon=False, columnspacing=1.4,
                   borderaxespad=0.2)
    # An outside legend is invisible to tight_layout, and this canvas is fixed
    # (savefig.bbox is None), so the row has to be paid for explicitly: add its
    # height to the figure and anchor the axes south, which puts all of the
    # slack above the axes where the legend sits rather than splitting it.
    legend_row(fig, ax_side)
    fig.tight_layout(pad=0.3)
    return fig


def plot_phase_traces(X, T, N, nq):
    """Figure 2: phase-normalised base traces, full text width."""
    pos, rpy, vel_world = base_pose(X, nq)
    v_mean = (pos[0, -1] - pos[0, 0]) / T
    dev = _position_deviation(pos, N)

    # Two of the three rows carry a legend above their axes, so the row spacing
    # has to hold one and the canvas has to be that much taller.
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
    # The phase axis drops the period, so the figure has to carry it itself.
    ax_v.margins(y=0.18)
    ax_v.annotate(rf"$T = {T:.2f}$\,s", xy=(0.01, 0.97), xycoords="axes fraction",
                  fontsize=8, ha="left", va="top", color="0.35")
    ax_v.set_ylabel(r"$v_x^{\mathrm{world}}$" "\n" r"[m\,s$^{-1}$]")

    for row, c, lab in zip(range(3), TRACE_C, ("$x$", "$y$", "$z$")):
        ax_p.plot(two, rep(dev[row]) * 1e3, color=c, lw=1.0, label=lab)
    ax_p.set_ylabel("translation\ndeviation [mm]")
    # Own legend per row rather than one shared above the figure: both rows use
    # the same three colours for different quantities, so a combined legend
    # would give blue two labels.
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
    # hspace is set after tight_layout, not via gridspec_kw on subplots(): passing
    # gridspec_kw makes tight_layout treat the whole figure as "not compatible"
    # (it warns and silently falls back to the default rcParams margins instead
    # of computing tight ones), which is what left the excess whitespace on the
    # right of this figure versus the side view.
    fig.tight_layout(pad=0.3)
    fig.subplots_adjust(hspace=0.45)
    return fig


def motion_stats(X, T, N, nq):
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
