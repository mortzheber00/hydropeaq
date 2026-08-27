#!/usr/bin/env python3
"""
Hind-leg reachable workspace vs. the foot path the paper gait commands.

``initial_guess/paper.py`` drives the front legs with the paper's paddling
trajectory and then hands the resulting hip-relative foot path to the hind legs
by inverse kinematics, so that every foot traces the same path.  ``_ik_leg`` is
an unconstrained damped-least-squares solve — it never sees the URDF joint
limits, and ``_grid_seed`` searches a joint box wider than they are — so it
tracks the path by driving the hind thigh through its mechanical stop rather
than reporting that the path is out of range.

This figure is the check that was missing.  The left panel sweeps the hind
leg over its *actual* joint limits to get the reachable set in the sagittal
plane, and draws the commanded path on top of it, split into the part that fits
and the part that does not.  The right panel shows the joint angles the IK
returned against those same limits.

The commanded path always comes from ``Front_Left`` because ``paper.py``
hardcodes it there; ``--leg`` picks which leg is checked against it.  Running it
on a front leg is a self-consistency check and can only report 0 % unreachable
— the path *is* that leg's forward kinematics — but the joint-angle panel still
shows how much margin the paddling stroke leaves to its stops.  The right- and
left-side legs are the same motion at a phase offset.

Note the guess overlay is only as fine as the OCP grid it was built on — the
IK is solved once per collocation node, so a coarse ``N`` samples the stroke
coarsely.  The commanded path is drawn from the analytic Fourier trajectory and
is independent of ``N``.

Usage:
  python hind_workspace.py
  python hind_workspace.py --leg Front_Left --save ../docs/figures/front_workspace.pdf
  python hind_workspace.py --gait LSPG33 --save ../docs/figures/hind_workspace.pdf
  python hind_workspace.py --guess ../task3_guess.npz --resolution 480
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from thesis_style import PALETTE  # also activates the shared plot style

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stage1_gait_optimization"))
from stage1_gait_optimization.hydro_model import load_robot
from stage1_gait_optimization.initial_guess.paper import (
    _CALF_OFFSET_DEG,
    _THIGH_OFFSET_DEG,
    GAITS,
    _leg_foot_xz,
    paper_fourier_trajectory,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GUESS = REPO_ROOT / "task3_guess.npz"

# paper.py defines the commanded foot path as this leg's own foot path.
FRONT_LEG = "Front_Left"
DEFAULT_LEG = "Hind_Left"   # the leg paper.py solves by IK, and the one at risk
# Legacy state layout: 7 base coordinates, then the actuated joints in spec order.
QJ0_LEGACY = 7


def reachable_set(robot, leg: str, resolution: int) -> np.ndarray:
    """Hip-relative (x, z) foot positions over the leg's thigh x calf limit box."""
    names = robot.actuated_joint_names
    lo, hi = robot.model.lowerPositionLimit[7:], robot.model.upperPositionLimit[7:]
    i_th, i_ca = names.index(f"{leg}_Thigh_joint"), names.index(f"{leg}_Calf_joint")
    th = np.linspace(lo[i_th], hi[i_th], resolution)
    ca = np.linspace(lo[i_ca], hi[i_ca], resolution)
    return np.array([_leg_foot_xz(robot, leg, a, b) for a in th for b in ca])


def commanded_path(robot, gait: str, samples: int) -> np.ndarray:
    """The front leg's own foot path — what the hind IK is asked to reproduce."""
    theta1, theta2 = paper_fourier_trajectory(GAITS[gait])
    t = np.arange(samples) / samples
    return np.array([
        _leg_foot_xz(robot, FRONT_LEG,
                     np.radians(a + _THIGH_OFFSET_DEG),
                     np.radians(_CALF_OFFSET_DEG - b))
        for a, b in zip(theta1(t), theta2(t))
    ])


def guess_leg_angles(robot, path: Path, leg: str) -> tuple[np.ndarray, np.ndarray]:
    """``(thigh, calf)`` the guess builder produced for ``leg``.

    For a front leg these are the paper angles applied directly; for a hind leg
    they are what ``_ik_leg`` returned.
    """
    d = np.load(path, allow_pickle=True)
    X, n = d["X"], d["U"].shape[1]
    names = robot.actuated_joint_names
    rows = [QJ0_LEGACY + names.index(f"{leg}_{j}") for j in ("Thigh_joint", "Calf_joint")]
    return X[rows[0], :n], X[rows[1], :n]


def plot_workspace(robot, cloud, cmd, outside, q_th, q_ca, gait, n_nodes, leg):
    driven = leg.startswith("Front")   # front legs are driven, hind legs IK-solved
    short = leg.replace("_", " ").lower()
    names = robot.actuated_joint_names
    lo, hi = robot.model.lowerPositionLimit[7:], robot.model.upperPositionLimit[7:]
    i_th = names.index(f"{leg}_Thigh_joint")
    i_ca = names.index(f"{leg}_Calf_joint")
    th_lo = np.degrees(lo[i_th])

    # Without a guess there are no IK angles to show, so the joint-angle panel
    # is dropped rather than drawn empty; the workspace panel stands alone.
    if n_nodes:
        fig, (ax, ax2) = plt.subplots(
            1, 2, figsize=(9.8, 4.1), gridspec_kw=dict(width_ratios=[1.25, 1]))
        ph = np.arange(n_nodes) / n_nodes
        q_th, q_ca = np.degrees(q_th), np.degrees(q_ca)
        viol = th_lo - q_th
    else:
        fig, ax = plt.subplots(figsize=(5.4, 4.1))
        ax2 = None

    # Sampling the limit box gives a point cloud, not a polygon; drawn as a
    # rasterised stipple so the PDF does not carry resolution**2 vector dots.
    ax.plot(cloud[:, 0] * 100, cloud[:, 1] * 100, ".", ms=0.7, color="0.84",
            rasterized=True, zorder=0)
    ax.plot(cmd[~outside, 0] * 100, cmd[~outside, 1] * 100, ".", ms=2.2,
            color=PALETTE[0], zorder=3)
    ax.plot(cmd[outside, 0] * 100, cmd[outside, 1] * 100, ".", ms=2.2,
            color=PALETTE[2], zorder=4)
    ax.set_xlabel(r"hip-relative $x$ [cm]")
    ax.set_ylabel(r"hip-relative $z$ [cm]")
    ax.set_aspect("equal")
    ax.set_title(rf"Commanded foot path vs.\ {short} workspace")
    handles = [
        plt.Line2D([], [], ls="", marker="s", ms=5, color="0.84",
                   label="reachable within joint limits"),
        plt.Line2D([], [], ls="", marker=".", ms=8, color=PALETTE[0],
                   label="commanded, reachable"),
        plt.Line2D([], [], ls="", marker=".", ms=8, color=PALETTE[2],
                   label=rf"commanded, unreachable ({outside.mean() * 100:.0f}\%)"),
    ]
    if n_nodes:
        ach = np.array([_leg_foot_xz(robot, leg, np.radians(a), np.radians(b))
                        for a, b in zip(q_th, q_ca)])
        ax.plot(ach[:, 0] * 100, ach[:, 1] * 100, "x", ms=4.2, mew=0.9,
                color="k", zorder=5)
        handles.append(plt.Line2D([], [], ls="", marker="x", ms=5, mew=1.0,
                                  color="k", label=(rf"driven angles ($N{{=}}{n_nodes}$)" if driven
                                         else rf"IK solution ($N{{=}}{n_nodes}$)")))
    ax.legend(handles=handles, loc="best", fontsize=7, framealpha=0.92)

    if ax2 is None:
        fig.tight_layout()
        return fig

    for q, i, lbl, col in ((q_th, i_th, "thigh", PALETTE[2]),
                           (q_ca, i_ca, "calf", PALETTE[0])):
        for b in (np.degrees(lo[i]), np.degrees(hi[i])):
            ax2.axhline(b, color=col, lw=0.9, ls="--", zorder=1)
        ax2.plot(ph, q, "-o", ms=2.6, lw=1.3, color=col, zorder=3,
                 label=rf"{lbl}")
    # Only the thigh runs out of range on this robot, and only on a hind leg;
    # a leg that stays inside its stops gets neither the shading nor the arrow.
    if viol.max() > 0:
        ax2.fill_between(ph, th_lo, q_th, where=q_th < th_lo, color=PALETTE[2],
                         alpha=0.30, lw=0, zorder=2,
                         label="outside the joint limit")
        ax2.annotate(rf"{viol.max():.0f}$^\circ$ past the stop, "
                     rf"{(viol > 0).mean() * 100:.0f}\% of the cycle",
                     xy=(ph[viol.argmax()], q_th.min()), xytext=(0.06, -57),
                     fontsize=7.5, color=PALETTE[2],
                     arrowprops=dict(arrowstyle="->", color=PALETTE[2], lw=0.8))
    ax2.text(0.012, th_lo + 3, "thigh limits", fontsize=6.5, color=PALETTE[2],
             va="bottom")
    ax2.text(0.012, np.degrees(hi[i_ca]) - 3, "calf limits", fontsize=6.5,
             color=PALETTE[0], va="top")
    ax2.set_xlabel("cycle phase [-]")
    ax2.set_ylabel("joint angle [deg]")
    ax2.set_xlim(0, 1)
    # Span the stops plus whatever the trajectory does outside them, so a leg
    # that stays in range is not drawn on a scale sized for one that does not.
    span = [np.degrees(lo[i_th]), np.degrees(hi[i_th]),
            np.degrees(lo[i_ca]), np.degrees(hi[i_ca]),
            q_th.min(), q_th.max(), q_ca.min(), q_ca.max()]
    pad = 0.16 * (max(span) - min(span))
    ax2.set_ylim(min(span) - pad, max(span) + pad)
    ax2.set_title(f"{short.capitalize()} angles the guess uses")
    ax2.legend(loc="upper right", fontsize=7, ncol=2, framealpha=0.92)

    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot", default="amph", help="registered robot name")
    parser.add_argument("--leg", default=DEFAULT_LEG,
                        help=f"leg to check against the commanded path "
                             f"(default {DEFAULT_LEG}); a front leg is reachable "
                             f"by construction, see the module docstring")
    parser.add_argument("--gait", default="LSPG25", choices=sorted(GAITS),
                        help="paper gait whose foot path is commanded")
    parser.add_argument("--guess", type=Path, default=DEFAULT_GUESS,
                        help="initial-guess npz to overlay; skipped if missing")
    parser.add_argument("--resolution", type=int, default=320,
                        help="joint-sweep samples per axis for the reachable set")
    parser.add_argument("--samples", type=int, default=720,
                        help="samples along the commanded foot path")
    parser.add_argument("--save", type=Path, default=None,
                        help="write the figure here (format from the extension)")
    args = parser.parse_args()

    robot = load_robot(args.robot)
    if args.leg not in robot.spec.leg_names:
        raise SystemExit(f"unknown leg {args.leg!r}; "
                         f"{args.robot} has {list(robot.spec.leg_names)}")
    cloud = reachable_set(robot, args.leg, args.resolution)
    cmd = commanded_path(robot, args.gait, args.samples)

    # Distance to the nearest reachable configuration: ~0 inside the workspace,
    # the excursion itself outside.  The sweep is a finite grid, so anything
    # within one grid step counts as reachable.
    excursion = cKDTree(cloud).query(cmd)[0] * 1000.0
    tol = 1000.0 * np.linalg.norm(cloud.max(0) - cloud.min(0)) / args.resolution
    outside = excursion > tol

    print(f"{args.gait} commanded foot path vs. {args.leg} workspace "
          f"({len(cloud)} sweep samples, {tol:.2f} mm resolution):")
    print(f"  {outside.sum()}/{args.samples} samples unreachable "
          f"({outside.mean() * 100:.0f}% of the cycle), "
          f"max {excursion.max():.1f} mm outside at phase "
          f"{excursion.argmax() / args.samples:.3f}")
    if args.leg == FRONT_LEG:
        print(f"  ({FRONT_LEG} defines the commanded path, so 0% is the only "
              f"possible answer — this is a self-consistency check)")

    q_th = q_ca = np.zeros(0)
    n_nodes = 0
    if args.guess.exists():
        q_th, q_ca = guess_leg_angles(robot, args.guess, args.leg)
        n_nodes = len(q_th)
        names = robot.actuated_joint_names
        lo = robot.model.lowerPositionLimit[7:]
        hi = robot.model.upperPositionLimit[7:]
        # The npz carries no gait tag, so an overlay from a run of a different
        # gait would be drawn against the wrong commanded path without warning.
        print(f"  overlay assumed to be a {args.gait} guess ({args.guess} "
              f"records no gait)")
        print(f"  {args.guess.name}: N={n_nodes}")
        for lbl, q in (("thigh", q_th), ("calf", q_ca)):
            i = names.index(f"{args.leg}_{lbl.capitalize()}_joint")
            slack = np.degrees(min(q.min() - lo[i], hi[i] - q.max()))
            print(f"    {lbl:<5s} [{np.degrees(q.min()):7.2f}, {np.degrees(q.max()):7.2f}] "
                  f"vs limits [{np.degrees(lo[i]):7.2f}, {np.degrees(hi[i]):7.2f}] deg"
                  f"  -> {'margin' if slack >= 0 else 'PAST THE STOP by'} "
                  f"{abs(slack):.1f} deg")
    else:
        print(f"  (no guess at {args.guess}; skipping the angle overlay)")

    fig = plot_workspace(robot, cloud, cmd, outside, q_th, q_ca, args.gait,
                         n_nodes, args.leg)
    if args.save:
        # pad_inches above the default: the tight bbox under-measures usetex text.
        fig.savefig(args.save, dpi=300, bbox_inches="tight", pad_inches=0.15)
        print(f"Saved → {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
