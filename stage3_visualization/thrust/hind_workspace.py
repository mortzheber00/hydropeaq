#!/usr/bin/env python3
"""Check a leg's commanded (or solved) foot path against its reachable workspace.

``<name>_workspace``  reachable sagittal foot positions (joint-limit sweep; for
                      BODY2 only the assemblable region connected to home) with
                      the path on top, unreachable samples marked
``<name>_angles``     joint angles of the IK / guess / solution vs the limits

The paper-gait IK (paper.py) ignores joint limits and can drive the hind thigh
past its stop; this makes that visible. For the paper gaits the path comes from
Front_Left, so checking a front leg is only a self-consistency test.
``--gait Prototype`` checks the firmware stroke (default for closed-chain
robots); note firmware.py's own reachability check is only active for
closed-chain robots. ``--solution`` checks a solved trajectory instead.

Usage:
  python stage3_visualization/thrust/hind_workspace.py
  python stage3_visualization/thrust/hind_workspace.py --leg Front_Left --save front.pdf
  python stage3_visualization/thrust/hind_workspace.py --solution task3_solution.npz --leg Hind_Left
  python stage3_visualization/thrust/hind_workspace.py --guess task3_guess.npz --resolution 480
  python stage3_visualization/thrust/hind_workspace.py --robot body2 --leg BL --solution task3_solution.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "stage1_gait_optimization"))
from stage3_visualization.common.thesis_style import HALF, PALETTE, half_width  # noqa: E402  also activates the shared style
from stage1_gait_optimization.hydro_model import load_robot
from stage1_gait_optimization.hydro_model.coordinate_map import IdentityMap
from stage1_gait_optimization.hydro_model.trajectory import load_solution
from stage1_gait_optimization.initial_guess.firmware import (
    _DEFAULT_GAIT,
    _firmware_foot_target,
)
from stage1_gait_optimization.initial_guess.paper import (
    _CALF_OFFSET_DEG,
    _THIGH_OFFSET_DEG,
    GAITS,
    _grid_seed,
    _ik_leg,
    hind_target_path,
    paper_fourier_trajectory,
)
from stage1_gait_optimization.ocp_common import limits_for

PROTOTYPE = "Prototype"      # the firmware IK gait

# Side-by-side figures: shared half-width canvas, saved uncropped.
half_width()

FRONT_LEG = "Front_Left"    # source of the paper gaits' foot path
DEFAULT_LEG = "Hind_Left"   # solved by IK in paper.py
QJ0_LEGACY = 7              # first joint row in the quaternion state


def leg_dofs(robot, leg: str) -> tuple[int, int]:
    """Actuated indices of the leg's two sagittal coordinates (the last two per leg).

    amph: thigh and calf (the side joint is pinned at zero); BODY2: both hip servos.
    """
    legs = list(robot.spec.leg_names)
    per = robot.n_actuated // len(legs)
    if per < 2:
        raise SystemExit(f"{robot.spec.name} has {per} actuated joint(s) per "
                         f"leg; this figure sweeps two")
    i0 = per * legs.index(leg) + per - 2
    return i0, i0 + 1


def dof_labels(robot) -> tuple[str, str]:
    """Display names of the two swept coordinates."""
    a, b = robot.spec.leg_joint_labels[-2:]
    return a.replace("_joint", "").lower(), b.replace("_joint", "").lower()


def home_theta(robot) -> np.ndarray:
    """Actuated coordinates at the home pose."""
    home = robot.spec.theta_home
    return (np.zeros(robot.n_actuated) if home is None
            else np.array(home, dtype=float))


def foot_xz(robot, leg: str, ab: np.ndarray) -> np.ndarray:
    """Hip-relative foot ``(x, z)`` for each row ``(a, b)`` of the swept coordinates.

    Other coordinates stay at home; the coordinate map closes the loops.
    """
    i0, i1 = leg_dofs(robot, leg)
    theta = home_theta(robot)
    q = robot.neutral_config()
    out = np.zeros((len(ab), 2))
    hip = None
    for k, (a, b) in enumerate(ab):
        theta[i0], theta[i1] = a, b
        q[robot.n_base_q:] = robot.coord_map.expand_numeric(theta)
        robot.forward_kinematics(q)
        if hip is None:
            hip = robot.leg_skeleton(leg)[0][0]
        out[k] = (robot.foot_positions()[leg] - hip)[[0, 2]]
    return out


def connected(ok: np.ndarray, seed) -> np.ndarray:
    """Cells of ``ok`` 4-connected to ``seed`` (no wrap-around)."""
    out = np.zeros_like(ok)
    if not ok[seed]:
        return out
    out[seed] = True
    stack = [seed]
    n, m = ok.shape
    while stack:
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < n and 0 <= nc < m and ok[nr, nc] and not out[nr, nc]:
                out[nr, nc] = True
                stack.append((nr, nc))
    return out


def reachable_set(robot, leg: str, resolution: int) -> np.ndarray:
    """Reachable hip-relative foot positions from a sweep over the joint-limit box.

    Keeps feasible cells connected to the home pose (relevant for closed chains;
    all cells for serial legs). Slightly larger than what the OCP allows, since
    its pose constraint keeps a margin.
    """
    i0, i1 = leg_dofs(robot, leg)
    q_lb, q_ub = limits_for(robot)[:2]
    a_ax = np.linspace(q_lb[i0], q_ub[i0], resolution)
    b_ax = np.linspace(q_lb[i1], q_ub[i1], resolution)

    home = home_theta(robot)
    theta = home.copy()
    ok = np.ones((resolution, resolution), bool)
    # Other legs stay at home and are always feasible.
    if robot.coord_map.feasibility(theta).numel():
        for ia, a in enumerate(a_ax):
            for ib, b in enumerate(b_ax):
                theta[i0], theta[i1] = a, b
                ok[ia, ib] = np.asarray(
                    robot.coord_map.feasibility(theta)).min() > 0.0

    seed = (int(np.abs(a_ax - home[i0]).argmin()),
            int(np.abs(b_ax - home[i1]).argmin()))
    island = connected(ok, seed)
    if not island.any():
        raise SystemExit(f"{robot.spec.name}: {leg}'s home pose is infeasible, "
                         f"so the sweep has nothing to grow from")
    ia, ib = np.nonzero(island)
    return foot_xz(robot, leg, np.column_stack([a_ax[ia], b_ax[ib]]))


def commanded_path(robot, gait: str, samples: int, leg: str) -> np.ndarray:
    """Hip-relative foot path the guess builder commands for ``leg``.

    Paper gaits: one path from Front_Left (transformed for hind legs).
    Prototype: a per-leg firmware stroke.
    """
    t = np.arange(samples) / samples
    if gait != PROTOTYPE:
        # Paper gaits are thigh/calf angles, only meaningful for serial legs.
        if not isinstance(robot.coord_map, IdentityMap):
            raise SystemExit(
                f"{robot.spec.name} is a closed-chain robot, so the paper "
                f"gaits' thigh/calf angles do not apply to it; "
                f"use --gait {PROTOTYPE}")
        theta1, theta2 = paper_fourier_trajectory(GAITS[gait])
        front = foot_xz(robot, FRONT_LEG, np.column_stack([
            np.radians(theta1(t) + _THIGH_OFFSET_DEG),
            np.radians(_CALF_OFFSET_DEG - theta2(t)),
        ]))
        # Hind legs get the transformed path (paper.hind_target_path).
        return front if leg.startswith("Front") else hind_target_path(front,
                                                                     robot.spec)

    # The firmware stroke is relative to the trim foot. Trim only moves the base,
    # so the hip-relative home foot is the same reference (no dynamics needed).
    # This breaks if find_trim_state ever changes the joint angles.
    p = {**_DEFAULT_GAIT, **robot.spec.firmware_gait}
    i0, i1 = leg_dofs(robot, leg)
    ref = foot_xz(robot, leg, home_theta(robot)[[i0, i1]][None])[0]

    is_front = list(robot.spec.leg_names).index(leg) < 2
    cx = p["center_x_front"] if is_front else p["center_x_rear"]
    # Firmware depths are positive downward; world z is up.
    dz_surface = -(p["depth_surface"] - p["stand_h"])
    dz_deep = -(p["depth_deep"] - p["stand_h"])
    # Leg phase offsets do not change the path shape.
    out = [_firmware_foot_target(tc, 1.0, p["ratio_recovery"], p["ratio_strike"],
                                 p["ratio_power"], cx + p["stroke_len"],
                                 cx - p["stroke_len"], dz_surface, dz_deep)
           for tc in t]
    return ref + np.array(out)


def solve_leg_angles(robot, gait: str, leg: str, n: int):
    """``(thigh, calf)`` as paper.py computes them for the current spec.

    Same seed and IK sequence as paper.py, so limit violations show up as they
    would in the guess.
    """
    theta1, theta2 = paper_fourier_trajectory(GAITS[gait])
    t = np.arange(n) / n
    thigh = np.radians(theta1(t) + _THIGH_OFFSET_DEG)
    calf = np.radians(_CALF_OFFSET_DEG - theta2(t))
    if leg.startswith("Front"):
        return thigh, calf          # driven directly; no IK involved

    front = foot_xz(robot, FRONT_LEG, np.column_stack([thigh, calf]))
    target = hind_target_path(front, robot.spec)
    out = np.zeros((n, 2))
    seed = _grid_seed(robot, leg, target[0])
    for k in range(n):
        seed = _ik_leg(robot, leg, target[k], seed)
        out[k] = seed
    return out[:, 0], out[:, 1]


def guess_leg_angles(robot, path: Path, leg: str) -> tuple[np.ndarray, np.ndarray]:
    """The leg's two swept coordinates from a saved initial guess."""
    d = np.load(path, allow_pickle=True)
    X, n = d["X"], d["U"].shape[1]
    i0, i1 = leg_dofs(robot, leg)
    return X[QJ0_LEGACY + i0, :n], X[QJ0_LEGACY + i1, :n]


def solution_leg_angles(robot, path: Path, leg: str) -> tuple[np.ndarray, np.ndarray]:
    """The leg's two swept coordinates per node of a solution (last node dropped)."""
    d = load_solution(str(path))
    if d["robot"] != robot.spec.name:
        raise SystemExit(f"{path.name} is a {d['robot']} solution, but this "
                         f"figure is being drawn for {robot.spec.name}")
    X = d["X"]
    i0, i1 = leg_dofs(robot, leg)
    return X[QJ0_LEGACY + i0, :d["N"]], X[QJ0_LEGACY + i1, :d["N"]]


def foot_path(robot, leg: str, q_th, q_ca) -> np.ndarray:
    """Hip-relative ``(x, z)`` the foot actually reaches at these angles."""
    return foot_xz(robot, leg, np.column_stack([q_th, q_ca]))


def plot_workspace(robot, cloud, traj, outside, q_th, q_ca, n_nodes, leg,
                   overlay: str | None):
    """Workspace and joint-angle figures as ``{name: fig}``.

    ``overlay`` labels the crosses for the angles' foot positions; None means
    the path itself comes from those angles (solution) and the crosses mark its nodes.
    """
    lo, hi = limits_for(robot)[:2]
    i_th, i_ca = leg_dofs(robot, leg)
    lbl_th, lbl_ca = dof_labels(robot)

    figs = {}
    fig, ax = plt.subplots(figsize=HALF)
    figs["workspace"] = fig

    # Rasterised to keep the PDF small
    ax.plot(cloud[:, 0] * 100, cloud[:, 1] * 100, ".", ms=0.7, color="0.84",
            rasterized=True, zorder=0)
    handles = [plt.Line2D([], [], ls="", marker="s", ms=5, color="0.84",
                          label="reachable")]
    if outside.any():
        # Points, since a line would bridge the unreachable gaps
        ax.plot(traj[~outside, 0] * 100, traj[~outside, 1] * 100, ".", ms=2.2,
                color=PALETTE[0], zorder=3)
        ax.plot(traj[outside, 0] * 100, traj[outside, 1] * 100, ".", ms=2.2,
                color=PALETTE[2], zorder=4)
        handles += [
            plt.Line2D([], [], ls="", marker=".", ms=8, color=PALETTE[0],
                       label="commanded"),
            plt.Line2D([], [], ls="", marker=".", ms=8, color=PALETTE[2],
                       label=rf"unreachable ({outside.mean() * 100:.0f}\%)"),
        ]
    else:
        # Fully reachable: draw as one closed curve
        closed = np.vstack([traj, traj[:1]])
        ax.plot(closed[:, 0] * 100, closed[:, 1] * 100, "-", lw=1.3,
                color=PALETTE[0], zorder=3)
        key = dict(lw=1.3, color=PALETTE[0], label="trajectory")
        if overlay is None and n_nodes:
            # Mark the solution nodes
            ax.plot(traj[:, 0] * 100, traj[:, 1] * 100, "x", ms=3.2, mew=0.7,
                    color="k", zorder=5)
            key.update(marker="x", ms=4, mew=0.8, mec="k",
                       label=rf"trajectory ($N{{=}}{n_nodes}$)")
        handles.append(plt.Line2D([], [], **key))
    ax.set_xlabel(r"hip-relative $x$ [cm]")
    ax.set_ylabel(r"hip-relative $z$ [cm]")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)

    if n_nodes and overlay:
        ach = foot_path(robot, leg, q_th, q_ca)
        ax.plot(ach[:, 0] * 100, ach[:, 1] * 100, "x", ms=3.2, mew=0.7,
                color="k", zorder=5)
        handles.append(plt.Line2D([], [], ls="", marker="x", ms=4, mew=0.8,
                                  color="k", label=rf"{overlay} ($N{{=}}{n_nodes}$)"))
    ax.legend(handles=handles, loc="best", framealpha=0.92)
    fig.tight_layout()

    # No angles, no angle figure
    if not n_nodes:
        return figs

    fig2, ax2 = plt.subplots(figsize=HALF)
    figs["angles"] = fig2
    ph = np.arange(n_nodes) / n_nodes
    q_th, q_ca = np.degrees(q_th), np.degrees(q_ca)
    for q, i, lbl, col in ((q_th, i_th, lbl_th, PALETTE[2]),
                           (q_ca, i_ca, lbl_ca, PALETTE[0])):
        for b in (np.degrees(lo[i]), np.degrees(hi[i])):
            ax2.axhline(b, color=col, lw=0.9, ls="--", zorder=1)
        ax2.plot(ph, q, "-o", ms=2.0, lw=1.2, color=col, zorder=3,
                 label=rf"{lbl}")
    # Place each limit label at the end of the cycle where the curve is farther away
    for q, i, lbl, col, at_lo in ((q_th, i_th, lbl_th, PALETTE[2], True),
                                  (q_ca, i_ca, lbl_ca, PALETTE[0], False)):
        y = np.degrees(lo[i]) + 3 if at_lo else np.degrees(hi[i]) - 3
        third = max(1, len(q) // 3)
        right = np.abs(q[-third:] - y).min() > np.abs(q[:third] - y).min()
        ax2.text(0.988 if right else 0.012, y, f"{lbl} limits", fontsize=7,
                 color=col, va="bottom" if at_lo else "top",
                 ha="right" if right else "left")
    ax2.set_xlabel("cycle phase [-]")
    ax2.set_ylabel("joint angle [deg]")
    ax2.set_xlim(0, 1)
    ax2.grid(alpha=0.3)
    # y range covers the limits and the trajectory
    span = [np.degrees(lo[i_th]), np.degrees(hi[i_th]),
            np.degrees(lo[i_ca]), np.degrees(hi[i_ca]),
            q_th.min(), q_th.max(), q_ca.min(), q_ca.max()]
    pad = 0.16 * (max(span) - min(span))
    ax2.set_ylim(min(span) - pad, max(span) + pad)
    ax2.legend(loc="upper right", ncol=2, framealpha=0.92)
    fig2.tight_layout()
    return figs


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot", default=None,
                        help="registered robot name; default: the robot the "
                             "--solution or --guess file records, else amph. "
                             "Passing it explicitly against a file that "
                             "disagrees is an error rather than a silent "
                             "reinterpretation")
    parser.add_argument("--leg", default=None,
                        help=f"leg to check against the commanded path "
                             f"(default {DEFAULT_LEG}, or the robot's first leg "
                             f"if it has no such leg); a front leg is reachable "
                             f"by construction, see the module docstring")
    parser.add_argument("--gait", default=None,
                        choices=sorted(GAITS) + [PROTOTYPE],
                        help=f"gait whose foot path is commanded; the paper "
                             f"gaits share one path across legs, {PROTOTYPE} has "
                             f"one per leg (default LSPG25, or {PROTOTYPE} for a "
                             f"closed-chain robot, whose legs the paper gaits' "
                             f"thigh/calf angles do not describe)")
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--guess", type=Path, default=None,
                     help="overlay this saved guess instead of recomputing "
                          "the IK; a saved file records the spec as it was "
                          "when that run happened, so it can disagree with "
                          "the path drawn here")
    src.add_argument("--solution", type=Path, default=None,
                     help="plot this solved trajectory's own foot path against "
                          "the workspace, instead of the guess builder's "
                          "commanded path")
    parser.add_argument("--resolution", type=int, default=320,
                        help="joint-sweep samples per axis for the reachable set")
    parser.add_argument("--samples", type=int, default=720,
                        help="samples along the commanded foot path")
    parser.add_argument("--save", type=Path, default=None,
                        help="write the figure here (format from the extension)")
    args = parser.parse_args()

    # Default robot from the input file; an explicit --robot is checked against it.
    src_file = args.solution or args.guess
    if args.robot is None:
        args.robot = ("amph" if src_file is None or not src_file.exists()
                      else load_solution(str(src_file))["robot"])

    robot = load_robot(args.robot)
    legs = list(robot.spec.leg_names)
    # Robot-dependent defaults
    if args.leg is None:
        args.leg = DEFAULT_LEG if DEFAULT_LEG in legs else legs[0]
    if args.gait is None:
        args.gait = "LSPG25" if isinstance(robot.coord_map, IdentityMap) else PROTOTYPE
    if args.leg not in legs:
        raise SystemExit(f"unknown leg {args.leg!r}; {args.robot} has {legs}")
    cloud = reachable_set(robot, args.leg, args.resolution)

    q_th = q_ca = np.zeros(0)
    n_nodes = 0
    overlay = None
    if args.solution is not None:
        # Path from the solution's own kinematics
        if not args.solution.exists():
            raise SystemExit(f"no solution at {args.solution}")
        q_th, q_ca = solution_leg_angles(robot, args.solution, args.leg)
        n_nodes = len(q_th)
        traj = foot_path(robot, args.leg, q_th, q_ca)
        source = f"{args.solution.name} foot path (N={n_nodes})"
    else:
        traj = commanded_path(robot, args.gait, args.samples, args.leg)
        source = f"{args.gait} commanded foot path"

    # Distance to the reachable cloud; within one grid step counts as reachable.
    excursion = cKDTree(cloud).query(traj)[0] * 1000.0
    tol = 1000.0 * np.linalg.norm(cloud.max(0) - cloud.min(0)) / args.resolution
    outside = excursion > tol

    print(f"{source} vs. {args.robot} {args.leg} workspace "
          f"({len(cloud)} sweep samples, {tol:.2f} mm resolution):")
    print(f"  {outside.sum()}/{len(traj)} samples unreachable "
          f"({outside.mean() * 100:.0f}% of the cycle), "
          f"max {excursion.max():.1f} mm outside at phase "
          f"{excursion.argmax() / len(traj):.3f}")
    if args.solution is not None:
        print("  (the OCP holds its own joint limits, so anything unreachable "
              "here is a disagreement between those limits and this sweep)")
    elif args.leg == FRONT_LEG and args.gait != PROTOTYPE:
        print(f"  ({FRONT_LEG} defines the commanded path, so 0% is the only "
              f"possible answer — this is a self-consistency check)")

    # Angle overlay (a solution's angles are already loaded)
    if args.solution is None:
        overlay = "driven angles" if args.leg.startswith("Front") else "IK solution"
        if args.guess is None and args.gait != PROTOTYPE:
            n_nodes = robot.spec.ocp.n
            q_th, q_ca = solve_leg_angles(robot, args.gait, args.leg, n_nodes)
            print(f"  overlay: IK solved here at N={n_nodes} against the path above")
        elif args.guess is None:
            print(f"  (no overlay: the {PROTOTYPE} guess comes from the firmware "
                  f"builder, which this script does not reproduce — pass --guess)")
        elif args.guess.exists():
            q_th, q_ca = guess_leg_angles(robot, args.guess, args.leg)
            n_nodes = len(q_th)
            # The guess file records no gait, so warn about the assumption.
            print(f"  overlay from {args.guess.name} (N={n_nodes}), assumed to be a "
                  f"{args.gait} guess — the file records no gait, and none of the "
                  f"spec it was built under")
        else:
            print(f"  (no guess at {args.guess}; skipping the angle overlay)")

    # Joint-limit margins of the overlay angles
    if n_nodes:
        lo, hi = limits_for(robot)[:2]
        for lbl, i, q in zip(dof_labels(robot), leg_dofs(robot, args.leg),
                             (q_th, q_ca)):
            slack = np.degrees(min(q.min() - lo[i], hi[i] - q.max()))
            # Within solver tolerance of the limit
            if abs(slack) < 1e-3:
                verdict = "at the stop"
            else:
                verdict = f"{'margin' if slack > 0 else 'PAST THE STOP by'} {abs(slack):.1f} deg"
            print(f"    {lbl:<5s} [{np.degrees(q.min()):7.2f}, {np.degrees(q.max()):7.2f}] "
                  f"vs limits [{np.degrees(lo[i]):7.2f}, {np.degrees(hi[i]):7.2f}] deg"
                  f"  -> {verdict}")

    figs = plot_workspace(robot, cloud, traj, outside, q_th, q_ca, n_nodes,
                          args.leg, overlay)
    if args.save:
        for name, fig in figs.items():
            path = args.save.with_name(f"{args.save.stem}_{name}{args.save.suffix}")
            # Uncropped (see half_width())
            fig.savefig(path, dpi=300)
            print(f"Saved → {path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
