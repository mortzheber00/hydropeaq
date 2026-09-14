#!/usr/bin/env python3
"""
Visualise the OCP solution saved in task3_solution.npz.

Produces:
  1. Joint angle trajectories (thigh/calf/side per leg) over the cycle.
  2. Foot position trajectories (x, z vs time).
  3. Base state trajectories (position + velocity).
  4. Animated 3D skeleton of the swim cycle.

Usage:
  python plot_solution.py
  python plot_solution.py --solution ../task3_solution.npz --save ocp.pdf

``--save`` writes one vector file per static figure (``*_joint_angles`` /
``*_base_state`` / ``*_foot_positions``, format from the extension) plus the
animation as ``*_swim_cycle.gif``; with no ``--save`` the figures are shown.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "stage1_gait_optimization"))
import thesis_style  # noqa: E402,F401  activates the shared style on import

from stage1_gait_optimization.hydro_model import get_spec, load_robot  # noqa: E402
from stage1_gait_optimization.hydro_model.trajectory import (  # noqa: E402
    expand_to_tree,
)
from stage1_gait_optimization.hydro_model.trajectory import (  # noqa: E402
    load_solution as _load,
)


def plot_joint_angles(spec, X, T, N, nq):
    """One row per leg, one column per actuated joint of a leg."""
    legs, labels = spec.leg_names, spec.leg_joint_labels
    n_per_leg = len(labels)
    fig, axes = plt.subplots(len(legs), n_per_leg,
                             figsize=(4.7 * n_per_leg, 2.5 * len(legs)),
                             sharex=True, squeeze=False)
    t = np.linspace(0, T, N + 1)

    for i, leg in enumerate(legs):
        for j, jtype in enumerate(labels):
            ax = axes[i, j]
            q_idx = 7 + i * n_per_leg + j
            ax.plot(t, X[q_idx, :], "b-o", markersize=3, label="angle [rad]")
            ax.axhline(0, color="k", linewidth=0.5, linestyle=":")
            ax.set_ylabel("rad")
            if i == 0:
                ax.set_title(jtype)
            if j == 0:
                ax.set_ylabel(f"{leg.replace('_', ' ')}\nrad")
            ax.grid(True, alpha=0.3)

    axes[-1, 1].set_xlabel("Time [s]")
    fig.suptitle("Joint Angle Trajectories (OCP Solution)", fontsize=13)
    fig.tight_layout()
    return fig


def plot_base_state(X, T, N, nq):
    """Base position (x, y, z) and velocities."""
    t = np.linspace(0, T, N + 1)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))

    labels_pos = ["x [m]", "y [m]", "z [m]"]
    labels_vel = ["vx [m/s]", "vy [m/s]", "vz [m/s]"]

    for j in range(3):
        axes[0, j].plot(t, X[j, :], "b-o", markersize=3)
        axes[0, j].set_title(labels_pos[j])
        axes[0, j].grid(True, alpha=0.3)

        axes[1, j].plot(t, X[nq + j, :], "r-o", markersize=3)
        axes[1, j].set_title(labels_vel[j])
        axes[1, j].set_xlabel("Time [s]")
        axes[1, j].grid(True, alpha=0.3)

    fig.suptitle("Base State Trajectories (OCP Solution)", fontsize=13)
    fig.tight_layout()
    return fig


def plot_foot_positions(robot, X, T, N, nq):
    """Foot trajectories over the cycle, in the base frame.

    Body-relative rather than world: base heave and pitch are comparable to the
    stroke itself, so a world-frame plot shows mostly the base moving.
    """
    LEG_NAMES = robot.spec.leg_names
    t = np.linspace(0, T, N + 1)
    X_tree = expand_to_tree(robot, X, nq)
    nq_tree = robot.nq

    foot_traj = {leg: {"x": [], "y": [], "z": []} for leg in LEG_NAMES}

    for k in range(N + 1):
        q_k = X_tree[:nq_tree, k]
        robot.forward_kinematics(q_k)
        # oMi[1] is the free-flyer placement, and it is a view into robot.data,
        # so read it inside the loop.  actInv undoes the base rotation as well
        # as its translation, which subtracting the base position would not.
        oMb = robot.data.oMi[1]
        feet = robot.foot_positions()
        for leg in LEG_NAMES:
            pos = oMb.actInv(feet[leg])
            foot_traj[leg]["x"].append(pos[0])
            foot_traj[leg]["y"].append(pos[1])
            foot_traj[leg]["z"].append(pos[2])

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    colors = thesis_style.LEG_COLORS
    ylabel = [r"$x_{\mathrm{body}}$ [m]", r"$y_{\mathrm{body}}$ [m]",
              r"$z_{\mathrm{body}}$ [m]"]

    for i, leg in enumerate(LEG_NAMES):
        for ax, key, yl in zip(axes, ["x", "y", "z"], ylabel):
            ax.plot(
                t,
                foot_traj[leg][key],
                color=colors[i],
                marker="o",
                markersize=3,
                label=leg.replace("_", " "),
            )

    for ax, yl in zip(axes, ylabel):
        ax.set_ylabel(yl)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    axes[-1].set_xlabel("Time [s]")
    fig.suptitle("Foot Positions in the Body Frame over Swim Cycle (OCP Solution)",
                 fontsize=13)
    fig.tight_layout()
    return fig


def animate_skeleton(robot, X, T, N, nq):
    """Animated 3D skeleton cycling through all N+1 poses.

    The skeleton is the robot's cylinder chain, so a closed-chain leg draws
    both of its sub-chains without any extra bookkeeping here.
    """
    LEG_NAMES = robot.spec.leg_names
    X_tree = expand_to_tree(robot, X, nq)
    nq_tree = robot.nq

    frames = []
    for k in range(N + 1):
        robot.forward_kinematics(X_tree[:nq_tree, k])
        feet = robot.foot_positions()
        leg_data = {leg: {"segments": robot.leg_skeleton(leg), "foot": feet[leg]}
                    for leg in LEG_NAMES}
        frames.append({"base": np.array(robot.data.oMi[1].translation), "legs": leg_data})

    all_pts = []
    for f in frames:
        all_pts.append(f["base"])
        for leg in LEG_NAMES:
            all_pts.append(f["legs"][leg]["foot"])
            for a, b in f["legs"][leg]["segments"]:
                all_pts.extend((a, b))
    all_pts = np.array(all_pts)
    mid = all_pts.mean(axis=0)
    span = (all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2 * 1.3

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=25, azim=-60)

    ax.set_title("")
    colors_leg = thesis_style.LEG_COLORS

    def draw_frame(k):
        LEG_NAMES = robot.spec.leg_names
        ax.cla()
        ax.set_xlim(mid[0] - span, mid[0] + span)
        ax.set_ylim(mid[1] - span, mid[1] + span)
        ax.set_zlim(mid[2] - span, mid[2] + span)
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")
        ax.set_title(f"Swim cycle, t = {k * T / N:.3f} s  (frame {k}/{N})")

        f = frames[k]
        base = f["base"]
        ax.scatter(*base, s=60, c="k", zorder=5)

        for i, leg in enumerate(LEG_NAMES):
            pts = f["legs"][leg]
            c = colors_leg[i]
            for a, b in pts["segments"]:
                ax.plot(*zip(a, b), color=c, linewidth=2.0, alpha=0.85)
                ax.scatter(*a, s=30, c="k", zorder=5)
            if pts["segments"]:
                ax.plot(*zip(base, pts["segments"][0][0]), color=c,
                        linewidth=2.0, alpha=0.85)
            ax.scatter(
                *pts["foot"],
                s=50,
                c=c,
                marker="v",
                edgecolors="k",
                linewidths=0.5,
                zorder=5,
            )

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xx, yy = np.meshgrid(xlim, ylim)
        ax.plot_surface(
            xx, yy, np.zeros_like(xx), alpha=0.08, color="cyan", zorder=0
        )

        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor=c, label=leg.replace("_", " "))
            for c, leg in zip(colors_leg, LEG_NAMES)
        ]
        ax.legend(handles=legend_elements, loc="upper left", fontsize=8)

    ani = animation.FuncAnimation(
        fig,
        draw_frame,
        frames=N + 1,
        interval=int(T / (N + 1) * 1000),
        repeat=True,
    )
    return fig, ani


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--solution", type=Path,
        default=Path(__file__).parent.parent / "task3_solution.npz",
    )
    parser.add_argument("--save", type=Path, default=None)
    parser.add_argument("--robot", default=None,
                        help="registered robot name; default: read from the solution")
    args = parser.parse_args()

    print(f"Loading solution from {args.solution}...")
    meta = _load(str(args.solution))
    X, U, T, N, nq = meta["X"], meta["U"], meta["T"], meta["N"], meta["nq"]
    robot_name = args.robot or meta["robot"]
    if args.robot and args.robot != meta["robot"]:
        raise SystemExit(f"solution is for {meta['robot']!r}, not {args.robot!r}")
    spec = get_spec(robot_name)
    print(f"  Robot: {robot_name}  (coords={meta['coords']}, n_theta={meta['n_theta']})")
    print(f"  State shape:   {X.shape}  (nx={X.shape[0]}, steps={X.shape[1]})")
    print(f"  Control shape: {U.shape}")
    print(f"  T={T:.3f}s, N={N}, nq={nq}")
    print()

    print("Summary:")
    print(f"  Forward distance : {X[0, -1] - X[0, 0]:.4f} m")
    print(f"  Average speed    : {(X[0, -1] - X[0, 0]) / T:.4f} m/s")
    print(f"  Base z range     : [{X[2, :].min():.4f}, {X[2, :].max():.4f}] m")
    print(f"  Torque RMS       : {np.sqrt(np.mean(U**2)):.4f} Nm")
    print(f"  Torque max |tau| : {np.max(np.abs(U)):.4f} Nm")
    print()

    robot = load_robot(robot_name)

    figs = {
        "joint_angles": plot_joint_angles(spec, X, T, N, nq),
        "base_state": plot_base_state(X, T, N, nq),
        "foot_positions": plot_foot_positions(robot, X, T, N, nq),
    }
    _, ani = animate_skeleton(robot, X, T, N, nq)

    if args.save:
        for tag, fig in figs.items():
            path = args.save.with_name(f"{args.save.stem}_{tag}{args.save.suffix}")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Saved → {path}")
        gif_path = args.save.with_name(f"{args.save.stem}_swim_cycle.gif")
        ani.save(str(gif_path), writer=animation.PillowWriter(fps=max(1, round((N + 1) / T))))
        print(f"Saved → {gif_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
