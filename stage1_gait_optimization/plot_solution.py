#!/usr/bin/env python3
"""
Visualise the OCP solution saved in task3_solution.npz.

Produces:
  1. Joint angle trajectories (thigh/calf/side per leg) over the cycle.
  2. Foot position trajectories (x, z vs time).
  3. Base state trajectories (position + velocity).
  4. Animated 3D skeleton of the swim cycle.
"""

import sys
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from hydro_model import QuadrupedRobot

URDF_PATH = Path(__file__).parent.parent / "src" / "amph" / "urdf" / "amph.urdf"
SOL_PATH = "task3_solution.npz"

if len(sys.argv) > 1:
    SOL_PATH = sys.argv[1]

LEG_NAMES = ["Front_Left", "Front_Right", "Hind_Left", "Hind_Right"]
JOINT_TYPES = ["Side", "Thigh", "Calf"]


def load_solution(path: str):
    data = np.load(path)
    return data["X"], data["U"], float(data["T"]), int(data["N"]), int(data["nq"])


def plot_joint_angles(X, T, N, nq):
    """4x3 grid: one row per leg, one column per joint type."""
    fig, axes = plt.subplots(4, 3, figsize=(14, 10), sharex=True)
    t = np.linspace(0, T, N + 1)

    for i, leg in enumerate(LEG_NAMES):
        for j, jtype in enumerate(JOINT_TYPES):
            ax = axes[i, j]
            q_idx = 7 + i * 3 + j
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
    fig.savefig("ocp_joint_angles.png", dpi=150, bbox_inches="tight")
    print("Saved ocp_joint_angles.png")
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
    fig.savefig("ocp_base_state.png", dpi=150, bbox_inches="tight")
    print("Saved ocp_base_state.png")
    return fig


def plot_foot_positions(robot, X, T, N, nq):
    """Foot x/z trajectories over the cycle."""
    t = np.linspace(0, T, N + 1)

    foot_traj = {leg: {"x": [], "y": [], "z": []} for leg in LEG_NAMES}

    for k in range(N + 1):
        q_k = X[:nq, k]
        robot.forward_kinematics(q_k)
        feet = robot.foot_positions()
        for leg in LEG_NAMES:
            pos = feet[leg]
            foot_traj[leg]["x"].append(pos[0] - X[0, k])
            foot_traj[leg]["y"].append(pos[1])
            foot_traj[leg]["z"].append(pos[2])

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    ylabel = ["x - base_x [m]", "y [m]", "z [m]"]

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
    fig.suptitle("Foot Positions over Swim Cycle (OCP Solution)", fontsize=13)
    fig.tight_layout()
    fig.savefig("ocp_foot_positions.png", dpi=150, bbox_inches="tight")
    print("Saved ocp_foot_positions.png")
    return fig


def animate_skeleton(robot, X, T, N, nq):
    """Animated 3D skeleton cycling through all N+1 poses."""
    frames = []
    for k in range(N + 1):
        q_k = X[:nq, k]
        robot.forward_kinematics(q_k)

        base_pos = np.array(robot.data.oMi[1].translation)
        leg_data = {}
        for leg in LEG_NAMES:
            proj = robot.leg_centerline_positions(leg)
            leg_data[leg] = {
                "side": proj["side"],
                "thigh": proj["thigh"],
                "calf": proj["calf"],
                "foot": proj["foot"],
            }
        frames.append({"base": base_pos, "legs": leg_data})

    all_pts = []
    for f in frames:
        all_pts.append(f["base"])
        for leg in LEG_NAMES:
            for v in f["legs"][leg].values():
                all_pts.append(v)
    all_pts = np.array(all_pts)
    mid = all_pts.mean(axis=0)
    span = (all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2 * 1.3

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=25, azim=-60)

    ax.set_title("")
    colors_leg = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    def draw_frame(k):
        ax.cla()
        ax.set_xlim(mid[0] - span, mid[0] + span)
        ax.set_ylim(mid[1] - span, mid[1] + span)
        ax.set_zlim(mid[2] - span, mid[2] + span)
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")
        ax.set_title(f"Swim Cycle — t = {k * T / N:.3f} s  (frame {k}/{N})")

        f = frames[k]
        base = f["base"]
        ax.scatter(*base, s=60, c="k", zorder=5)

        for i, leg in enumerate(LEG_NAMES):
            pts = f["legs"][leg]
            c = colors_leg[i]
            chain = [base, pts["side"], pts["thigh"], pts["calf"], pts["foot"]]
            for a, b in zip(chain[:-1], chain[1:]):
                ax.plot(*zip(a, b), color=c, linewidth=2.0, alpha=0.85)
            for key in ["side", "thigh", "calf"]:
                ax.scatter(*pts[key], s=30, c="k", zorder=5)
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
    print(f"Loading solution from {SOL_PATH}...")
    X, U, T, N, nq = load_solution(SOL_PATH)
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

    robot = QuadrupedRobot(URDF_PATH)
    robot.forward_kinematics(np.zeros(robot.nq))
    robot.build_cylinders()

    plot_joint_angles(X, T, N, nq)
    plot_base_state(X, T, N, nq)
    plot_foot_positions(robot, X, T, N, nq)
    _, ani = animate_skeleton(robot, X, T, N, nq)

    plt.show()

    save = input("Save animation to ocp_swim_cycle.gif? [y/N] ").strip().lower()
    if save == "y":
        writer = animation.PillowWriter(fps=max(1, N // int(T)))
        ani.save("ocp_swim_cycle.gif", writer=writer)
        print("Saved ocp_swim_cycle.gif")


if __name__ == "__main__":
    main()
