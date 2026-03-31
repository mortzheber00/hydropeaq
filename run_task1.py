#!/usr/bin/env python3
"""
Task 1 — Simplified hydrodynamic model for the AMPH quadruped robot.

This script:
  1. Parses the URDF via Pinocchio and approximates each link as a cylinder.
  2. Computes buoyancy forces and center of buoyancy per link.
  3. Evaluates viscous drag, pressure-gradient, and added-mass forces
     at a sample configuration and velocity.
  4. Visualises the cylinder-approximated robot.
"""

import matplotlib.pyplot as plt
import numpy as np

from hydro_model import HydrodynamicModel, QuadrupedRobot
from hydro_model.visualization import visualize_robot, visualize_skeleton

URDF_PATH = "urdf/amph.urdf"


def main():
    # ── 1. Parse URDF & build robot model ──────────────────────────────
    robot = QuadrupedRobot(URDF_PATH)
    print(robot)
    print()

    # FK at neutral pose, with test joints angles, then build skeleton-aligned cylinders.
    q = robot.neutral_config()
    robot.forward_kinematics(q)
    robot.build_cylinders()

    # ── 2. Build hydrodynamic model ────────────────────────────────────
    hydro = HydrodynamicModel(robot)
    hydro.print_summary()
    print()

    # ── 3. Evaluate forces at a sample configuration ───────────────────

    # -- Buoyancy --
    F_buoy, T_buoy = hydro.total_buoyancy()
    print(f"Total buoyancy force:  {F_buoy} N")
    print(f"Total buoyancy torque: {T_buoy} N·m")
    print()

    # -- Drag at a sample forward velocity of 0.1 m/s --
    v_forward = np.array([0.1, 0.0, 0.0])
    velocities = {name: v_forward for name in robot.links}
    F_drag = hydro.total_drag(velocities)
    print(f"Total drag at v={v_forward} m/s:  {F_drag} N")
    print()

    # -- Added mass matrix (composite body) --
    M_A = hydro.total_added_mass_matrix()
    print("Total added-mass matrix (translational block):")
    print(np.array2string(M_A[:3, :3], precision=6, suppress_small=True))
    print()

    # -- Added-mass force for a sample acceleration --
    a_sample = np.array([0.05, 0.0, 0.0])
    F_am_total = np.zeros(3)
    for name in hydro.link_hydro:
        F_am_total += hydro.added_mass_force(name, a_sample)
    print(f"Total added-mass force at a={a_sample} m/s²: {F_am_total} N")
    print()

    # -- Pressure-gradient (Froude-Krylov) for still water = 0 --
    F_fk = sum(
        hydro.pressure_gradient_force(name, np.zeros(3)) for name in hydro.link_hydro
    )
    print(f"Pressure-gradient force (still water): {F_fk} N")

    # -- Pinocchio dynamics demo --
    print("\n── Pinocchio dynamics ──")
    M = robot.mass_matrix(q)
    print(f"Mass matrix shape: {M.shape}")
    print(f"Mass matrix diagonal: {np.diag(M)}")
    g = robot.gravity_torque(q)
    print(f"Gravity torque: {g}")

    # -- Foot positions --
    feet = robot.foot_positions()
    print("\nFoot positions (neutral pose):")
    for leg, pos in feet.items():
        print(f"  {leg}: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}] m")

    # ── 4. Visualise ───────────────────────────────────────────────────
    visualize_skeleton(
        robot,
        q,
        title="AMPH — Kinematic Skeleton (Neutral Pose)",
        save_path="robot_skeleton.png",
    )

    visualize_robot(
        robot,
        q,
        title="AMPH — Cylinder Approximation (Neutral Pose)",
        save_path="robot_cylinder_approximation.png",
    )
    plt.show()


if __name__ == "__main__":
    main()
