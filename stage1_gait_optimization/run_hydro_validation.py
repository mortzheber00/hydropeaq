#!/usr/bin/env python3
"""
Hydrodynamic model validation.

This script:
  1. Parses the URDF via Pinocchio and approximates each link as a cylinder.
  2. Builds the CasADi symbolic dynamics (including hydrodynamics).
  3. Evaluates buoyancy, drag, and added-mass at a sample configuration.
  4. Visualises the cylinder-approximated robot.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from hydro_model import QuadrupedRobot, SymbolicDynamics
from hydro_model.visualization import visualize_robot, visualize_skeleton

URDF_PATH = Path(__file__).parent.parent / "urdf" / "amph.urdf"


def main():
    # ── 1. Parse URDF & build robot model ──────────────────────────────
    robot = QuadrupedRobot(URDF_PATH)
    print(robot)
    print()

    # FK at neutral pose, then build skeleton-aligned cylinders.
    q = robot.neutral_config()
    robot.forward_kinematics(q)
    robot.build_cylinders()

    # ── 2. Build symbolic dynamics (includes hydro) ────────────────────
    print("Building CasADi symbolic dynamics...")
    dyn = SymbolicDynamics(robot)
    dyn.print_summary()
    print()

    # ── 3. Evaluate forces at neutral configuration ────────────────────
    # -- Buoyancy --
    tau_buoy = np.array(dyn.f_tau_buoyancy(q)).flatten()
    print(f"Buoyancy joint-space torque (base DOFs): {tau_buoy[:6]}")
    print(f"  Heave (z-force):  {tau_buoy[2]:.4f} N")
    print()

    # -- Gravity --
    g_rb = np.array(dyn.f_g_rb(q)).flatten()
    print(f"Gravity joint-space torque (base DOFs): {g_rb[:6]}")
    print(f"  Weight (z-force): {g_rb[2]:.4f} N")
    print()

    # Net vertical
    print(f"Net vertical (buoyancy - weight): {tau_buoy[2] - g_rb[2]:+.4f} N")
    print()

    # -- Drag at a sample velocity --
    v_test = np.zeros(robot.nv)
    v_test[0] = 0.1  # forward body-frame velocity
    tau_drag = np.array(dyn.f_tau_drag(q, v_test)).flatten()
    print(f"Drag joint-space torque at v_x=0.1 m/s (base DOFs): {tau_drag[:6]}")
    print()

    # -- Added mass matrix --
    M_added = np.array(dyn.f_M_added(q))
    print("Added-mass matrix (base translational block):")
    print(np.array2string(M_added[:3, :3], precision=6, suppress_small=True))
    print()

    # -- Pinocchio dynamics --
    print("── Pinocchio dynamics ──")
    M = robot.mass_matrix(q)
    print(f"Mass matrix shape: {M.shape}")
    print(f"Mass matrix diagonal (first 6): {np.diag(M)[:6]}")
    print()

    # -- Foot positions --
    feet = robot.foot_positions()
    print("Foot positions (neutral pose):")
    for leg, pos in feet.items():
        print(f"  {leg}: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}] m")
    print()

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
