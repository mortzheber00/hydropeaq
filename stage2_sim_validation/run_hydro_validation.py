#!/usr/bin/env python3
"""
Hydrodynamic model validation.

This script:
  1. Parses the URDF via Pinocchio and approximates each link as a cylinder.
  2. Builds the CasADi symbolic dynamics (including hydrodynamics).
  3. Evaluates buoyancy, drag, and added-mass at a sample configuration.
  4. Visualises the cylinder-approximated robot.

Usage:
  python run_hydro_validation.py
  python run_hydro_validation.py --save hydro.pdf

``--save`` writes the skeleton and the four geometry representations as
``hydro_skeleton`` / ``hydro_{mesh,drag,buoyancy,overlay}`` (format from the
extension); with no ``--save`` the figures are shown.
"""

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401  registers the 'science' matplotlib style
sys.path.insert(0, str(Path(__file__).parents[1]))
from stage1_gait_optimization.hydro_model import QuadrupedRobot, SymbolicDynamics
from stage1_gait_optimization.hydro_model.visualization import visualize_robot_representations, visualize_skeleton

# Professional thesis style with real LaTeX text rendering (Computer Modern).
plt.style.use(["science"])
plt.rcParams["text.usetex"] = True

URDF_PATH = Path(__file__).parent.parent / "src" / "amph" / "urdf" / "amph.urdf"


def _derived(base: Path, tag: str) -> str:
    """``<stem>_<tag><suffix>`` next to ``base``, as a string for savefig."""
    return str(base.with_name(f"{base.stem}_{tag}{base.suffix}"))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--save", type=Path, default=None)
    args = parser.parse_args()

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
    save = args.save
    visualize_skeleton(
        robot,
        q,
        title="AMPH -- Kinematic Skeleton (Neutral Pose)",
        save_path=_derived(save, "skeleton") if save else None,
    )

    visualize_robot_representations(
        robot,
        q,
        title="AMPH -- Geometry (Neutral Pose)",
        save_prefix=str(save.with_suffix("")) if save else None,
        save_suffix=save.suffix if save else ".png",
    )

    if save:
        print(f"Saved → {_derived(save, 'skeleton')}")
        for kind in ("mesh", "drag", "buoyancy", "overlay"):
            print(f"Saved → {_derived(save, kind)}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
