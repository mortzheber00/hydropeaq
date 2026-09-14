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
  python run_hydro_validation.py --robot body2 --save body2_hydro.pdf

``--save`` writes the skeleton and the four geometry representations as
``hydro_skeleton`` / ``hydro_{mesh,drag,buoyancy,overlay}`` (format from the
extension); with no ``--save`` the figures are shown.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401  registers the 'science' matplotlib style

# resolve() first: run as "python run_hydro_validation.py" from this directory,
# __file__ is relative and parents[1] does not exist, which is what the usage
# line above asks for.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from stage1_gait_optimization.hydro_model import SymbolicDynamics, load_robot, registry
from stage1_gait_optimization.hydro_model.visualization import (
    visualize_robot_representations,
    visualize_skeleton,
)

# Professional thesis style with real LaTeX text rendering (Computer Modern).
plt.style.use(["science"])
plt.rcParams["text.usetex"] = True
# 'science' sets savefig.bbox to 'tight', which crops every figure to its own
# content and so hands LaTeX five slightly different page sizes; at a common
# \includegraphics width they would each scale differently and their type would
# not match.  None keeps the canvas, which FIGSIZE makes identical for all five.
# (savefig(bbox_inches=None) would not do this — that means "use this rcParam".)
plt.rcParams["savefig.bbox"] = None

ROBOT = "amph"   # default registered robot name; see hydro_model/robots/

# One canvas for every figure here.  The 3D content only reaches ~5.9 x 5.8 in
# of it, so the rest is margin: nothing is clipped by dropping the crop above.
FIGSIZE = (5.0, 4.0)


def _derived(base: Path, tag: str) -> str:
    """``<stem>_<tag><suffix>`` next to ``base``, as a string for savefig."""
    return str(base.with_name(f"{base.stem}_{tag}{base.suffix}"))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--save", type=Path, default=None)
    parser.add_argument("--robot", default=ROBOT, choices=sorted(registry()))
    args = parser.parse_args()

    # ── 1. Parse URDF & build robot model ──────────────────────────────
    # load_robot runs FK at the neutral pose and builds the cylinders.
    robot = load_robot(args.robot)
    q = robot.neutral_config()
    print(robot)
    print()

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
    # Untitled: each of these goes into its own LaTeX float, where the caption
    # describes it.  Saved here rather than through the builders' own save
    # arguments, which crop to content — writing them from one place is what
    # keeps the five files a single page size.
    figs = {"skeleton": visualize_skeleton(robot, q, figsize=FIGSIZE)}
    figs.update(visualize_robot_representations(robot, q, figsize=FIGSIZE))

    if save:
        for tag, fig in figs.items():
            fig.savefig(_derived(save, tag), dpi=300)
            print(f"Saved → {_derived(save, tag)}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
