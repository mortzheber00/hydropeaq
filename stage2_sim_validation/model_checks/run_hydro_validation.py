#!/usr/bin/env python3
"""Print the hydrodynamic terms at the neutral pose and plot the robot geometry.

``--save hydro.pdf`` writes ``hydro_skeleton.pdf`` and
``hydro_{mesh,drag,buoyancy,overlay}.pdf``; without it the figures are shown.

Usage:
  python stage2_sim_validation/model_checks/run_hydro_validation.py
  python stage2_sim_validation/model_checks/run_hydro_validation.py --robot body2 --save body2_hydro.pdf
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401  registers the 'science' matplotlib style

# resolve() so this also works when __file__ is relative
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from stage1_gait_optimization.hydro_model import SymbolicDynamics, load_robot, registry
from stage1_gait_optimization.hydro_model.visualization import (
    visualize_robot_representations,
    visualize_skeleton,
)

plt.style.use(["science"])
plt.rcParams["text.usetex"] = True
# No tight cropping, so all figures keep the same page size and scale equally in
# LaTeX. (savefig(bbox_inches=None) would fall back to this rcParam instead.)
plt.rcParams["savefig.bbox"] = None

ROBOT = "amph"   # default registered robot name; see hydro_model/robots/

FIGSIZE = (5.0, 4.0)  # shared canvas for all figures


def _derived(base: Path, tag: str) -> str:
    """``<stem>_<tag><suffix>`` next to ``base``."""
    return str(base.with_name(f"{base.stem}_{tag}{base.suffix}"))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--save", type=Path, default=None)
    parser.add_argument("--robot", default=ROBOT, choices=sorted(registry()))
    args = parser.parse_args()

    # --- Robot ---
    robot = load_robot(args.robot)
    q = robot.neutral_config()
    print(robot)
    print()

    # --- Symbolic dynamics ---
    print("Building CasADi symbolic dynamics...")
    dyn = SymbolicDynamics(robot)
    dyn.print_summary()
    print()

    # --- Forces at the neutral pose ---
    tau_buoy = np.array(dyn.f_tau_buoyancy(q)).flatten()
    print(f"Buoyancy joint-space torque (base DOFs): {tau_buoy[:6]}")
    print(f"  Heave (z-force):  {tau_buoy[2]:.4f} N")
    print()

    g_rb = np.array(dyn.f_g_rb(q)).flatten()
    print(f"Gravity joint-space torque (base DOFs): {g_rb[:6]}")
    print(f"  Weight (z-force): {g_rb[2]:.4f} N")
    print()

    print(f"Net vertical (buoyancy - weight): {tau_buoy[2] - g_rb[2]:+.4f} N")
    print()

    v_test = np.zeros(robot.nv)
    v_test[0] = 0.1  # forward body-frame velocity
    tau_drag = np.array(dyn.f_tau_drag(q, v_test)).flatten()
    print(f"Drag joint-space torque at v_x=0.1 m/s (base DOFs): {tau_drag[:6]}")
    print()

    M_added = np.array(dyn.f_M_added(q))
    print("Added-mass matrix (base translational block):")
    print(np.array2string(M_added[:3, :3], precision=6, suppress_small=True))
    print()

    print("── Pinocchio dynamics ──")
    M = robot.mass_matrix(q)
    print(f"Mass matrix shape: {M.shape}")
    print(f"Mass matrix diagonal (first 6): {np.diag(M)[:6]}")
    print()

    feet = robot.foot_positions()
    print("Foot positions (neutral pose):")
    for leg, pos in feet.items():
        print(f"  {leg}: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}] m")
    print()

    # --- Figures ---
    save = args.save
    # Saved here, not via the builders' save arguments, which crop to content.
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
