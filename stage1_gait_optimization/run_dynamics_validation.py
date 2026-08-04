#!/usr/bin/env python3
"""
CasADi-symbolic dynamics validation.

This script:
  1. Builds the Pinocchio model and cylinder geometry.
  2. Constructs CasADi symbolic functions for all dynamics terms.
  3. Verifies the symbolic functions against Pinocchio numeric results.
  4. Demonstrates forward and inverse dynamics with hydrodynamic forces.
"""

import numpy as np
from hydro_model import SymbolicDynamics, load_robot

ROBOT = "body2"   # registered robot name; see hydro_model/robots/



def main():
    # ── 1. Build robot + cylinders ─────────────────────────────────────
    # load_robot already runs FK at the neutral configuration and builds the
    # cylinders.  Do not redo it with np.zeros(nq): that is not a valid
    # configuration for a robot with continuous joints, whose entries are
    # (cos, sin) pairs, and rebuilding from it yields degenerate cylinders.
    robot = load_robot(ROBOT)
    print(robot)
    print()

    # ── 2. Build symbolic dynamics ─────────────────────────────────────
    print("Building CasADi symbolic dynamics...")
    dyn = SymbolicDynamics(robot)
    dyn.print_summary()
    print()

    # ── 2. Trim state ──────────────────────────────────────────────────
    print("\nFinding trim state...")
    q_trim = dyn.find_trim_state()
    print(f"\nq_trim = {q_trim}")
    print()

    # ── 3. Verify against Pinocchio numeric ────────────────────────────
    # A pose off the home configuration, written in the robot's actuated
    # coordinates and expanded onto the tree.  The perturbation is kept small
    # so a closed-chain robot stays inside its assemblable set.
    spec = robot.spec
    theta_home = (np.zeros(robot.n_actuated) if spec.theta_home is None
                  else np.asarray(spec.theta_home, dtype=float))
    theta_test = theta_home + 0.05 * np.sin(np.arange(robot.n_actuated) + 1.0)

    q_test = robot.neutral_config()
    q_test[0:3] = [0.1, 0.0, 0.0]
    q_test[3:7] = [0.0, 0.0, 0.0, 1.0]
    q_test[7:] = robot.coord_map.expand_numeric(theta_test)
    v_test = np.ones(robot.nv) * 0.05

    # Mass matrix
    M_sym = np.array(dyn.f_M_rb(q_test))
    M_pin = robot.mass_matrix(q_test)
    M_pin = np.triu(M_pin) + np.triu(M_pin, 1).T
    err_M = np.max(np.abs(M_sym - M_pin))
    print(f"Mass matrix max error (symbolic vs Pinocchio): {err_M:.2e}")

    # Gravity vector
    g_sym = np.array(dyn.f_g_rb(q_test)).flatten()
    g_pin = robot.gravity_torque(q_test)
    err_g = np.max(np.abs(g_sym - g_pin))
    print(f"Gravity vector max error: {err_g:.2e}")

    # FK foot positions
    robot.forward_kinematics(q_test)
    for leg in robot.spec.leg_names:
        pos_sym = np.array(dyn.f_foot_pos[leg](q_test)).flatten()
        pos_pin = robot.foot_positions()[leg]
        err = np.linalg.norm(pos_sym - pos_pin)
        print(f"  Foot {leg} FK error: {err:.2e}")
    print()

    # ── 4. Evaluate dynamics ───────────────────────────────────────────
    tau_zero = np.zeros(robot.nv)

    a_free = dyn.eval_forward_dynamics(q_test, v_test, tau_zero)
    print("Forward dynamics (zero torque, v=0.05):")
    print(f"  qdd = {a_free}")
    print()

    tau_hold = dyn.eval_inverse_dynamics(
        q_test, np.zeros(robot.nv), np.zeros(robot.nv)
    )

    print("Inverse dynamics (hold position, zero velocity/acceleration):")
    print(f"  tau = {tau_hold}")
    print()

    # ── 5. Hydrodynamic contributions ──────────────────────────────────
    tau_buoy = np.array(dyn.f_tau_buoyancy(q_test)).flatten()
    tau_drag = np.array(dyn.f_tau_drag(q_test, v_test)).flatten()
    M_added = np.array(dyn.f_M_added(q_test))

    print("Hydrodynamic joint-space contributions:")
    print(f"  tau_buoyancy = {tau_buoy}")
    print(f"  tau_drag     = {tau_drag}")
    print(f"  M_added diag = {np.diag(M_added)}")
    print(f"  M_added / M_rb ratio (diag): {np.diag(M_added) / np.diag(M_sym)}")
    print()

    # ── 6. State-space ODE ─────────────────────────────────────────────
    x0 = np.concatenate([q_test, v_test])
    xdot = np.array(dyn.f_xdot(x0, tau_zero)).flatten()
    print("State-space ODE (x = [q, v]):")
    print(f"  xdot[:nq]  = v     = {xdot[: robot.nq]}")
    print(f"  xdot[nq:]  = qdd   = {xdot[robot.nq :]}")


if __name__ == "__main__":
    main()
