#!/usr/bin/env python3
"""
Optimal Control Problem for efficient swimming gait.

Minimises squared joint torques over a periodic swim cycle.
Period T is fixed to avoid symbolic dt ill-conditioning.
"""

from pathlib import Path

import casadi as ca
import numpy as np
from hydro_model import QuadrupedRobot, SymbolicDynamics
from initial_guess import build_initial_guess

URDF_PATH = Path(__file__).parent.parent / "src" / "amph" / "urdf" / "amph.urdf"

# ── OCP parameters ──────────────────────────────────────────────────────
N = 15  # shooting intervals
T_FIXED = 1.0  # fixed cycle period [s]
D_MIN = 0.001  # minimum forward distance per cycle [m] (avoids trivial solution)
TAU_MAX = 3.5  # joint torque limit [Nm]
W_PERIODIC = 10.0  # weight for soft periodicity terms (y, z, quat, base vel)
W_DIST = 1.0  # weight for forward distance reward (tune relative to W_torque=1)

# ── Initial guess gait (Qu et al. 2025) ─────────────────────────────────
# "LSPG25" : lateral-sequence paddling, 25 % power phase
# "LSPG33" : lateral-sequence paddling, 33 % power phase (fastest in paper)
# "TLPG50" : trot-like paddling,        50 % power phase (most stable)
GAIT = "TLPG50"

# Constant angle offsets [rad] added to both hind leg joints in the initial guess.
HIND_THIGH_OFFSET = 0.0  # [rad]
HIND_CALF_OFFSET = 0.5  # [rad]


def build_rk4_integrator(f_xdot: ca.Function, nq: int, nv: int, dt: float):
    """RK4 integrator with fixed numeric dt. No quaternion re-normalisation
    inside — the unit-norm constraint at each shooting node handles it."""
    nx = nq + nv
    x = ca.SX.sym("x", nx)
    u = ca.SX.sym("u", nv)

    h = dt
    k1 = f_xdot(x, u)
    k2 = f_xdot(x + h / 2 * k1, u)
    k3 = f_xdot(x + h / 2 * k2, u)
    k4 = f_xdot(x + h * k3, u)
    x_next = x + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    return ca.Function("rk4_step", [x, u], [x_next], ["x0", "u"], ["xf"])


def build_ocp():
    # ── 1. Robot & dynamics ─────────────────────────────────────────────
    print("Building robot and symbolic dynamics...")
    robot = QuadrupedRobot(URDF_PATH)
    q0 = np.zeros(robot.nq)
    robot.forward_kinematics(q0)
    robot.build_cylinders()
    dyn = SymbolicDynamics(robot)

    nq, nv = robot.nq, robot.nv
    nx = nq + nv
    n_act = robot.n_actuated
    # Joint angle limits
    q_lb = robot.model.lowerPositionLimit[7:]
    q_ub = robot.model.upperPositionLimit[7:]
    # Joint velocity limits
    v_lb = -robot.model.velocityLimit[6:]
    v_ub = robot.model.velocityLimit[6:]
    # Joint torque limits
    tau_lb = -robot.model.effortLimit[6:]
    tau_ub = robot.model.effortLimit[6:]
    dt_val = T_FIXED / N

    # ── 3. RK4 integrator (fixed dt) ───────────────────────────────────
    print("Building RK4 integrator...")
    F = build_rk4_integrator(dyn.f_xdot, nq, nv, dt_val)

    # ── 4. Kinematic initial guess (Qu et al. 2025) ────────────────────
    print(f"Building initial guess from paper trajectory ({GAIT})...")
    X_guess, U_guess = build_initial_guess(
        dyn,
        GAIT,
        N,
        T_FIXED,
        D_MIN,
        TAU_MAX,
        hind_thigh_offset=HIND_THIGH_OFFSET,
        hind_calf_offset=HIND_CALF_OFFSET,
    )

    print(f"  Torque guess RMS = {np.sqrt(np.mean(U_guess**2)):.3f} Nm")
    print(f"  Torque guess max = {np.max(np.abs(U_guess)):.3f} Nm")
    np.savez("task3_guess.npz", T=T_FIXED, X=X_guess, U=U_guess, N=N, nq=nq)
    print("  Initial guess saved to task3_guess.npz")
    print()

    save = input("Stop optimization after initial guess? [y/N] ").strip().lower()
    if save == "y":
        return

    # ── 5. NLP setup ───────────────────────────────────────────────────
    print("Setting up NLP...")
    opti = ca.Opti()

    X = opti.variable(nx, N + 1)
    U = opti.variable(n_act, N)

    # Objective: minimise torques, maximise forward distance
    torque_cost = 0
    for k in range(N):
        torque_cost += ca.sumsqr(U[:, k])
    torque_cost /= N

    # Dynamics for multiple shooting
    for k in range(N):
        tau_full = ca.vertcat(ca.DM.zeros(6, 1), U[:, k])
        x_next = F(X[:, k], tau_full)
        opti.subject_to(X[:, k + 1] == x_next)

    # Bounds
    for k in range(N + 1):
        # Unit quaternion constraint to prevent ill-conditioning.
        quat_k = X[3:7, k]
        opti.subject_to(ca.dot(quat_k, quat_k) == 1.0)
        # Leg joint angle limits
        opti.subject_to(opti.bounded(q_lb, X[7:nq, k], q_ub))
        # Leg joint velocity limits
        opti.subject_to(opti.bounded(v_lb, X[nq + 6 :, k], v_ub))
        # Base velocity limits to avoid unrealistic speeds
        opti.subject_to(opti.bounded(-2.0, X[nq : nq + 6, k], 2.0))

    for k in range(N):
        opti.subject_to(opti.bounded(tau_lb, U[:, k], tau_ub))

    # Periodicity
    x0, xN = X[:, 0], X[:, N]
    # Hard: joint angles, joint velocities, and base velocity must be exactly periodic
    opti.subject_to(xN[7:nq] == x0[7:nq])
    opti.subject_to(xN[nq + 6 :] == x0[nq + 6 :])
    opti.subject_to(xN[nq : nq + 6] == x0[nq : nq + 6])
    # base pose (y, z, quat) penalised in cost
    opti.subject_to(xN[1] == x0[1])
    opti.subject_to(xN[2] == x0[2])
    opti.subject_to(xN[3:7] == x0[3:7])

    dist = xN[0] - x0[0]
    opti.subject_to(dist == 0.0325)
    opti.minimize(torque_cost)

    # Forward progress & anchors
    opti.subject_to(X[0, 0] == 0.0)
    opti.subject_to(X[1, 0] == 0.0)

    # Initial guess
    for k in range(N + 1):
        opti.set_initial(X[:, k], X_guess[:, k])
    for k in range(N):
        opti.set_initial(U[:, k], U_guess[:, k])

    # ── 6. Solve ───────────────────────────────────────────────────────
    opti.solver(
        "ipopt",
        {"expand": False},
        {
            "max_iter": 1000,
            "tol": 1e-4,
            "acceptable_tol": 1e-3,
            "acceptable_iter": 15,
            "print_level": 5,
            "linear_solver": "mumps",
            "mu_strategy": "adaptive",
            "nlp_scaling_method": "gradient-based",
        },
    )

    print("Solving OCP...")
    print("=" * 60)
    try:
        sol = opti.solve()
        print("=" * 60)
        print("\n* OCP solved!\n")
        extract_solution(sol, X, U, nq)
    except RuntimeError as e:
        print("=" * 60)
        print(f"\n* Solver failed: {e}")
        print("  Extracting best iterate...\n")
        extract_solution(opti.debug, X, U, nq)


def extract_solution(sol, X, U, nq):
    X_val = sol.value(X)
    U_val = sol.value(U)

    print(f"  Cycle period T       = {T_FIXED:.4f} s")
    print(f"  Forward distance     = {X_val[0, -1] - X_val[0, 0]:.4f} m")
    print(f"  Average forward vel  = {(X_val[0, -1] - X_val[0, 0]) / T_FIXED:.4f} m/s")
    print(
        f"  Base z range         = [{X_val[2, :].min():.4f}, {X_val[2, :].max():.4f}] m"
    )
    print(f"  Torque RMS           = {np.sqrt(np.mean(U_val**2)):.4f} Nm")
    print(f"  Torque max |tau|     = {np.max(np.abs(U_val)):.4f} Nm")
    print(f"  Cost (avg tau^2/N)   = {np.sum(U_val**2) / N:.6f}")

    x0, xN = X_val[:, 0], X_val[:, -1]
    print("\n  Periodicity residuals:")
    print(f"    dq_y   = {xN[1] - x0[1]:.2e}")
    print(f"    dq_z   = {xN[2] - x0[2]:.2e}")
    print(f"    dquat  = {np.linalg.norm(xN[3:7] - x0[3:7]):.2e}")
    print(f"    djoints= {np.linalg.norm(xN[7:nq] - x0[7:nq]):.2e}")
    print(f"    dv     = {np.linalg.norm(xN[nq:] - x0[nq:]):.2e}")

    np.savez("task3_solution.npz", T=T_FIXED, X=X_val, U=U_val, N=N, nq=nq)
    print("\n  Solution saved to task3_solution.npz")


if __name__ == "__main__":
    build_ocp()
