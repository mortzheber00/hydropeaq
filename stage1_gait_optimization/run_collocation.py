#!/usr/bin/env python3
"""
Direct collocation OCP for efficient swimming gait.

Uses degree-3 Radau collocation within each interval.
Minimises squared joint torques over a periodic swim cycle.
Period T is fixed to avoid symbolic dt ill-conditioning.
"""

from pathlib import Path

import casadi as ca
import mlflow
import numpy as np
from hydro_model import QuadrupedRobot, SymbolicDynamics
from initial_guess import build_initial_guess, build_robot_ik_initial_guess
from ocp_common import _log_solver_stats, diagnose_initial_guess, extract_solution

URDF_PATH = Path(__file__).parent.parent / "src" / "amph" / "urdf" / "amph.urdf"

# ── OCP parameters ──────────────────────────────────────────────────────
N = 30          # collocation intervals
T_FIXED = 1.0   # fixed cycle period [s]
D_TARGET = 0.15  # forward distance per cycle [m]
TAU_MAX = 3.5   # joint torque limit [Nm]
W_TORQUE = 0.0   # weight for sum-of-squared-torques (0 ⇒ feasibility stage)
W_DIST = 0.5    # weight for forward distance reward
W_VEL_SMOOTH = 1.0  # weight for velocity smoothing
W_DRIFT = 1.0   # weight for drift penalty
HEADING_TOL = 0.05  # max |qw*qz + qx*qy| at start/end (≈ yaw/2 for small yaw)

D_COLLOC = 3    # polynomial degree (Radau collocation points)

# ── Initial guess gait (Qu et al. 2025) ─────────────────────────────────
GAIT = "Prototype"
HIND_THIGH_OFFSET = 0.0  # [rad]
HIND_CALF_OFFSET = 0.5   # [rad]


def _collocation_coefficients(d: int):
    """Lagrange basis derivative matrix C and endpoint vector D for Radau collocation.

    tau_root = [0, tau_1, ..., tau_d]  (d+1 points)
    C[i, j] = d/dtau L_i(tau_root[j])
    D[i]    = L_i(1)
    """
    tau_root = np.concatenate([[0.0], np.array(ca.collocation_points(d, "radau"))])
    C = np.zeros((d + 1, d + 1))
    D = np.zeros(d + 1)
    for i in range(d + 1):
        p = np.poly1d([1.0])
        for r in range(d + 1):
            if r != i:
                p *= np.poly1d([1.0, -tau_root[r]]) / (tau_root[i] - tau_root[r])
        D[i] = float(p(1.0))
        pd = np.polyder(p)
        for j in range(d + 1):
            C[i, j] = float(pd(tau_root[j]))
    return tau_root, C, D


def build_ocp():
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("gait_ocp")
    mlflow.start_run(tags={"initial gait": GAIT, "method": "collocation"})
    mlflow.log_params({
        "N": N, "T_FIXED": T_FIXED, "TAU_MAX": TAU_MAX,
        "GAIT": GAIT, "D_TARGET": D_TARGET, "D_COLLOC": D_COLLOC,
        "HEADING_TOL": HEADING_TOL,
    })

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
    q_lb = robot.model.lowerPositionLimit[7:]
    q_ub = robot.model.upperPositionLimit[7:]
    v_lb = -robot.model.velocityLimit[6:]
    v_ub = robot.model.velocityLimit[6:]
    tau_lb = -robot.model.effortLimit[6:]
    tau_ub = robot.model.effortLimit[6:]
    dt_val = T_FIXED / N

    # ── 2. Collocation coefficients ────────────────────────────────────
    tau_root, C, D = _collocation_coefficients(D_COLLOC)
    d = D_COLLOC

    # ── 3. Initial guess ───────────────────────────────────────────────
    print(f"Building initial guess from paper trajectory ({GAIT})...")
    if GAIT in ["LSPG25", "LSPG33", "TLPG50"]:
        X_guess, U_guess = build_initial_guess(
            dyn, GAIT, N, T_FIXED, TAU_MAX,
            hind_thigh_offset=HIND_THIGH_OFFSET,
            hind_calf_offset=HIND_CALF_OFFSET,
        )
    elif GAIT == "Prototype":
        X_guess, U_guess = build_robot_ik_initial_guess(dyn, N, T_FIXED, TAU_MAX)
    else:
        raise ValueError(f"Unknown GAIT: {GAIT}")

    print(f"  Torque guess RMS = {np.sqrt(np.mean(U_guess**2)):.3f} Nm")
    print(f"  Torque guess max = {np.max(np.abs(U_guess)):.3f} Nm")
    np.savez("task3_guess.npz", T=T_FIXED, X=X_guess, U=U_guess, N=N, nq=nq)
    mlflow.log_artifact("task3_guess.npz")
    print("  Initial guess saved to task3_guess.npz")

    # F=None: collocation has no single-step integrator to check defects
    # against; the cost-term breakdown is still useful for weight tuning.
    diagnose_initial_guess(
        X_guess, U_guess, nq, N, T_FIXED,
        W_TORQUE, W_DIST, W_VEL_SMOOTH, W_DRIFT, F=None,
    )

    save = input("Stop optimization after initial guess? [y/N] ").strip().lower()
    if save == "y":
        mlflow.log_param("solver_status", "guess_only")
        mlflow.end_run()
        return

    # ── 4. NLP setup ───────────────────────────────────────────────────
    print("Setting up NLP...")
    opti = ca.Opti()

    X = opti.variable(nx, N + 1)        # states at grid points
    Xc = opti.variable(nx, N * d)       # states at collocation points
    U = opti.variable(n_act, N)         # controls (piecewise constant)

    torque_cost = sum(ca.sumsqr(U[:, k]) for k in range(N)) / N
    dist_cost = -W_DIST * (X[0, -1] - X[0, 0]) / T_FIXED
    vel_smooth_cost = sum(ca.sumsqr(X[nq:nq+6, k+1] - X[nq:nq+6, k]) for k in range(N)) / N
    drift_cost = sum(X[1, k]**2 + (X[2, k] - X[2, 0])**2 for k in range(N + 1)) / (N + 1)

    # Collocation constraints
    for k in range(N):
        uk_full = ca.vertcat(ca.DM.zeros(6, 1), U[:, k])
        # All state points in interval: [x_k, xc_k0, ..., xc_k(d-1)]
        x_all = [X[:, k]] + [Xc[:, k * d + j] for j in range(d)]

        for j in range(1, d + 1):
            # Derivative of interpolating polynomial at collocation point j
            xp = sum(C[i, j] * x_all[i] for i in range(d + 1))
            opti.subject_to(dt_val * dyn.f_xdot(x_all[j], uk_full) == xp)

        # Continuity: polynomial evaluated at end of interval
        x_end = sum(D[i] * x_all[i] for i in range(d + 1))
        opti.subject_to(X[:, k + 1] == x_end)

    # Bounds at grid points
    for k in range(N + 1):
        opti.subject_to(ca.dot(X[3:7, k], X[3:7, k]) == 1.0)
        opti.subject_to(opti.bounded(q_lb, X[7:nq, k], q_ub))
        opti.subject_to(opti.bounded(v_lb, X[nq + 6:, k], v_ub))
        opti.subject_to(opti.bounded(-2.0, X[nq:nq + 6, k], 2.0))
        # Limit Side Joint angles to near zero (indices 7, 10, 13, 16)
        for side_idx in range(7, nq, 3):
            opti.subject_to(opti.bounded(-0.001, X[side_idx, k], 0.001))

    # Bounds at collocation points
    for k in range(N):
        for j in range(d):
            xc_kj = Xc[:, k * d + j]
            opti.subject_to(ca.dot(xc_kj[3:7], xc_kj[3:7]) == 1.0)
            opti.subject_to(opti.bounded(q_lb, xc_kj[7:nq], q_ub))
            opti.subject_to(opti.bounded(v_lb, xc_kj[nq + 6:], v_ub))
            opti.subject_to(opti.bounded(-2.0, xc_kj[nq:nq + 6], 2.0))

    for k in range(N):
        opti.subject_to(opti.bounded(tau_lb, U[:, k], tau_ub))

    # Periodicity
    x0, xN = X[:, 0], X[:, N]
    opti.subject_to(xN[7:nq] == x0[7:nq])
    opti.subject_to(xN[nq + 6:] == x0[nq + 6:])
    opti.subject_to(xN[nq:nq + 6] == x0[nq:nq + 6])
    opti.subject_to(xN[1] == x0[1])
    opti.subject_to(xN[2] == x0[2])
    opti.subject_to(xN[3:7] == x0[3:7])

    opti.subject_to(X[0, 0] == 0.0)
    opti.subject_to(X[1, 0] == 0.0)
    opti.subject_to(xN[0] - x0[0] >= D_TARGET)

    # Heading: bound yaw indicator (qw*qz + qx*qy ≈ yaw/2) at start and end
    for xk in (x0, xN):
        opti.subject_to(opti.bounded(-HEADING_TOL, xk[6] * xk[5] + xk[3] * xk[4], HEADING_TOL))
    opti.minimize(W_TORQUE * torque_cost + dist_cost + W_VEL_SMOOTH * vel_smooth_cost + W_DRIFT * drift_cost)

    # Initial guess: grid points from trajectory, collocation points interpolated
    for k in range(N + 1):
        opti.set_initial(X[:, k], X_guess[:, k])
    for k in range(N):
        for j in range(d):
            alpha = tau_root[j + 1]
            x_interp = (1 - alpha) * X_guess[:, k] + alpha * X_guess[:, k + 1]
            opti.set_initial(Xc[:, k * d + j], x_interp)
        opti.set_initial(U[:, k], U_guess[:, k])

    # ── 5. Solve ───────────────────────────────────────────────────────
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
        mlflow.log_param("solver_status", "optimal")
        _log_solver_stats(opti.stats())
        extract_solution(sol, X, U, nq, N, T_FIXED)
    except RuntimeError as e:
        print("=" * 60)
        print(f"\n* Solver failed: {e}")
        print("  Extracting best iterate...\n")
        mlflow.log_param("solver_status", "failed")
        _log_solver_stats(opti.stats())
        extract_solution(opti.debug, X, U, nq, N, T_FIXED)
    finally:
        mlflow.end_run()


if __name__ == "__main__":
    build_ocp()
