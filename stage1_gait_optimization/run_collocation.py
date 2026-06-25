#!/usr/bin/env python3
"""
Direct collocation OCP for efficient swimming gait.

Uses degree-3 Radau collocation within each interval.
Minimises squared joint torques over a periodic swim cycle.
The cycle period T is a free optimization variable; the performance
target is an average forward speed (distance / T), not distance per
cycle, so the optimizer can pick the most efficient stride frequency.
"""

from pathlib import Path

import casadi as ca
import mlflow
import numpy as np
from hydro_model import QuadrupedRobot, SymbolicDynamics
from initial_guess import build_initial_guess, build_robot_ik_initial_guess
from ocp_common import (
    _log_solver_stats,
    diagnose_initial_guess,
    extract_solution,
    legacy_to_tangent,
    tangent_to_legacy,
)

URDF_PATH = Path(__file__).parent.parent / "src" / "amph" / "urdf" / "amph.urdf"

# ── OCP parameters ──────────────────────────────────────────────────────
N = 32          # collocation intervals
T_INIT = 1.0    # initial-guess cycle period [s] (warm start; T is now free)
T_MIN = 1.0     # cycle-period bounds [s]
T_MAX = 1.0
D_TARGET = 0.2  # forward distance per nominal cycle [m]
V_TARGET = D_TARGET / T_INIT  # required average forward speed [m/s]
TAU_MAX = 3.5   # joint torque limit [Nm]
F_C = 20.0      # actuator bandwidth [Hz] — first-order filter cutoff
W_POWER = 2.0    # weight for sum-of-squared per-joint mechanical power (τ·q̇)²
W_DIST = 0.5    # weight for forward distance reward
W_VEL_SMOOTH = 20.0  # weight for velocity smoothing
W_DRIFT = 10.0   # weight for drift penalty
HEADING_TOL = 0.05  # max yaw angle at endpoint (radians)
ENFORCE_SYMMETRY = False  # LSPG: q_right(t) = q_left(t + T/2) for thigh & calf

D_COLLOC = 3    # polynomial degree (Radau collocation points)

# ── Initial guess gait (Qu et al. 2025) ─────────────────────────────────
GAIT = "LSPG33"  # "LSPG25", "LSPG33", "TLPG50", or "Prototype" (robot IK guess)


def _collocation_coefficients(d: int):
    """Lagrange basis derivative matrix C and endpoint vector D for Radau collocation.

    tau_root = [0, tau_1, ..., tau_d]  (d+1 points)
    C[i, r] = d/dtau L_i(tau_root[r+1])   (r = 0..d-1, the d Radau points)
    D[i]    = L_i(1)
    """
    tau_root = np.concatenate([[0.0], np.array(ca.collocation_points(d, "radau"))])
    C, D, _ = ca.collocation_coeff(ca.collocation_points(d, "radau"))
    return tau_root, np.array(C), np.array(D).flatten()


def build_ocp():
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("gait_ocp")
    mlflow.start_run(tags={"initial gait": GAIT, "method": "collocation"})
    mlflow.log_params({
        "N": N, "T_INIT": T_INIT, "T_MIN": T_MIN, "T_MAX": T_MAX,
        "TAU_MAX": TAU_MAX, "GAIT": GAIT, "V_TARGET": V_TARGET,
        "D_COLLOC": D_COLLOC, "HEADING_TOL": HEADING_TOL, "F_C": F_C,
        "ENFORCE_SYMMETRY": ENFORCE_SYMMETRY,
    })

    # ── 1. Robot & dynamics ─────────────────────────────────────────────
    print("Building robot and symbolic dynamics...")
    robot = QuadrupedRobot(URDF_PATH)
    q0 = np.zeros(robot.nq)
    robot.forward_kinematics(q0)
    robot.build_cylinders()
    dyn = SymbolicDynamics(robot)

    nq, nv = robot.nq, robot.nv
    n_act = robot.n_actuated
    # Tangent state layout (nx = 2*nv = 36):
    # Quaternions live on S^3 (unit-norm constraint), which makes NLP equality
    # constraints nonlinear and introduces redundancy. The tangent state phi
    # linearises orientation around a fixed reference quaternion q_ref, giving
    # a Euclidean R^3 rotation coordinate with no norm constraint needed.
    #   x[0:3]            base position (world)
    #   x[3:6]            base rotation tangent phi  (around q_ref)
    #   x[6 : 6+n_act]    joint positions
    #   x[6+n_act : 12+n_act]  base velocity (body frame)
    #   x[12+n_act:]      joint velocities
    nx = 2 * nv
    # Slices for indexing into tangent state vector.
    # Joint positions
    Q_J = slice(6, 6 + n_act)
    # Base velocity (body frame)
    V_B = slice(6 + n_act, 12 + n_act)
    # Joint velocities
    V_J = slice(12 + n_act, nx)
    
    # Physical bounds
    q_lb = robot.model.lowerPositionLimit[7:]
    q_ub = robot.model.upperPositionLimit[7:]
    v_lb = -robot.model.velocityLimit[6:]
    v_ub = robot.model.velocityLimit[6:]
    tau_lb = -robot.model.effortLimit[6:]
    tau_ub = robot.model.effortLimit[6:]

    # ── 2. Collocation coefficients ────────────────────────────────────
    tau_root, C, D = _collocation_coefficients(D_COLLOC)
    d = D_COLLOC

    # ── 3. Initial guess ───────────────────────────────────────────────
    print(f"Building initial guess from paper trajectory ({GAIT})...")
    if GAIT in ["LSPG25", "LSPG33", "TLPG50"]:
        X_guess, U_guess = build_initial_guess(dyn, GAIT, N, T_INIT, TAU_MAX)
    elif GAIT == "Prototype":
        X_guess, U_guess = build_robot_ik_initial_guess(dyn, N, T_INIT, TAU_MAX)
    else:
        raise ValueError(f"Unknown GAIT: {GAIT}")

    print(f"  Torque guess RMS = {np.sqrt(np.mean(U_guess**2)):.3f} Nm")
    print(f"  Torque guess max = {np.max(np.abs(U_guess)):.3f} Nm")
    np.savez("task3_guess.npz", T=T_INIT, X=X_guess, U=U_guess, N=N, nq=nq)
    mlflow.log_artifact("task3_guess.npz")
    print("  Initial guess saved to task3_guess.npz")

    # F=None: collocation has no single-step integrator to check defects
    # against; the cost-term breakdown is still useful for weight tuning.
    diagnose_initial_guess(
        X_guess, U_guess, nq, N, T_INIT,
        W_POWER, W_DIST, W_VEL_SMOOTH, W_DRIFT, F=None,
    )

    save = input("Stop optimization after initial guess? [y/N] ").strip().lower()
    if save == "y":
        mlflow.log_param("solver_status", "guess_only")
        mlflow.end_run()
        return

    # ── 4. NLP setup ───────────────────────────────────────────────────
    print("Setting up NLP...")
    # Reference quaternion: anchor tangent representation at the guess's
    # initial base orientation, so phi(t=0) = 0 by construction.
    q_ref_quat = X_guess[3:7, 0].copy()
    f_kin, f_inv_dyn = dyn.build_tangent_dynamics(q_ref_quat)
    Xt_guess = legacy_to_tangent(X_guess, q_ref_quat, robot.model)
    
    n_kin = 6 + n_act  # kinematic (position) rows of xt: pos(3) + phi(3) + joints

    opti = ca.Opti()

    X = opti.variable(nx, N + 1)        # tangent states at grid points
    Xc = opti.variable(nx, N * d)       # tangent states at collocation points
    U = opti.variable(n_act, N)         # controls (piecewise constant)

    # Free cycle period. dt is now symbolic (appears as a multiplier in the
    # kinematic defects and a divisor in the dynamic defects); bound T and warm
    # start at T_INIT to keep the NLP well-conditioned.
    T = opti.variable()
    dt = T / N
    opti.subject_to(opti.bounded(T_MIN, T, T_MAX))
    opti.set_initial(T, T_INIT)

    # Cost terms
    # Per-joint mechanical power: τ_j · q̇_j (element-wise). Summed-of-squares so
    # that high power on any single joint is penalized regardless of others.
    power_cost = sum(ca.sumsqr(U[:, k] * X[V_J, k]) for k in range(N)) / N
    dist_cost = -W_DIST * (X[0, -1] - X[0, 0]) / T
    vel_smooth_cost = sum(ca.sumsqr(X[V_B, k + 1] - X[V_B, k]) for k in range(N)) / N
    drift_cost = sum(X[1, k]**2 + (X[2, k] - X[2, 0])**2 for k in range(N + 1)) / (N + 1)

    # Collocation constraints (kinematic + inverse-dynamics split — no M⁻¹)
    for k in range(N):
        # Padding controls with zeros for unactuated base DOF
        uk_full = ca.vertcat(ca.DM.zeros(6, 1), U[:, k])
        
        # States within this interval: x_k, x_kc1, ..., x_kcd
        x_all = [X[:, k]] + [Xc[:, k * d + j] for j in range(d)]

        for j in range(1, d + 1):
            # time derivative at collocation point via Lagrange derivative matrix C
            xp = sum(C[i, j - 1] * x_all[i] for i in range(d + 1))
            
            # Kinematic: dq/dt from R(q)·v_lin etc. equals polynomial deriv.
            # Multiply by dt because xp is derivative w.r.t. interval tau ∈ [0, 1]
            opti.subject_to(dt * f_kin(x_all[j]) == xp[:n_kin])

            # Dynamic: τ = M(q)·v̇ + C·v + g − τ_hydro with v̇ from polynomial.
            # Velocity rows of xp divided by dt give a
            a_poly = xp[n_kin:] / dt
            # inverse dynamics must equal full control input
            opti.subject_to(f_inv_dyn(x_all[j], a_poly) == uk_full)

        # Continuity at interval boundary
        # Evaluates polynomial at tau=1 via endpoint coefficients D
        x_end = sum(D[i] * x_all[i] for i in range(d + 1))
        opti.subject_to(X[:, k + 1] == x_end)

    # Bounds at grid points on tangent state
    for k in range(N + 1):
        opti.subject_to(opti.bounded(q_lb, X[Q_J, k], q_ub))
        opti.subject_to(opti.bounded(v_lb, X[V_J, k], v_ub))
        # Base velocity bounds
        opti.subject_to(opti.bounded(-2.0, X[V_B, k], 2.0))
        # Side joints (indices 6, 9, 12, 15 in tangent layout) pinned to 0.
        for side_idx in range(6, 6 + n_act, 3):
            opti.subject_to(X[side_idx, k] == 0.0)

    # Bounds at collocation points
    for k in range(N):
        for j in range(d):
            xc_kj = Xc[:, k * d + j]
            opti.subject_to(opti.bounded(q_lb, xc_kj[Q_J], q_ub))
            opti.subject_to(opti.bounded(v_lb, xc_kj[V_J], v_ub))
            # Base velocity bounds
            opti.subject_to(opti.bounded(-2.0, xc_kj[V_B], 2.0))

    # Bounds on controls at grid points
    for k in range(N):
        opti.subject_to(opti.bounded(tau_lb, U[:, k], tau_ub))

    # Torque-rate (bandwidth) constraints — paper §3.2 eq. 14-15.
    # First-order filter: u_k ∈ [(1-α)u_{k-1} + α·u̲, (1-α)u_{k-1} + α·ū]
    alpha = 2 * np.pi * dt * F_C / (2 * np.pi * dt * F_C + 1)  # symbolic in T
    for k in range(1, N):
        opti.subject_to(opti.bounded(
            (1 - alpha) * U[:, k - 1] + alpha * tau_lb,
            U[:, k],
            (1 - alpha) * U[:, k - 1] + alpha * tau_ub,
        ))
    # Wrap-around for cyclic gait: U[:,0] is constrained by U[:,N-1]
    opti.subject_to(opti.bounded(
        (1 - alpha) * U[:, N - 1] + alpha * tau_lb,
        U[:, 0],
        (1 - alpha) * U[:, N - 1] + alpha * tau_ub,
    ))

    # Periodicity constraints
    x0, xN = X[:, 0], X[:, N]
    opti.subject_to(xN[Q_J] == x0[Q_J])             # joints
    opti.subject_to(xN[V_J] == x0[V_J])             # joint velocities
    opti.subject_to(xN[V_B] == x0[V_B])             # base velocity
    opti.subject_to(xN[1] == x0[1])                 # base y
    opti.subject_to(xN[2] == x0[2])                 # base z
    opti.subject_to(xN[3:6] == x0[3:6])             # base orientation (phi)

    opti.subject_to(X[0, 0] == 0.0)
    opti.subject_to(X[1, 0] == 0.0)
    opti.subject_to(X[3:6, 0] == 0.0)               # anchor phi at q_ref
    # Average forward speed floor (distance / T >= V_TARGET), written linearly
    # in T to stay well-scaled.
    opti.subject_to(xN[0] - x0[0] >= V_TARGET * T)

    # Heading: phi[2] ≈ yaw angle for small tangent rotations.
    # x0[5] = 0 by anchor above just bound the endpoint.
    opti.subject_to(opti.bounded(-HEADING_TOL, xN[5], HEADING_TOL))

    # ── Left–right symmetry (LSPG) ──────────────────────────────────────
    # Right-leg joints at time t equal left-leg joints at t + T/2.
    # T/2 lands on grid point k + N//2 (needs even N). Periodicity makes the
    # reverse pairing automatic, so one direction per leg pair suffices.
    # Side joints are pinned to 0 (sign-flip under reflection), so only the
    # sagittal-plane thigh/calf positions are constrained — the matching joint
    # velocities follow from the kinematic collocation constraint (q̇ = v), so
    # constraining them too would be redundant and over-determine the NLP.
    if ENFORCE_SYMMETRY:
        assert N % 2 == 0, "LSPG symmetry needs even N so T/2 lands on a grid point"
        half = N // 2
        qj0 = 6           # first joint-position row in tangent state
        # Leg order is [FL, FR, HL, HR]; (right_leg, left_leg) pairs:
        sym_pairs = [(1, 0), (3, 2)]   # FR↔FL, HR↔HL
        for k in range(N):
            kp = (k + half) % N
            for r_leg, l_leg in sym_pairs:
                for off in (1, 2):     # thigh, calf
                    opti.subject_to(X[qj0 + 3 * r_leg + off, k] == X[qj0 + 3 * l_leg + off, kp])

    opti.minimize(W_POWER * power_cost + dist_cost + W_VEL_SMOOTH * vel_smooth_cost + W_DRIFT * drift_cost)

    # Initial guess
    for k in range(N + 1):
        opti.set_initial(X[:, k], Xt_guess[:, k])
    for k in range(N):
        for j in range(d):
            tau_j = tau_root[j + 1]
            x_interp = (1 - tau_j) * Xt_guess[:, k] + tau_j * Xt_guess[:, k + 1]
            opti.set_initial(Xc[:, k * d + j], x_interp)
        opti.set_initial(U[:, k], U_guess[:, k])

    # ── 5. Solve ───────────────────────────────────────────────────────
    opti.solver(
        "ipopt",
        {
            "expand": False,
        },
        {
            "max_iter": 3000,
            "tol": 1e-4,
            "acceptable_tol": 1e-3,
            "acceptable_iter": 15,
            "print_level": 5,
            "linear_solver": "ma97",
            "hsllib": "/usr/local/lib/libcoinhsl.so",
            "ma97_order": "metis",
            "mu_strategy": "adaptive",
            "nlp_scaling_method": "gradient-based",
            "ma97_scaling": "mc64",
        },
    )

    print("Solving OCP...")
    print("=" * 60)
    def _save(src):
        T_val = float(src.value(T))
        Xt_val = src.value(X)
        U_val = src.value(U)
        mlflow.log_param("T_solved", round(T_val, 4))
        mlflow.log_param("bandwidth_alpha", round(float(src.value(alpha)), 4))
        X_val = tangent_to_legacy(Xt_val, q_ref_quat, robot.model)
        extract_solution(X_val, U_val, nq, N, T_val)

    try:
        sol = opti.solve()
        print("=" * 60)
        print("\n* OCP solved!\n")
        mlflow.log_param("solver_status", "optimal")
        _log_solver_stats(opti.stats())
        _save(sol)
    except RuntimeError as e:
        print("=" * 60)
        print(f"\n* Solver failed: {e}")
        print("  Extracting best iterate...\n")
        mlflow.log_param("solver_status", "failed")
        _log_solver_stats(opti.stats())
        _save(opti.debug)
    finally:
        mlflow.end_run()


if __name__ == "__main__":
    build_ocp()
