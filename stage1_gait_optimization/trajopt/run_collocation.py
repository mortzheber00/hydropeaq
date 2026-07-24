#!/usr/bin/env python3
"""
Direct collocation OCP for efficient swimming gait.

Uses degree-3 Radau collocation within each interval.
Minimises squared joint torques over a periodic swim cycle.
The cycle period T is a free optimization variable; the performance
target is an average forward speed (distance / T), not distance per
cycle, so the optimizer can pick the most efficient stride frequency.
"""

import sys
from pathlib import Path

STAGE1_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STAGE1_DIR))

import mlflow
import numpy as np
from hydro_model import QuadrupedRobot, SymbolicDynamics
from initial_guess import build_initial_guess, build_robot_ik_initial_guess
from ocp_common import (
    _log_solver_stats,
    build_collocation_nlp,
    diagnose_initial_guess,
    extract_solution,
    tangent_to_legacy,
)

URDF_PATH = STAGE1_DIR.parent / "src" / "amph" / "urdf" / "amph.urdf"

# ── OCP parameters ──────────────────────────────────────────────────────
N = 64          # collocation intervals
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
    robot.forward_kinematics(np.zeros(robot.nq))
    robot.build_cylinders()
    dyn = SymbolicDynamics(robot)
    nq = robot.nq

    # ── 2. Initial guess ────────────────────────────────────────────────
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

    # ── 3. NLP setup (shared collocation transcription) ─────────────────
    print("Setting up NLP...")
    nlp = build_collocation_nlp(
        dyn, robot, X_guess, U_guess, N,
        t_lo=T_MIN, t_hi=T_MAX, t_init=T_INIT,
        v_target=V_TARGET, f_c=F_C, heading_tol=HEADING_TOL, d_colloc=D_COLLOC,
    )
    opti = nlp["opti"]
    X, U, T = nlp["X"], nlp["U"], nlp["T"]
    alpha = nlp["alpha"]
    q_ref_quat = nlp["q_ref_quat"]

    # Objective: shared effort / smoothness / drift terms plus a forward-distance
    # reward (this driver rewards distance directly; the codesign evaluator does
    # not, relying on the speed floor instead).
    dist_cost = -W_DIST * (X[0, -1] - X[0, 0]) / T
    opti.minimize(
        W_POWER * nlp["power_cost"]
        + dist_cost
        + W_VEL_SMOOTH * nlp["vel_smooth_cost"]
        + W_DRIFT * nlp["drift_cost"]
    )

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

    # ── 4. Solve ────────────────────────────────────────────────────────
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
