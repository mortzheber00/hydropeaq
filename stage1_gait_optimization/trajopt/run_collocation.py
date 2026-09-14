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
from hydro_model import SymbolicDynamics, load_robot
from hydro_model.trajectory import coords_of, save_solution
from initial_guess import (
    build_initial_guess,
    build_robot_ik_initial_guess,
)
from ocp_common import (
    _log_solver_stats,
    add_symmetry_constraints,
    build_collocation_nlp,
    cost_of_transport,
    cycle_energy,
    diagnose_initial_guess,
    extract_solution,
    tangent_to_legacy,
)

# ── OCP parameters ──────────────────────────────────────────────────────
# Per robot, on its spec (hydro_model/robots/<name>.py -> OCPSettings).  They
# were module globals here, which meant retuning them for one robot silently
# retuned the other; amph's values are OCPSettings' defaults.


def build_ocp(robot_name: str = "amph"):
    # ── 1. Robot & dynamics ─────────────────────────────────────────────
    print("Building robot and symbolic dynamics...")
    robot = load_robot(robot_name)
    dyn = SymbolicDynamics(robot)
    nq = robot.nq_reduced
    COORDS = coords_of(robot)

    cfg = robot.spec.ocp
    N, T_INIT, GAIT, TAU_MAX = cfg.n, cfg.t_init, cfg.gait, cfg.tau_max

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("gait_ocp")
    mlflow.start_run(tags={"initial gait": GAIT, "method": "collocation",
                       "robot": robot_name})
    mlflow.log_params({
        "robot": robot_name, "N": N, "T_INIT": T_INIT,
        "T_MIN": cfg.t_min, "T_MAX": cfg.t_max,
        "TAU_MAX": TAU_MAX, "GAIT": GAIT, "V_TARGET": cfg.v_target,
        "D_COLLOC": cfg.d_colloc, "HEADING_TOL": cfg.heading_tol, "F_C": cfg.f_c,
        "ENFORCE_SYMMETRY": cfg.enforce_symmetry,
        "SYMMETRY_PHASE": cfg.symmetry_phase,
        "W_POWER": cfg.w_power, "W_DIST": cfg.w_dist,
        "W_VEL_SMOOTH": cfg.w_vel_smooth, "W_DRIFT": cfg.w_drift,
        "W_JOINT_SMOOTH": cfg.w_joint_smooth,
    })

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
    guess_path = save_solution("task3_guess.npz", T=T_INIT, X=X_guess, U=U_guess,
                               N=N, nq=nq, robot=robot_name, coords=COORDS)
    mlflow.log_artifact(str(guess_path))
    print(f"  Initial guess saved to {guess_path}")

    # F=None: collocation has no single-step integrator to check defects
    # against; the cost-term breakdown is still useful for weight tuning.
    diagnose_initial_guess(
        X_guess, U_guess, nq, N, T_INIT,
        cfg.w_power, cfg.w_dist, cfg.w_vel_smooth, cfg.w_drift, F=None,
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
        t_lo=cfg.t_min, t_hi=cfg.t_max, t_init=T_INIT,
        v_target=cfg.v_target, f_c=cfg.f_c, heading_tol=cfg.heading_tol,
        d_colloc=cfg.d_colloc,
    )
    opti = nlp["opti"]
    X, U, T = nlp["X"], nlp["U"], nlp["T"]
    alpha = nlp["alpha"]
    q_ref_quat = nlp["q_ref_quat"]

    # Objective: shared effort / smoothness / drift terms plus a forward-distance
    # reward (this driver rewards distance directly; the codesign evaluator does
    # not, relying on the speed floor instead).
    dist_cost = -cfg.w_dist * (X[0, -1] - X[0, 0]) / T
    opti.minimize(
        cfg.w_power * nlp["power_cost"]
        + dist_cost
        + cfg.w_vel_smooth * nlp["vel_smooth_cost"]
        + cfg.w_joint_smooth * nlp["joint_smooth_cost"]
        + cfg.w_drift * nlp["drift_cost"]
    )

    # Left–right symmetry, phase free; see ocp_common.add_symmetry_constraints.
    sym_phase = []      # solved phase per pair, reported after the solve
    if cfg.enforce_symmetry:
        sym_phase = add_symmetry_constraints(
            opti, X, X_guess, robot, N, phases=cfg.symmetry_phase)

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
        for (r_leg, _), p in zip(robot.spec.lr_leg_pairs, sym_phase):
            solved = float(src.value(p)) % 1.0    # delta is unbounded; wrap it
            mlflow.log_param(f"phase_solved_{r_leg}", round(solved, 4))
            print(f"  solved phase {r_leg}: {solved:.4f} cycles")
        X_val = tangent_to_legacy(Xt_val, q_ref_quat, robot.model,
                                  nq=robot.nq_reduced, nv=robot.nv_reduced)
        # Tangent, not legacy: Xc is what the objective's quadrature ran on, so
        # it is saved in the coordinates the transcription used.
        Xc_val = src.value(nlp["Xc"])
        # COT on the same quadrature the co-design sweep uses, so a nominal
        # solve can be placed on the sweep's Pareto front.
        forward = float(X_val[0, -1] - X_val[0, 0])
        energy = cycle_energy(U_val, Xc_val[nlp["V_J"], :],
                              nlp["B"], nlp["d"], N, T_val)
        cot = cost_of_transport(energy, robot, forward)
        print(f"  Cycle energy         = {energy:.4f} J")
        print(f"  Cost of transport    = {cot:.4f}")
        mlflow.log_metrics({"energy": energy, "cot": cot})
        extract_solution(X_val, U_val, nq, N, T_val,
                         robot=robot_name, coords=COORDS,
                         Xc_val=Xc_val)

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
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", default="amph", help="registered robot name")
    build_ocp(parser.parse_args().robot)
