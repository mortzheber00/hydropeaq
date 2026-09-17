#!/usr/bin/env python3
"""Mesh-refinement study: solve the OCP over a ladder of N with continuation.

Independent solves at different N land in different local optima, so each rung
is warm-started from the previous solution resampled to the new grid. Run the
ladder in both directions: if both agree at each N, the differences reflect
discretisation error. Symmetry is off; each rung is logged as its own MLflow
run (``ladder``, ``ladder_pos``, ``warm_start_from``). Plot the result with
plot_mesh_convergence.py.

Usage:
  python stage2_sim_validation/sensitivity_and_robustness/n_sweep_continuation.py
  python stage2_sim_validation/sensitivity_and_robustness/n_sweep_continuation.py --ladder 64 48 32 24 16
  python stage2_sim_validation/sensitivity_and_robustness/n_sweep_continuation.py --ladder 96 --tag <tag> \\
      --seed sweep_results/<tag>_N64_solution.npz
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import casadi as ca
import mlflow
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "stage1_gait_optimization"))

from hydro_model import SymbolicDynamics, load_robot          # noqa: E402
from hydro_model.trajectory import coords_of, load_solution, save_solution  # noqa: E402
from initial_guess import build_initial_guess, build_robot_ik_initial_guess  # noqa: E402
from ocp_common import (                                      # noqa: E402
    _log_solver_stats,
    build_collocation_nlp,
    diagnose_initial_guess,
    extract_solution,
    tangent_to_legacy,
)

LADDER = (16, 24, 32, 48, 64)
OUT_DIR = Path(__file__).parent / "sweep_results"

MLFLOW_TRACKING_URI = "http://localhost:5000"
MLFLOW_EXPERIMENT = "gait_ocp"

# Same options as trajopt/run_collocation.py (defined inside build_ocp there,
# so they cannot be imported). Keep in sync.
IPOPT_OPTS = {
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
}


def cold_guess(dyn, gait: str, n: int, t_init: float, tau_max: float):
    """Initial guess as built by the collocation driver (first rung only)."""
    if gait in ("LSPG25", "LSPG33", "TLPG50"):
        return build_initial_guess(dyn, gait, n, t_init, tau_max)
    if gait == "Prototype":
        return build_robot_ik_initial_guess(dyn, n, t_init, tau_max)
    raise ValueError(f"unsupported gait for this sweep: {gait!r}")


def resample(X: np.ndarray, U: np.ndarray, n_new: int) -> tuple[np.ndarray, np.ndarray]:
    """Resample a solution onto ``n_new`` intervals as a warm start.

    ``X`` is interpolated linearly in phase without periodic wrap (base x is
    not periodic). ``U`` is piecewise constant, so each new interval takes the
    old interval containing its midpoint.
    """
    n_old = X.shape[1] - 1
    ph_old = np.arange(n_old + 1) / n_old
    ph_new = np.arange(n_new + 1) / n_new
    X_new = np.array([np.interp(ph_new, ph_old, row) for row in X])

    # Renormalise the interpolated base quaternion.
    quat = X_new[3:7, :]
    X_new[3:7, :] = quat / np.linalg.norm(quat, axis=0, keepdims=True)

    mid = (np.arange(n_new) + 0.5) / n_new
    idx = np.clip((mid * n_old).astype(int), 0, n_old - 1)
    return X_new, U[:, idx]


def solve_rung(robot, dyn, cfg, n, X_guess, U_guess, gait, robot_name,
               warm_from, ladder_pos, ladder_tag):
    """Solve one rung and log it like the collocation driver does."""
    nq = robot.nq_reduced
    coords = coords_of(robot)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=f"{ladder_tag}_N{n}",
                          tags={"initial gait": gait, "method": "collocation",
                                "robot": robot_name}):
        mlflow.log_params({
            "robot": robot_name, "N": n, "T_INIT": cfg.t_init,
            "T_MIN": cfg.t_min, "T_MAX": cfg.t_max,
            "TAU_MAX": cfg.tau_max, "GAIT": gait, "V_TARGET": cfg.v_target,
            "D_COLLOC": cfg.d_colloc, "HEADING_TOL": cfg.heading_tol,
            "F_C": cfg.f_c,
            "ENFORCE_SYMMETRY": False, "SYMMETRY_PHASE": None,
            "W_POWER": cfg.w_power, "W_DIST": cfg.w_dist,
            "W_VEL_SMOOTH": cfg.w_vel_smooth, "W_DRIFT": cfg.w_drift,
            "ladder": ladder_tag,
            "ladder_pos": ladder_pos,
            "warm_start_from": warm_from,
        })

        guess_path = save_solution(OUT_DIR / f"{ladder_tag}_N{n}_guess.npz",
                                   T=cfg.t_init, X=X_guess, U=U_guess, N=n,
                                   nq=nq, robot=robot_name, coords=coords)
        mlflow.log_artifact(str(guess_path))
        diagnose_initial_guess(X_guess, U_guess, nq, n, cfg.t_init,
                               cfg.w_power, cfg.w_dist, cfg.w_vel_smooth,
                               cfg.w_drift, F=None)

        nlp = build_collocation_nlp(
            dyn, robot, X_guess, U_guess, n,
            t_lo=cfg.t_min, t_hi=cfg.t_max, t_init=cfg.t_init,
            v_target=cfg.v_target, f_c=cfg.f_c, heading_tol=cfg.heading_tol,
            d_colloc=cfg.d_colloc,
        )
        opti, X, U, T = nlp["opti"], nlp["X"], nlp["U"], nlp["T"]

        # Objective of run_collocation.py without the joint smoothing term
        opti.minimize(
            cfg.w_power * nlp["power_cost"]
            - cfg.w_dist * (X[0, -1] - X[0, 0]) / T
            + cfg.w_vel_smooth * nlp["vel_smooth_cost"]
            + cfg.w_drift * nlp["drift_cost"]
        )
        opti.solver("ipopt", {"expand": False}, IPOPT_OPTS)

        try:
            src = opti.solve()
            ok = True
            print(f"\n* N={n} solved\n")
        except RuntimeError as e:
            src = opti.debug
            ok = False
            print(f"\n* N={n} failed: {e}\n  Extracting best iterate...\n")

        mlflow.log_param("solver_status", "optimal" if ok else "failed")
        mlflow.log_param("T_solved", round(float(src.value(T)), 4))
        mlflow.log_param("bandwidth_alpha", round(float(src.value(nlp["alpha"])), 4))
        _log_solver_stats(opti.stats())

        T_val = float(src.value(T))
        X_val = tangent_to_legacy(src.value(X), nlp["q_ref_quat"], robot.model,
                                  nq=robot.nq_reduced, nv=robot.nv_reduced)
        U_val = src.value(U)
        extract_solution(X_val, U_val, nq, n, T_val, robot=robot_name,
                         coords=coords, Xc_val=src.value(nlp["Xc"]),
                         out_path=str(OUT_DIR / f"{ladder_tag}_N{n}_solution.npz"))
    return X_val, U_val, ok


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot", default="amph", help="registered robot name")
    parser.add_argument("--ladder", type=int, nargs="+", default=list(LADDER),
                        help="collocation-interval counts, in solve order; "
                             "reverse it to run the ladder downward")
    parser.add_argument("--gait", default=None,
                        help="initial gait for the cold start (default: the "
                             "robot's own OCPSettings.gait)")
    parser.add_argument("--tag", default=None,
                        help="MLflow run-name prefix (default: a timestamp)")
    parser.add_argument("--seed", type=Path, default=None,
                        help="solution npz to warm-start the FIRST rung from, "
                             "instead of building a cold guess — use it to "
                             "extend an existing ladder by one rung without "
                             "re-solving the ones below it")
    parser.add_argument("--stop-on-failure", action="store_true",
                        help="abort instead of carrying a failed iterate up the "
                             "ladder as the next warm start")
    args = parser.parse_args()

    robot = load_robot(args.robot)
    dyn = SymbolicDynamics(robot)
    cfg = robot.spec.ocp
    gait = args.gait or cfg.gait
    tag = args.tag or datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"Continuation ladder {args.ladder}  gait={gait}  d={cfg.d_colloc}  "
          f"T=[{cfg.t_min}, {cfg.t_max}]  v_target={cfg.v_target:.4f}  tag={tag}")

    X_prev = U_prev = None
    n_prev = None
    if args.seed is not None:
        seed = load_solution(args.seed)
        X_prev, U_prev, n_prev = seed["X"], seed["U"], seed["N"]
        print(f"seeding the first rung from {args.seed} (N={n_prev})")

    results = []
    for pos, n in enumerate(args.ladder):
        print(f"\n{'=' * 70}\n  rung {pos + 1}/{len(args.ladder)}: N={n}"
              f"{'  (cold start)' if X_prev is None else f'  (warm start from N={n_prev})'}"
              f"\n{'=' * 70}")
        if X_prev is None:
            X_guess, U_guess = cold_guess(dyn, gait, n, cfg.t_init, cfg.tau_max)
            warm_from = "cold"
        else:
            X_guess, U_guess = resample(X_prev, U_prev, n)
            warm_from = f"N{n_prev}"

        X_prev, U_prev, ok = solve_rung(robot, dyn, cfg, n, X_guess, U_guess,
                                        gait, args.robot, warm_from, pos, tag)
        n_prev = n
        results.append((n, ok))
        if not ok and args.stop_on_failure:
            print(f"\nrung N={n} failed; stopping (--stop-on-failure)")
            break

    print(f"\n{'=' * 70}\nladder {tag}: "
          + "  ".join(f"N={n}:{'ok' if ok else 'FAILED'}" for n, ok in results))
    print(f"artifacts in {OUT_DIR}")


if __name__ == "__main__":
    main()
