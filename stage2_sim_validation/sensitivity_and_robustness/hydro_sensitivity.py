#!/usr/bin/env python3
"""Sensitivity of the optimal gait to the hydrodynamic coefficients.

Scales each coefficient by +-25 % (one at a time), re-solves the OCP and
reports the change in COT and joint motion. All solves are warm-started from
the same nominal solution so they stay on one solution branch; cold starts
would mostly measure jumps between local optima. Main effects only, no
interactions. Plot the result with plot_hydro_sensitivity.py.

Usage:
  python stage2_sim_validation/sensitivity_and_robustness/hydro_sensitivity.py --seed sweep_results/<tag>_N64_solution.npz
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import casadi as ca
import mlflow
import numpy as np
import pinocchio as pin

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "stage1_gait_optimization"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hydro_model import SymbolicDynamics, load_robot, hydro_params   # noqa: E402
from hydro_model.trajectory import coords_of, load_solution, save_solution  # noqa: E402
from n_sweep_continuation import IPOPT_OPTS                          # noqa: E402
from ocp_common import (                                             # noqa: E402
    _log_solver_stats,
    build_collocation_nlp,
    collocation_coefficients,
    extract_solution,
    tangent_to_legacy,
)

# Perturbed coefficients (SymbolicDynamics kwargs). Cd_lin_* keep their own
# defaults, so scaling Cd_t/Cd_a does not change the linear damping.
COEFFS = ("Cd_t", "Cd_a", "Ca_t", "Ca_a")
NOMINAL = {
    "Cd_t": hydro_params.CD_T,
    "Cd_a": hydro_params.CD_A,
    "Ca_t": hydro_params.CA_T,
    "Ca_a": hydro_params.CA_A,
}
PERTURBATIONS = (-0.25, +0.25)

OUT_DIR = Path(__file__).parent / "hydro_results"
MLFLOW_TRACKING_URI = "http://localhost:5000"
MLFLOW_EXPERIMENT = "gait_ocp"
GRAVITY = 9.81


def coefficients(coef: str | None, factor: float) -> dict:
    """Nominal coefficients with ``coef`` scaled by ``1 + factor``."""
    vals = dict(NOMINAL)
    if coef is not None:
        vals[coef] = NOMINAL[coef] * (1.0 + factor)
    return vals


def cot_of(X, U, Xc, T, robot, B, d) -> float:
    """Cost of transport with energy integrated on the collocation points."""
    n = U.shape[1]
    dt = T / n
    vc = Xc[slice(12 + robot.n_actuated, 2 * robot.nv_reduced), :]
    energy = float(sum(
        B[i] * dt * np.sum(np.abs(U[:, k] * vc[:, k * d + i]))
        for k in range(n) for i in range(d)
    ))
    forward = float(X[0, -1] - X[0, 0])
    mass = pin.computeTotalMass(robot.model)
    return energy / (mass * GRAVITY * abs(forward))


def solve_case(robot, cfg, n, X_guess, U_guess, gait, robot_name,
               coef, factor, tag, B):
    """Solve one (possibly perturbed) case warm-started and log it to MLflow."""
    vals = coefficients(coef, factor)
    label = "nominal" if coef is None else f"{coef}{factor:+.0%}".replace("%", "pct")
    nq = robot.nq_reduced
    coords = coords_of(robot)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}\n  {label}: " +
          "  ".join(f"{k}={v:.4f}" for k, v in vals.items()) + f"\n{'=' * 70}")

    # Coefficients are baked into the expressions, so rebuild per case.
    dyn = SymbolicDynamics(robot, **vals)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=f"{tag}_{label}",
                          tags={"initial gait": gait, "method": "collocation",
                                "robot": robot_name}):
        mlflow.log_params({
            "robot": robot_name, "N": n, "T_INIT": cfg.t_init,
            "T_MIN": cfg.t_min, "T_MAX": cfg.t_max, "TAU_MAX": cfg.tau_max,
            "GAIT": gait, "V_TARGET": cfg.v_target, "D_COLLOC": cfg.d_colloc,
            "HEADING_TOL": cfg.heading_tol, "F_C": cfg.f_c,
            "ENFORCE_SYMMETRY": False, "SYMMETRY_PHASE": None,
            "W_POWER": cfg.w_power, "W_DIST": cfg.w_dist,
            "W_VEL_SMOOTH": cfg.w_vel_smooth, "W_DRIFT": cfg.w_drift,
            "sweep": "hydro_sensitivity",
            "sweep_tag": tag,
            "perturbed": "none" if coef is None else coef,
            "perturbation": 0.0 if coef is None else factor,
            **{k: round(v, 6) for k, v in vals.items()},
        })

        nlp = build_collocation_nlp(
            dyn, robot, X_guess, U_guess, n,
            t_lo=cfg.t_min, t_hi=cfg.t_max, t_init=cfg.t_init,
            v_target=cfg.v_target, f_c=cfg.f_c, heading_tol=cfg.heading_tol,
            d_colloc=cfg.d_colloc,
        )
        opti, X, U, T = nlp["opti"], nlp["X"], nlp["U"], nlp["T"]
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
        except RuntimeError as e:
            src = opti.debug
            ok = False
            print(f"\n* {label} FAILED: {e}\n  Extracting best iterate...\n")

        mlflow.log_param("solver_status", "optimal" if ok else "failed")
        mlflow.log_param("T_solved", round(float(src.value(T)), 4))
        _log_solver_stats(opti.stats())

        T_val = float(src.value(T))
        Xc_val = src.value(nlp["Xc"])
        X_val = tangent_to_legacy(src.value(X), nlp["q_ref_quat"], robot.model,
                                  nq=robot.nq_reduced, nv=robot.nv_reduced)
        U_val = src.value(U)
        extract_solution(X_val, U_val, nq, n, T_val, robot=robot_name,
                         coords=coords, Xc_val=Xc_val,
                         out_path=str(OUT_DIR / f"{tag}_{label}.npz"))
        cot = cot_of(X_val, U_val, Xc_val, T_val, robot, B, cfg.d_colloc)
        mlflow.log_metric("cot", cot)
    return dict(label=label, coef=coef, factor=factor, ok=ok, cot=cot,
                X=X_val, U=U_val, vals=vals)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot", default="amph", help="registered robot name")
    parser.add_argument("--seed", type=Path, required=True,
                        help="nominal solution npz; every case warm-starts from "
                             "it and takes its N")
    parser.add_argument("--gait", default=None,
                        help="gait label for the MLflow tag (default: the "
                             "robot's own OCPSettings.gait)")
    parser.add_argument("--tag", default=None,
                        help="MLflow run-name prefix (default: a timestamp)")
    parser.add_argument("--coeffs", nargs="+", default=list(COEFFS),
                        help=f"coefficients to perturb (default: {' '.join(COEFFS)})")
    args = parser.parse_args()

    bad = [c for c in args.coeffs if c not in NOMINAL]
    if bad:
        raise SystemExit(f"unknown coefficient(s) {bad}; choose from {list(NOMINAL)}")

    robot = load_robot(args.robot)
    cfg = robot.spec.ocp
    gait = args.gait or cfg.gait
    tag = args.tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    _, _, _, B = collocation_coefficients(cfg.d_colloc)

    seed = load_solution(args.seed)
    X_guess, U_guess, n = seed["X"], seed["U"], seed["N"]
    print(f"Hydro sensitivity {tag}: N={n} from {args.seed}, "
          f"{len(args.coeffs)} coefficients x {len(PERTURBATIONS)} perturbations "
          f"+ nominal")
    print("nominal coefficients: " +
          "  ".join(f"{k}={v:.4f}" for k, v in NOMINAL.items()))

    cases = [(None, 0.0)] + [(c, f) for c in args.coeffs for f in PERTURBATIONS]
    results = []
    for coef, factor in cases:
        results.append(solve_case(robot, cfg, n, X_guess, U_guess, gait,
                                  args.robot, coef, factor, tag, B))

    # Compare against the nominal re-solve, not the seed (which may differ in setup).
    base = results[0]
    q0 = np.degrees(base["X"][7:7 + robot.n_actuated, :])
    print(f"\n{'=' * 70}")
    print(f"{'case':<16s}{'status':>9s}{'COT':>10s}{'dCOT':>9s}"
          f"{'|dq| [deg]':>12s}{'tau_max':>9s}")
    for r in results:
        q = np.degrees(r["X"][7:7 + robot.n_actuated, :])
        dq = float(np.sqrt(((q - q0) ** 2).mean()))
        print(f"{r['label']:<16s}{'ok' if r['ok'] else 'FAILED':>9s}"
              f"{r['cot']:>10.4f}{100 * (r['cot'] / base['cot'] - 1):>+8.2f}%"
              f"{dq:>12.3f}{np.abs(r['U']).max():>9.4f}")
    n_bad = sum(not r["ok"] for r in results)
    if n_bad:
        print(f"\n{n_bad} case(s) did not converge — a perturbation that makes "
              f"the speed floor unreachable is a result, not a gap.")
    print(f"\nartifacts in {OUT_DIR}, MLflow sweep_tag={tag}")


if __name__ == "__main__":
    main()
