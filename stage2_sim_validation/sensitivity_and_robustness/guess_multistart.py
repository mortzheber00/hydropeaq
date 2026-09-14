#!/usr/bin/env python3
"""
Multi-start study: how far apart do different initial guesses land?

Throwaway analysis script — not part of the pipeline.  Everything is held fixed
except the initial guess: same N, same d, same pinned cycle period, same speed
floor, same weights, same hydrodynamic coefficients.  Only the starting point
moves, so any spread in the converged solutions is the optimiser's, not the
model's.

This is the one study where cold-starting is *correct*.  Everywhere else in this
folder the solves are warm-started to keep the sweep on a single solution branch;
here the question is precisely whether the branch depends on where you begin, so
each guess gets a cold solve.

Motivation from the ladders: continuation up from N=16 and down from N=96 — same
NLP, same coefficients, same grid at the end — reached optima 45 deg apart in
joint angle and 33% apart in COT.  Two samples is not a distribution, and it is
currently the weakest-supported number in the analysis.  This turns it into a
measured spread.

The two ladder solutions at the same N are additional samples of the same
question and are folded into the comparison with ``--include``, so nothing
already computed is wasted.

Run:
    python guess_multistart.py
    python guess_multistart.py --gaits LSPG25 TLPG50 --n 32
    python guess_multistart.py --include sweep_results/<tag>_N48_solution.npz
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import mlflow
import numpy as np
import pinocchio as pin

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "stage1_gait_optimization"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hydro_model import SymbolicDynamics, load_robot                 # noqa: E402
from hydro_model.trajectory import coords_of, load_solution          # noqa: E402
from initial_guess import (                                          # noqa: E402
    build_initial_guess,
    build_robot_ik_initial_guess,
    build_theta_sinusoid_guess,
)
from n_sweep_continuation import IPOPT_OPTS                          # noqa: E402
from ocp_common import (                                             # noqa: E402
    _log_solver_stats,
    build_collocation_nlp,
    collocation_coefficients,
    extract_solution,
    tangent_to_legacy,
)

GAITS = ("LSPG25", "LSPG33", "TLPG50", "Prototype", "ThetaSinusoid")
PAPER = ("LSPG25", "LSPG33", "TLPG50")
# ThetaSinusoid needs a robot with a real coordinate map: its builder calls
# cmap.feasibility(), and IdentityMap returns an empty *symbolic* matrix that
# np.min cannot consume (theta_sinusoid.py:91).  So it works for body2 and
# raises for amph.  Selectable with --gaits, out of the default set.
DEFAULT_GAITS = ("LSPG25", "LSPG33", "TLPG50", "Prototype")
OUT_DIR = Path(__file__).parent / "multistart_results"
MLFLOW_TRACKING_URI = "http://localhost:5000"
MLFLOW_EXPERIMENT = "gait_ocp"
GRAVITY = 9.81
PHASE_SAMPLES = 256


def cold_guess(dyn, gait: str, n: int, t_init: float, tau_max: float):
    if gait in PAPER:
        return build_initial_guess(dyn, gait, n, t_init, tau_max)
    if gait == "Prototype":
        return build_robot_ik_initial_guess(dyn, n, t_init, tau_max)
    if gait == "ThetaSinusoid":
        return build_theta_sinusoid_guess(dyn, n, t_init, tau_max)
    raise ValueError(f"unknown gait {gait!r}; choose from {list(GAITS)}")


def cot_of(X, U, Xc, T, robot, B, d) -> float:
    n = U.shape[1]
    dt = T / n
    vc = Xc[slice(12 + robot.n_actuated, 2 * robot.nv_reduced), :]
    energy = float(sum(
        B[i] * dt * np.sum(np.abs(U[:, k] * vc[:, k * d + i]))
        for k in range(n) for i in range(d)
    ))
    forward = float(X[0, -1] - X[0, 0])
    return energy / (pin.computeTotalMass(robot.model) * GRAVITY * abs(forward))


def joint_angles(X, n_act, samples=PHASE_SAMPLES) -> np.ndarray:
    """Joint angles on a common cycle-phase grid, so two N can be compared."""
    n = X.shape[1] - 1
    ph = np.arange(n + 1) / n
    g = np.arange(samples) / samples
    return np.array([np.interp(g, ph, X[7 + j, :]) for j in range(n_act)])


def solve_from(robot, cfg, n, gait, robot_name, tag, B, dyn):
    """One cold solve from ``gait``'s guess, logged as its own MLflow run."""
    nq = robot.nq_reduced
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n{'=' * 70}\n  cold start from {gait}, N={n}\n{'=' * 70}")
    X_guess, U_guess = cold_guess(dyn, gait, n, cfg.t_init, cfg.tau_max)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=f"{tag}_{gait}",
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
            "sweep": "guess_multistart", "sweep_tag": tag, "start_from": gait,
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
            print(f"\n* {gait} FAILED: {e}\n  Extracting best iterate...\n")

        mlflow.log_param("solver_status", "optimal" if ok else "failed")
        mlflow.log_param("T_solved", round(float(src.value(T)), 4))
        _log_solver_stats(opti.stats())

        T_val = float(src.value(T))
        Xc_val = src.value(nlp["Xc"])
        X_val = tangent_to_legacy(src.value(X), nlp["q_ref_quat"], robot.model,
                                  nq=robot.nq_reduced, nv=robot.nv_reduced)
        U_val = src.value(U)
        extract_solution(X_val, U_val, nq, n, T_val, robot=robot_name,
                         coords=coords_of(robot), Xc_val=Xc_val,
                         out_path=str(OUT_DIR / f"{tag}_{gait}.npz"))
        cot = cot_of(X_val, U_val, Xc_val, T_val, robot, B, cfg.d_colloc)
        mlflow.log_metric("cot", cot)
    return dict(label=gait, ok=ok, cot=cot, X=X_val, U=U_val, T=T_val,
                iters=opti.stats().get("iter_count", 0))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot", default="amph", help="registered robot name")
    parser.add_argument("--gaits", nargs="+", default=list(DEFAULT_GAITS),
                        help=f"initial guesses to start from (default: "
                             f"{' '.join(DEFAULT_GAITS)}; ThetaSinusoid also "
                             f"exists but needs a robot with a coordinate map)")
    parser.add_argument("--n", type=int, default=None,
                        help="collocation intervals (default: the robot's "
                             "OCPSettings.n)")
    parser.add_argument("--include", type=Path, nargs="+", default=[],
                        help="existing solution npz to fold into the comparison "
                             "without re-solving — e.g. the two ladder solutions "
                             "at this N, which are samples of the same question")
    parser.add_argument("--tag", default=None,
                        help="MLflow run-name prefix (default: a timestamp)")
    args = parser.parse_args()

    bad = [g for g in args.gaits if g not in GAITS]
    if bad:
        raise SystemExit(f"unknown gait(s) {bad}; choose from {list(GAITS)}")

    robot = load_robot(args.robot)
    cfg = robot.spec.ocp
    n = args.n or cfg.n
    tag = args.tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    _, _, _, B = collocation_coefficients(cfg.d_colloc)
    dyn = SymbolicDynamics(robot)

    print(f"Multi-start {tag}: N={n}, d={cfg.d_colloc}, "
          f"T=[{cfg.t_min}, {cfg.t_max}], v_target={cfg.v_target:.4f}, "
          f"{len(args.gaits)} cold starts")

    results = [solve_from(robot, cfg, n, g, args.robot, tag, B, dyn)
               for g in args.gaits]
    for path in args.include:
        s = load_solution(path)
        if s["N"] != n:
            print(f"  (skipping {path}: N={s['N']}, not {n})")
            continue
        if "Xc" not in s:
            print(f"  (skipping {path}: no Xc, cannot measure COT the same way)")
            continue
        results.append(dict(
            label=path.stem, ok=True, X=s["X"], U=s["U"], T=s["T"], iters=0,
            cot=cot_of(s["X"], s["U"], s["Xc"], s["T"], robot, B, cfg.d_colloc)))

    n_act = robot.n_actuated
    q = {r["label"]: np.degrees(joint_angles(r["X"], n_act)) for r in results}
    labels = [r["label"] for r in results]

    print(f"\n{'=' * 70}")
    print(f"{'start':<28s}{'status':>9s}{'COT':>10s}{'iters':>8s}{'speed':>10s}")
    for r in results:
        v = float(r["X"][0, -1] - r["X"][0, 0]) / r["T"]
        print(f"{r['label']:<28s}{'ok' if r['ok'] else 'FAILED':>9s}"
              f"{r['cot']:>10.4f}{r['iters']:>8.0f}{v:>10.4f}")

    cots = [r["cot"] for r in results if r["ok"]]
    if len(cots) > 1:
        print(f"\nCOT spread over {len(cots)} converged starts: "
              f"{min(cots):.4f} to {max(cots):.4f}  "
              f"({100 * (max(cots) / min(cots) - 1):.1f}% worst-to-best)")

    print("\npairwise RMS joint difference [deg] — the clustering is the result")
    w = max(len(l) for l in labels) + 2
    # Included solutions carry file-stem labels, which are long; give the
    # columns room rather than truncating them into ambiguity.
    cw = max(13, min(20, max(len(l) for l in labels) + 2))
    print(" " * w + "".join(f"{l[:cw - 2]:>{cw}s}" for l in labels))
    for a in labels:
        print(f"{a:<{w}s}" + "".join(
            f"{np.sqrt(((q[a] - q[b]) ** 2).mean()):>{cw}.2f}" for b in labels))
    print(f"\nartifacts in {OUT_DIR}, MLflow sweep_tag={tag}")


if __name__ == "__main__":
    main()
