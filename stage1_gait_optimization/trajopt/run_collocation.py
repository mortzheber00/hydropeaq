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

import casadi as ca
import mlflow
import numpy as np
from hydro_model import SymbolicDynamics, load_robot
from hydro_model.trajectory import coords_of, save_solution
from initial_guess import (
    build_initial_guess,
    build_robot_ik_initial_guess,
    build_theta_sinusoid_guess,
)
from ocp_common import (
    _log_solver_stats,
    build_collocation_nlp,
    diagnose_initial_guess,
    extract_solution,
    tangent_to_legacy,
)

# ── OCP parameters ──────────────────────────────────────────────────────
# Per robot, on its spec (hydro_model/robots/<name>.py -> OCPSettings).  They
# were module globals here, which meant retuning them for one robot silently
# retuned the other; amph's values are OCPSettings' defaults.

# Shape resolution of the symmetry parametrisation below; Nyquist caps it at n/2.
N_HARMONICS = 5


def _fourier_basis(N: int, n_harmonics: int) -> np.ndarray:
    """``(N, 1 + 2H)`` unshifted series basis on the grid.

    Constant rather than a function of the free period: T cancels out of
    ``2*pi*m*t_k/T = 2*pi*m*k/N``.
    """
    k = np.arange(N)
    cols = [np.ones(N)]
    for m in range(1, n_harmonics + 1):
        cols.append(np.cos(2.0 * np.pi * m * k / N))
        cols.append(np.sin(2.0 * np.pi * m * k / N))
    return np.column_stack(cols)


def _detect_mirror_phase(right, left, N):
    """Cycle fraction at which ``left`` already mirrors onto ``right``.

    Both are ``(n_joints, N)`` with the mirror sign folded into ``left``.
    Returns ``(phase, residual)``.  Node resolution is enough for a seed since
    the OCP solves for the phase from there; a large residual means the guess
    is not mirror-symmetric at any phase.
    """
    # np.roll(left, s)[k] is left[k - s], matching _fourier_at's delay of s / N.
    err = [float(np.abs(right - np.roll(left, s, axis=1)).max()) for s in range(N)]
    best = int(np.argmin(err))
    return best / N, err[best]


def _fourier_at(coeffs, k: int, N: int, n_harmonics: int, delay):
    """Series for one joint at node ``k``, delayed by ``delay`` cycles.

    ``delay`` may be a CasADi variable — it enters only through sin/cos, so the
    expression stays smooth in it.
    """
    out = coeffs[0]
    for m in range(1, n_harmonics + 1):
        arg = 2.0 * np.pi * m * (k / N - delay)
        out = out + coeffs[2 * m - 1] * ca.cos(arg) + coeffs[2 * m] * ca.sin(arg)
    return out


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
    })

    # ── 2. Initial guess ────────────────────────────────────────────────
    print(f"Building initial guess from paper trajectory ({GAIT})...")
    if GAIT in ["LSPG25", "LSPG33", "TLPG50"]:
        X_guess, U_guess = build_initial_guess(dyn, GAIT, N, T_INIT, TAU_MAX)
    elif GAIT == "Prototype":
        X_guess, U_guess = build_robot_ik_initial_guess(dyn, N, T_INIT, TAU_MAX)
    elif GAIT == "ThetaSinusoid":
        X_guess, U_guess = build_theta_sinusoid_guess(dyn, N, T_INIT, TAU_MAX)
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
        + cfg.w_drift * nlp["drift_cost"]
    )

    # ── Left–right symmetry, phase free ─────────────────────────────────
    #     theta_left(t)  = series(c)(t)
    #     theta_right(t) = mirror_joint_sign * series(c)(t - delta * T)
    #
    # Harmonics are what make delta a decision variable: shifting a series is an
    # exact rotation of its coefficients, where shifting a collocation
    # trajectory would need non-differentiable interpolation.  One delta per
    # pair suffices — a pair's own shift is absorbed into its coefficients, so
    # the front–hind phase needs no variable.  Positions only; the velocities
    # follow from the kinematic collocation constraint (q̇ = v).
    sym_phase = []      # solved phase per pair, reported after the solve
    if cfg.enforce_symmetry:
        pairs = robot.spec.lr_leg_pairs
        signs = robot.spec.mirror_joint_sign
        if not pairs or signs is None:
            raise ValueError(
                f"{robot.spec.name}: enforce_symmetry needs lr_leg_pairs and "
                f"mirror_joint_sign on the RobotSpec"
            )
        n_per_leg = robot.n_actuated // len(robot.spec.leg_names)
        if len(signs) != n_per_leg:
            raise ValueError(
                f"{robot.spec.name}: mirror_joint_sign has {len(signs)} entries, "
                f"expected {n_per_leg} (one per joint of a leg)"
            )
        sym_joints = robot.spec.symmetry_joints
        if sym_joints is None:
            sym_joints = tuple(range(n_per_leg))
        if not sym_joints or any(not 0 <= j < n_per_leg for j in sym_joints):
            raise ValueError(
                f"{robot.spec.name}: symmetry_joints {sym_joints} out of range for "
                f"{n_per_leg} joints per leg"
            )
        sym_signs = np.array([signs[j] for j in sym_joints])
        # None -> read the phase off the guess; set it only to force a phasing
        # other than the guess's own.
        phases = cfg.symmetry_phase
        if phases is not None:
            if np.isscalar(phases):
                phases = (float(phases),) * len(pairs)
            if len(phases) != len(pairs):
                raise ValueError(
                    f"{robot.spec.name}: symmetry_phase has {len(phases)} entries, "
                    f"expected {len(pairs)} (one per left-right pair) or a scalar"
                )

        basis = _fourier_basis(N, N_HARMONICS)
        leg_index = {leg: i for i, leg in enumerate(robot.spec.leg_names)}
        qj0 = 6           # first joint-position row in tangent state
        qj0_legacy = 7    # ... and in the legacy guess, which carries a quaternion

        worst_fit = 0.0
        for i, (r_leg, l_leg) in enumerate(pairs):
            l0 = qj0 + n_per_leg * leg_index[l_leg]
            r0 = qj0 + n_per_leg * leg_index[r_leg]
            gl0 = qj0_legacy + n_per_leg * leg_index[l_leg]
            gr0 = qj0_legacy + n_per_leg * leg_index[r_leg]

            g_left = X_guess[[gl0 + j for j in sym_joints], :N]
            g_right = X_guess[[gr0 + j for j in sym_joints], :N]

            detected, mirror_err = _detect_mirror_phase(
                g_right, g_left * sym_signs[:, None], N)
            phase0 = detected if phases is None else phases[i]
            print(f"  {r_leg}/{l_leg}: guess mirrors at phase {detected:.4f} "
                  f"(residual {np.degrees(mirror_err):.2f} deg), seeding {phase0:.4f}")

            coeffs = opti.variable(len(sym_joints), 1 + 2 * N_HARMONICS)
            # Unbounded: delta is periodic, so a bound would be a false edge.
            delta = opti.variable()
            opti.set_initial(delta, phase0)
            sym_phase.append(delta)

            target = g_left.T                             # seed coeffs from the guess
            fit, *_ = np.linalg.lstsq(basis, target, rcond=None)
            opti.set_initial(coeffs, fit.T)
            worst_fit = max(worst_fit, float(np.abs(basis @ fit - target).max()))

            for k in range(N):
                for row, j in enumerate(sym_joints):
                    c = coeffs[row, :].T
                    opti.subject_to(
                        X[l0 + j, k] == _fourier_at(c, k, N, N_HARMONICS, 0.0))
                    opti.subject_to(
                        X[r0 + j, k] == signs[j] * _fourier_at(c, k, N, N_HARMONICS, delta))

        # Large fit error -> the stroke needs more than N_HARMONICS to describe.
        print(f"  symmetry: {len(pairs)} pair(s), joints {list(sym_joints)}, "
              f"{N_HARMONICS} harmonics, phase free")
        print(f"  worst harmonic fit error on a left leg of the guess: "
              f"{np.degrees(worst_fit):.2f} deg")

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
        extract_solution(X_val, U_val, nq, N, T_val,
                         robot=robot_name, coords=COORDS,
                         Xc_val=src.value(nlp["Xc"]))

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
