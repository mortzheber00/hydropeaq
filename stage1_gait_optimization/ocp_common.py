"""Collocation OCP building blocks shared by the trajectory-optimisation drivers."""
from __future__ import annotations

import casadi as ca
import mlflow
import numpy as np
import pinocchio as pin
from hydro_model.trajectory import save_solution

# Grid size at which W_VEL_SMOOTH was tuned. Changing it rescales the smoothness
# penalties for every robot and N, so treat it as a retune, not a knob.
VEL_SMOOTH_REF_N = 32


def _base_ref(model: pin.Model, q_ref_quat: np.ndarray) -> np.ndarray:
    """Neutral configuration with the reference base orientation.

    Uses ``pin.neutral`` so continuous joints get a valid ``(cos, sin)`` pair.
    """
    q = pin.neutral(model)
    q[3:7] = q_ref_quat
    return q


def legacy_to_tangent(
    X_legacy: np.ndarray, q_ref_quat: np.ndarray, model: pin.Model,
    *, nq: int | None = None, nv: int | None = None,
) -> np.ndarray:
    """Convert ``(nq+nv, K)`` quaternion states to ``(2*nv, K)`` tangent states.

    The base orientation becomes a 3-vector ``phi`` around ``q_ref_quat``
    (scalar-last). For closed-chain robots pass the reduced ``nq``/``nv``;
    ``model`` is always the tree model and is only used for the base.
    """
    nq = model.nq if nq is None else nq
    nv = model.nv if nv is None else nv
    n_act = nv - 6
    q_ref_full = _base_ref(model, q_ref_quat)
    X_tan = np.zeros((2 * nv, X_legacy.shape[1]))
    for k in range(X_legacy.shape[1]):
        q_k = q_ref_full.copy(); q_k[3:7] = X_legacy[3:7, k]
        phi = pin.difference(model, q_ref_full, q_k)[3:6]
        X_tan[0:3, k]            = X_legacy[0:3, k]
        X_tan[3:6, k]            = phi
        X_tan[6 : 6 + n_act, k]  = X_legacy[7:nq, k]
        X_tan[6 + n_act :, k]    = X_legacy[nq:, k]
    return X_tan


def tangent_to_legacy(
    X_tan: np.ndarray, q_ref_quat: np.ndarray, model: pin.Model,
    *, nq: int | None = None, nv: int | None = None,
) -> np.ndarray:
    """Inverse of ``legacy_to_tangent``."""
    nq = model.nq if nq is None else nq
    nv = model.nv if nv is None else nv
    n_act = nv - 6
    q_ref_full = _base_ref(model, q_ref_quat)
    X_leg = np.zeros((nq + nv, X_tan.shape[1]))
    for k in range(X_tan.shape[1]):
        dv = np.zeros(model.nv); dv[3:6] = X_tan[3:6, k]
        q_int = pin.integrate(model, q_ref_full, dv)
        X_leg[0:3, k]   = X_tan[0:3, k]
        X_leg[3:7, k]   = q_int[3:7]
        X_leg[7:nq, k]  = X_tan[6 : 6 + n_act, k]
        X_leg[nq:, k]   = X_tan[6 + n_act :, k]
    return X_leg


def diagnose_initial_guess(
    X_guess: np.ndarray,
    U_guess: np.ndarray,
    nq: int,
    N: int,
    T_FIXED: float,
    W_POWER: float,
    W_DIST: float,
    W_VEL_SMOOTH: float,
    W_DRIFT: float,
    F=None,
) -> None:
    """Print and log shooting defects and the cost-term balance of the guess.

    Defects are only computed if an integrator ``F(x, tau_full)`` is given.
    The weighted cost terms help rebalance ``W_*`` before solving.
    """
    print("\n  -- Initial-guess diagnostics --")

    # --- Shooting defects ---
    if F is not None:
        defects = np.zeros(N)
        for k in range(N):
            tau_full = np.concatenate([np.zeros(6), U_guess[:, k]])
            x_next = np.array(F(X_guess[:, k], tau_full)).flatten()
            defects[k] = np.linalg.norm(x_next - X_guess[:, k + 1])
        print(
            f"    shooting defects: max = {defects.max():.3e}, "
            f"mean = {defects.mean():.3e}, "
            f"worst step = {int(defects.argmax())}/{N}"
        )
        if defects.max() > 1e-1:
            print(
                "    (large — call rollout_guess() to replace X_guess with a "
                "dynamically consistent trajectory before solving)"
            )
        mlflow.log_metrics({
            "guess_defect_max":  float(defects.max()),
            "guess_defect_mean": float(defects.mean()),
        })

    # --- Cost-term breakdown ---
    # The guess has no collocation states, so power is a node sum here: good
    # enough to balance weights, but not the objective's actual starting value.
    n_act = U_guess.shape[0]
    joint_vels = X_guess[nq + 6 : nq + 6 + n_act, :N]
    power_cost = float(np.sum((U_guess * joint_vels) ** 2)) / N
    forward = float(X_guess[0, -1] - X_guess[0, 0])
    dist_cost_raw = -forward / T_FIXED                       # negated: reward
    dv = X_guess[nq : nq + 6, 1:] - X_guess[nq : nq + 6, :-1]
    vel_smooth = float(np.sum(dv ** 2)) / N
    drift_y = float(np.sum(X_guess[1, :] ** 2))
    drift_z = float(np.sum((X_guess[2, :] - X_guess[2, 0]) ** 2))
    drift = (drift_y + drift_z) / (N + 1)

    terms = [
        ("power",     power_cost,     W_POWER),
        ("dist",      dist_cost_raw,  W_DIST),
        ("vel_smooth", vel_smooth,    W_VEL_SMOOTH),
        ("drift",     drift,          W_DRIFT),
    ]
    print(f"    {'term':<11s}{'unweighted':>14s}{'W':>10s}{'weighted':>14s}")
    for name, raw, w in terms:
        print(f"    {name:<11s}{raw:>14.4e}{w:>10.2f}{w * raw:>14.4e}")

    # Flag terms more than 5x off from power (no reference in a feasibility stage).
    if W_POWER > 0:
        ref = abs(terms[0][2] * terms[0][1]) + 1e-30
        for name, raw, w in terms[1:]:
            ratio = abs(w * raw) / ref
            if ratio > 5.0 or ratio < 0.05:
                print(
                    f"    (warn) '{name}' weighted contribution is {ratio:.1f}x "
                    f"power — consider rebalancing W_{name.upper()}"
                )

    mlflow.log_metrics({
        "guess_cost_power":       power_cost,
        "guess_cost_dist":        dist_cost_raw,
        "guess_cost_vel_smooth":  vel_smooth,
        "guess_cost_drift":       drift,
    })
    print()


def rollout_guess(
    X_guess: np.ndarray,
    U_guess: np.ndarray,
    N: int,
    nq: int,
    F,
) -> np.ndarray:
    """Forward-integrate ``U_guess`` from ``X_guess[:, 0]`` for a dynamically consistent guess.

    Trades per-step defects for periodicity residuals, which IPOPT handles
    better. The residuals are printed.
    """
    X_rolled = np.zeros_like(X_guess)
    X_rolled[:, 0] = X_guess[:, 0]
    for k in range(N):
        tau_full = np.concatenate([np.zeros(6), U_guess[:, k]])
        x_next = np.array(F(X_rolled[:, k], tau_full)).flatten()
        qn = np.linalg.norm(x_next[3:7])
        if qn > 1e-8:
            x_next[3:7] /= qn
        X_rolled[:, k + 1] = x_next

    x0, xN = X_rolled[:, 0], X_rolled[:, N]
    djoints = np.linalg.norm(xN[7:nq] - x0[7:nq])
    dv      = np.linalg.norm(xN[nq:]  - x0[nq:])
    print("    periodicity residuals of rolled-out guess:")
    print(f"      Δq_y    = {xN[1] - x0[1]:.3e}")
    print(f"      Δq_z    = {xN[2] - x0[2]:.3e}")
    print(f"      Δquat   = {np.linalg.norm(xN[3:7] - x0[3:7]):.3e}")
    print(f"      Δjoints = {djoints:.3e}")
    print(f"      Δv      = {dv:.3e}")
    if djoints > 0.5 or dv > 1.0:
        print(
            "      (large Δjoints/Δv — the guess gait is far from periodic "
            "under dynamics; IPOPT must close the cycle)"
        )
    mlflow.log_metrics({
        "rollout_periodic_dq_y":    float(xN[1] - x0[1]),
        "rollout_periodic_dq_z":    float(xN[2] - x0[2]),
        "rollout_periodic_djoints": float(djoints),
        "rollout_periodic_dv":      float(dv),
    })
    return X_rolled


def _log_solver_stats(stats: dict) -> None:
    mlflow.log_metrics({
        "cpu_time_s":  stats.get("t_proc_total", 0.0),
        "wall_time_s": stats.get("t_wall_total", 0.0),
        "iterations":  float(stats.get("iter_count", 0)),
    })
    iters = stats.get("iterations", {})
    for step, (obj, inf_pr, inf_du) in enumerate(zip(
        iters.get("obj", []),
        iters.get("inf_pr", []),
        iters.get("inf_du", []),
    )):
        mlflow.log_metrics({"convergence_obj": obj, "inf_pr": inf_pr, "inf_du": inf_du}, step=step)


def extract_solution(X_val, U_val, nq: int, N: int, T_FIXED: float,
                     robot: str = "amph", coords: str = "tree",
                     out_path: str = "task3_solution.npz", Xc_val=None) -> None:
    print(f"  Cycle period T       = {T_FIXED:.4f} s")
    print(f"  Forward distance     = {X_val[0, -1] - X_val[0, 0]:.4f} m")
    print(f"  Average forward vel  = {(X_val[0, -1] - X_val[0, 0]) / T_FIXED:.4f} m/s")
    print(f"  Base z range         = [{X_val[2, :].min():.4f}, {X_val[2, :].max():.4f}] m")
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

    save_solution(out_path, T=T_FIXED, X=X_val, U=U_val, N=N, nq=nq,
                  robot=robot, coords=coords, Xc=Xc_val)
    mlflow.log_metrics({
        "forward_dist": float(X_val[0, -1] - X_val[0, 0]),
        "forward_vel":  float((X_val[0, -1] - X_val[0, 0]) / T_FIXED),
        "torque_rms":   float(np.sqrt(np.mean(U_val**2))),
        "torque_max":   float(np.max(np.abs(U_val))),
        "cost":         float(np.sum(U_val**2) / N),
        "base_z_min":   float(X_val[2, :].min()),
        "base_z_max":   float(X_val[2, :].max()),
    })
    mlflow.log_artifact(out_path)
    print(f"\n  Solution saved to {out_path}")


def collocation_coefficients(d: int):
    """Radau collocation coefficients ``(tau_root, C, D, B)``.

    tau_root = [0, tau_1, ..., tau_d]
    C[i, r]  = dL_i/dtau at tau_root[r+1]
    D[i]     = L_i(1)
    B[r]     = quadrature weight of the r-th Radau point
    """
    tau_root = np.concatenate([[0.0], np.array(ca.collocation_points(d, "radau"))])
    C, D, B = ca.collocation_coeff(ca.collocation_points(d, "radau"))
    return tau_root, np.array(C), np.array(D).flatten(), np.array(B).flatten()


def limits_for(robot):
    """Actuated-joint limits ``(q_lb, q_ub, v_ub, tau_ub)``.

    Taken from the robot spec where set, otherwise from the URDF. Closed-chain
    robots must set them in the spec. Velocity and effort limits are symmetric.
    Also used by plot_limit_activity.py so figures show the limits the solver used.
    """
    spec = robot.spec

    def _limit(given, fallback):
        return fallback if given is None else np.asarray(given, dtype=float)

    q_lb = _limit(spec.theta_lower, robot.model.lowerPositionLimit[7:])
    q_ub = _limit(spec.theta_upper, robot.model.upperPositionLimit[7:])
    v_ub = _limit(spec.theta_vel_limit, robot.model.velocityLimit[6:])
    tau_ub = _limit(spec.theta_effort_limit, robot.model.effortLimit[6:])
    for label, arr in (("position", q_lb), ("velocity", v_ub), ("effort", tau_ub)):
        if len(arr) != robot.n_actuated:
            raise ValueError(
                f"{spec.name}: {label} limits have length {len(arr)}, expected "
                f"{robot.n_actuated} — set RobotSpec.theta_* for this robot"
            )
    return q_lb, q_ub, v_ub, tau_ub


def build_collocation_nlp(
    dyn, robot, X_guess, U_guess, n, *,
    t_lo, t_hi, t_init, v_target, f_c, heading_tol, d_colloc=3,
):
    """Radau collocation NLP shared by run_collocation.py and codesign/solver.py.

    Adds the variables (``X``, ``Xc``, ``U``, free period ``T``), the dynamics
    defects, bounds, torque-rate filter, periodicity, average-speed floor and
    heading bound, and warm-starts from the guess. The caller adds the
    objective, optional symmetry constraints and solver options using the
    returned handles and cost terms.
    """
    # Reduced coordinates: equal to the tree dimensions for a serial robot.
    nq, nv = robot.nq_reduced, robot.nv_reduced
    n_act = robot.n_actuated
    nx = 2 * nv
    Q_J = slice(6, 6 + n_act)
    V_B = slice(6 + n_act, 12 + n_act)
    V_J = slice(12 + n_act, nx)

    q_lb, q_ub, v_ub, tau_ub = limits_for(robot)
    v_lb, tau_lb = -v_ub, -tau_ub

    tau_root, C, D, B = collocation_coefficients(d_colloc)
    d = d_colloc

    # Anchor the tangent representation at the guess's initial orientation.
    q_ref_quat = X_guess[3:7, 0].copy()
    f_kin, f_inv_dyn = dyn.build_tangent_dynamics(q_ref_quat)
    Xt_guess = legacy_to_tangent(X_guess, q_ref_quat, robot.model, nq=nq, nv=nv)
    n_kin = 6 + n_act

    opti = ca.Opti()
    X = opti.variable(nx, n + 1)        # tangent states at grid points
    Xc = opti.variable(nx, n * d)       # tangent states at collocation points
    U = opti.variable(n_act, n)         # controls (piecewise constant)

    # Free cycle period; t_lo == t_hi fixes it.
    T = opti.variable()
    dt = T / n
    opti.subject_to(opti.bounded(t_lo, T, t_hi))
    opti.set_initial(T, t_init)

    # Mean power, integrated on the collocation points. A node sum lets the
    # optimiser hide large torques at nodes where the velocity happens to be small.
    # (1/T) * sum B_i * dt * f  ==  (1/n) * sum B_i * f
    power_cost = sum(
        B[i] * ca.sumsqr(U[:, k] * Xc[V_J, k * d + i])
        for k in range(n) for i in range(d)
    ) / n
    drift_cost = sum(X[1, k] ** 2 + (X[2, k] - X[2, 0]) ** 2 for k in range(n + 1)) / (n + 1)

    # Collocation defects, split into kinematics and inverse dynamics (no M^-1).
    a_base = []          # (weight, base acceleration) per collocation point
    a_joint = []         # (weight, joint acceleration) per collocation point
    for k in range(n):
        uk_full = ca.vertcat(ca.DM.zeros(6, 1), U[:, k])
        x_all = [X[:, k]] + [Xc[:, k * d + j] for j in range(d)]
        for j in range(1, d + 1):
            xp = sum(C[i, j - 1] * x_all[i] for i in range(d + 1))
            opti.subject_to(dt * f_kin(x_all[j]) == xp[:n_kin])
            a_poly = xp[n_kin:] / dt
            opti.subject_to(f_inv_dyn(x_all[j], a_poly) == uk_full)
            a_base.append((B[j - 1], a_poly[:6]))
            a_joint.append((B[j - 1], a_poly[6:]))
        x_end = sum(D[i] * x_all[i] for i in range(d + 1))
        opti.subject_to(X[:, k + 1] == x_end)

    # Mean squared base acceleration. It is mesh-independent, unlike a sum of
    # velocity differences. The (t_init / VEL_SMOOTH_REF_N)^2 factor keeps the
    # magnitude W_VEL_SMOOTH was tuned for.
    vel_smooth_cost = (t_init / VEL_SMOOTH_REF_N) ** 2 * sum(
        b * ca.sumsqr(a) for b, a in a_base
    ) / n

    # Same measure for the joints. Power alone does not regularise joint motion
    # at low torque, which leaves chattering joint trajectories.
    joint_smooth_cost = (t_init / VEL_SMOOTH_REF_N) ** 2 * sum(
        b * ca.sumsqr(a) for b, a in a_joint
    ) / n

    # Robot-specific pose constraints (e.g. amph's pinned side joints, closed-chain
    # assemblability). Equalities apply at grid points only; inequalities also at
    # collocation points, since a branch flip mid-interval corrupts the solve.
    def _pose(theta):
        if robot.spec.pose_constraints is None:
            return [], []
        return robot.spec.pose_constraints(theta)

    # State bounds at grid points
    for k in range(n + 1):
        opti.subject_to(opti.bounded(q_lb, X[Q_J, k], q_ub))
        opti.subject_to(opti.bounded(v_lb, X[V_J, k], v_ub))
        opti.subject_to(opti.bounded(-2.0, X[V_B, k], 2.0))
        eqs, ineqs = _pose(X[Q_J, k])
        for expr in eqs:
            opti.subject_to(expr == 0.0)
        for expr in ineqs:
            opti.subject_to(expr >= 0.0)

    # State bounds at collocation points. The last Radau point coincides with the
    # next grid point, which is already bounded; bounding it twice violates LICQ
    # and IPOPT fails to converge. Only valid for Radau (last point at tau = 1).
    for k in range(n):
        for j in range(d - 1):
            xc_kj = Xc[:, k * d + j]
            opti.subject_to(opti.bounded(q_lb, xc_kj[Q_J], q_ub))
            opti.subject_to(opti.bounded(v_lb, xc_kj[V_J], v_ub))
            opti.subject_to(opti.bounded(-2.0, xc_kj[V_B], 2.0))
            for expr in _pose(xc_kj[Q_J])[1]:
                opti.subject_to(expr >= 0.0)

    # Control bounds
    for k in range(n):
        opti.subject_to(opti.bounded(tau_lb, U[:, k], tau_ub))

    # Torque rate limited by a first-order filter with cutoff f_c (cyclic).
    alpha = 2 * np.pi * dt * f_c / (2 * np.pi * dt * f_c + 1)
    for k in range(1, n):
        opti.subject_to(opti.bounded(
            (1 - alpha) * U[:, k - 1] + alpha * tau_lb,
            U[:, k],
            (1 - alpha) * U[:, k - 1] + alpha * tau_ub,
        ))
    opti.subject_to(opti.bounded(
        (1 - alpha) * U[:, n - 1] + alpha * tau_lb,
        U[:, 0],
        (1 - alpha) * U[:, n - 1] + alpha * tau_ub,
    ))

    # Periodicity, start pose anchor, average-speed floor and heading bound
    x0, xN = X[:, 0], X[:, n]
    opti.subject_to(xN[Q_J] == x0[Q_J])
    opti.subject_to(xN[V_J] == x0[V_J])
    opti.subject_to(xN[V_B] == x0[V_B])
    opti.subject_to(xN[1] == x0[1])
    opti.subject_to(xN[2] == x0[2])
    opti.subject_to(xN[3:6] == x0[3:6])
    opti.subject_to(X[0, 0] == 0.0)
    opti.subject_to(X[1, 0] == 0.0)
    opti.subject_to(X[3:6, 0] == 0.0)
    opti.subject_to(xN[0] - x0[0] >= v_target * T)
    opti.subject_to(opti.bounded(-heading_tol, xN[5], heading_tol))

    # Warm start; collocation states are linearly interpolated.
    for k in range(n + 1):
        opti.set_initial(X[:, k], Xt_guess[:, k])
    for k in range(n):
        for j in range(d):
            tau_j = tau_root[j + 1]
            x_interp = (1 - tau_j) * Xt_guess[:, k] + tau_j * Xt_guess[:, k + 1]
            opti.set_initial(Xc[:, k * d + j], x_interp)
        opti.set_initial(U[:, k], U_guess[:, k])

    return {
        "opti": opti, "X": X, "Xc": Xc, "U": U, "T": T,
        "alpha": alpha, "q_ref_quat": q_ref_quat,
        # For integrating quantities the same way as the objective
        "B": B, "d": d,
        "Q_J": Q_J, "V_B": V_B, "V_J": V_J,
        "power_cost": power_cost,
        "vel_smooth_cost": vel_smooth_cost,
        "joint_smooth_cost": joint_smooth_cost,
        "drift_cost": drift_cost,
    }


# --- Cycle energy and cost of transport ---
GRAVITY = 9.81


def cycle_energy(U_val, vc, B, d: int, n: int, T_val: float) -> float:
    """Absolute mechanical work over the cycle, ∫Σ_j|τ_j·q̇_j|dt.

    Integrated on the collocation points like the objective; ``vc`` is
    ``Xc[V_J, :]``. Only first-order accurate across velocity sign changes.
    """
    dt = T_val / n
    return float(sum(
        B[i] * dt * np.sum(np.abs(U_val[:, k] * vc[:, k * d + i]))
        for k in range(n) for i in range(d)
    ))


def cost_of_transport(energy: float, robot, forward: float) -> float:
    """Dimensionless COT, ``E / (m g |d|)``; infinite for a stalled cycle."""
    if abs(forward) <= 1e-9:
        return np.inf
    return energy / (pin.computeTotalMass(robot.model) * GRAVITY * abs(forward))


# --- Left-right symmetry with free phase ---
N_HARMONICS = 8  # must stay <= N/2


def _fourier_basis(N: int, n_harmonics: int) -> np.ndarray:
    """``(N, 1 + 2H)`` Fourier basis on the grid (independent of T)."""
    k = np.arange(N)
    cols = [np.ones(N)]
    for m in range(1, n_harmonics + 1):
        cols.append(np.cos(2.0 * np.pi * m * k / N))
        cols.append(np.sin(2.0 * np.pi * m * k / N))
    return np.column_stack(cols)


def _detect_mirror_phase(right, left, N):
    """Node-resolution phase at which ``left`` best matches ``right``; ``(phase, residual)``.

    Both are ``(n_joints, N)``, with the mirror sign already applied to ``left``.
    """
    # np.roll(left, s)[k] == left[k - s], i.e. a delay of s / N as in _fourier_at.
    err = [float(np.abs(right - np.roll(left, s, axis=1)).max()) for s in range(N)]
    best = int(np.argmin(err))
    return best / N, err[best]


def _fourier_at(coeffs, k: int, N: int, n_harmonics: int, delay):
    """Fourier series at node ``k`` delayed by ``delay`` cycles (may be symbolic)."""
    out = coeffs[0]
    for m in range(1, n_harmonics + 1):
        arg = 2.0 * np.pi * m * (k / N - delay)
        out = out + coeffs[2 * m - 1] * ca.cos(arg) + coeffs[2 * m] * ca.sin(arg)
    return out


def add_symmetry_constraints(opti, X, X_guess, robot, N, phases=None, verbose=True):
    """Constrain each left-right leg pair to one shape with a free phase offset.

        theta_left(t)  = series(c)(t)
        theta_right(t) = mirror_joint_sign * series(c)(t - delta * T)

    A Fourier series makes the shift ``delta`` differentiable. Only positions
    are constrained; velocities follow from the kinematics.

    ``phases`` seeds ``delta`` in cycles (scalar, per pair, or None to detect it
    from the guess). Returns the ``delta`` variables in ``lr_leg_pairs`` order.
    """
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
    qj0 = 6           # first joint row in the tangent state
    qj0_legacy = 7    # first joint row in the quaternion guess

    sym_phase = []
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
        if verbose:
            print(f"  {r_leg}/{l_leg}: guess mirrors at phase {detected:.4f} "
                  f"(residual {np.degrees(mirror_err):.2f} deg), seeding {phase0:.4f}")

        coeffs = opti.variable(len(sym_joints), 1 + 2 * N_HARMONICS)
        delta = opti.variable()  # periodic, so left unbounded
        opti.set_initial(delta, phase0)
        sym_phase.append(delta)

        target = g_left.T                             # seed coefficients from the guess
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

    if verbose:
        # A large fit error means N_HARMONICS is too low for the stroke.
        print(f"  symmetry: {len(pairs)} pair(s), joints {list(sym_joints)}, "
              f"{N_HARMONICS} harmonics, phase free")
        print(f"  worst harmonic fit error on a left leg of the guess: "
              f"{np.degrees(worst_fit):.2f} deg")
    return sym_phase
