import casadi as ca
import mlflow
import numpy as np
import pinocchio as pin


def legacy_to_tangent(
    X_legacy: np.ndarray, q_ref_quat: np.ndarray, model: pin.Model
) -> np.ndarray:
    """Convert (nq+nv, K) state with base quaternion to (2*nv, K) with a
    3-vector base tangent ``phi`` around ``q_ref_quat`` (scalar-last)."""
    nq, nv = model.nq, model.nv
    n_act = nv - 6
    q_ref_full = np.zeros(nq); q_ref_full[3:7] = q_ref_quat
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
    X_tan: np.ndarray, q_ref_quat: np.ndarray, model: pin.Model
) -> np.ndarray:
    """Inverse of ``legacy_to_tangent``."""
    nq, nv = model.nq, model.nv
    n_act = nv - 6
    q_ref_full = np.zeros(nq); q_ref_full[3:7] = q_ref_quat
    X_leg = np.zeros((nq + nv, X_tan.shape[1]))
    for k in range(X_tan.shape[1]):
        dv = np.zeros(nv); dv[3:6] = X_tan[3:6, k]
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
    """Sanity-check the initial guess and report cost-term balance.

    Prints (and logs to MLflow):
      - per-step shooting defects ‖F(x_k, [0;u_k]) − x_{k+1}‖ if an RK4
        integrator F(x, tau_full) is supplied.  Large defects ⇒ guess is
        not dynamically consistent ⇒ IPOPT burns iterations on feasibility.
      - the four cost terms exactly as built in the OCP, both unweighted
        and weighted, so the user can rebalance W_* before solving.
    """
    print("\n  -- Initial-guess diagnostics --")

    # ── Shooting defects ─────────────────────────────────────────────────
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

    # ── Cost-term breakdown ──────────────────────────────────────────────
    # Per-joint mechanical power (τ_j · q̇_j); legacy state stores joint
    # velocities at [nq+6 : nq+6+n_act]. Match U_guess width (N).
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

    # Flag terms whose weighted magnitude is more than 5x off from power.
    # Skip when W_POWER == 0 (feasibility stage) — reference is meaningless.
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
    """Return a dynamically consistent state trajectory by forward integration.

    Starts from X_guess[:,0], applies U_guess[:,k] at each step via F, and
    returns X_rolled with shooting defects ≈ 0 (machine epsilon).  This trades
    dynamics defects for periodicity residuals, which IPOPT handles better
    because they are global rather than per-step equality constraints.

    Prints periodicity residuals of the rolled-out trajectory so you can
    judge how well the guess gait closes the cycle.
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


def extract_solution(X_val, U_val, nq: int, N: int, T_FIXED: float) -> None:
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

    np.savez("task3_solution.npz", T=T_FIXED, X=X_val, U=U_val, N=N, nq=nq)
    mlflow.log_metrics({
        "forward_dist": float(X_val[0, -1] - X_val[0, 0]),
        "forward_vel":  float((X_val[0, -1] - X_val[0, 0]) / T_FIXED),
        "torque_rms":   float(np.sqrt(np.mean(U_val**2))),
        "torque_max":   float(np.max(np.abs(U_val))),
        "cost":         float(np.sum(U_val**2) / N),
        "base_z_min":   float(X_val[2, :].min()),
        "base_z_max":   float(X_val[2, :].max()),
    })
    mlflow.log_artifact("task3_solution.npz")
    print("\n  Solution saved to task3_solution.npz")


def collocation_coefficients(d: int):
    """Lagrange basis derivative matrix C and endpoint vector D for Radau collocation.

    tau_root = [0, tau_1, ..., tau_d]  (d+1 points)
    C[i, r] = d/dtau L_i(tau_root[r+1])   (r = 0..d-1, the d Radau points)
    D[i]    = L_i(1)
    """
    tau_root = np.concatenate([[0.0], np.array(ca.collocation_points(d, "radau"))])
    C, D, _ = ca.collocation_coeff(ca.collocation_points(d, "radau"))
    return tau_root, np.array(C), np.array(D).flatten()


def build_collocation_nlp(
    dyn, robot, X_guess, U_guess, n, *,
    t_lo, t_hi, t_init, v_target, f_c, heading_tol, d_colloc=3,
):
    """Build the collocation transcription shared by the standalone driver
    (``trajopt/run_collocation.py``) and the co-design evaluator
    (``codesign/solver.py``).

    Creates the ``Opti`` problem with variables (grid states ``X``, collocation
    states ``Xc``, controls ``U``, free period ``T``), and adds every constraint
    that is identical between the two formulations: the kinematic + inverse-
    dynamics collocation defects, state/control bounds, the first-order
    torque-rate (bandwidth) filter, periodicity, the phi anchor, the average-speed
    floor (``distance >= v_target * T``), and the heading bound.  It also warm-
    starts from the guess and returns the common cost terms.

    What is *not* added here — because it genuinely differs between the two —
    is the objective (the driver adds a forward-distance reward; the evaluator
    does not), the optional left–right symmetry constraints, and the solver
    configuration.  The caller assembles those from the returned handles.
    """
    nq, nv = robot.nq, robot.nv
    n_act = robot.n_actuated
    nx = 2 * nv
    Q_J = slice(6, 6 + n_act)
    V_B = slice(6 + n_act, 12 + n_act)
    V_J = slice(12 + n_act, nx)

    q_lb = robot.model.lowerPositionLimit[7:]
    q_ub = robot.model.upperPositionLimit[7:]
    v_lb = -robot.model.velocityLimit[6:]
    v_ub = robot.model.velocityLimit[6:]
    tau_lb = -robot.model.effortLimit[6:]
    tau_ub = robot.model.effortLimit[6:]

    tau_root, C, D = collocation_coefficients(d_colloc)
    d = d_colloc

    # Reference quaternion anchors the tangent representation at the guess's
    # initial base orientation, so phi(t=0) = 0 by construction.
    q_ref_quat = X_guess[3:7, 0].copy()
    f_kin, f_inv_dyn = dyn.build_tangent_dynamics(q_ref_quat)
    Xt_guess = legacy_to_tangent(X_guess, q_ref_quat, robot.model)
    n_kin = 6 + n_act

    opti = ca.Opti()
    X = opti.variable(nx, n + 1)        # tangent states at grid points
    Xc = opti.variable(nx, n * d)       # tangent states at collocation points
    U = opti.variable(n_act, n)         # controls (piecewise constant)

    # Free cycle period, bounded to a (possibly degenerate) band.
    T = opti.variable()
    dt = T / n
    opti.subject_to(opti.bounded(t_lo, T, t_hi))
    opti.set_initial(T, t_init)

    # Common cost terms (the objective itself is assembled by the caller).
    power_cost = sum(ca.sumsqr(U[:, k] * X[V_J, k]) for k in range(n)) / n
    vel_smooth_cost = sum(ca.sumsqr(X[V_B, k + 1] - X[V_B, k]) for k in range(n)) / n
    drift_cost = sum(X[1, k] ** 2 + (X[2, k] - X[2, 0]) ** 2 for k in range(n + 1)) / (n + 1)

    # Collocation constraints (kinematic + inverse-dynamics split — no M⁻¹)
    for k in range(n):
        uk_full = ca.vertcat(ca.DM.zeros(6, 1), U[:, k])
        x_all = [X[:, k]] + [Xc[:, k * d + j] for j in range(d)]
        for j in range(1, d + 1):
            xp = sum(C[i, j - 1] * x_all[i] for i in range(d + 1))
            opti.subject_to(dt * f_kin(x_all[j]) == xp[:n_kin])
            a_poly = xp[n_kin:] / dt
            opti.subject_to(f_inv_dyn(x_all[j], a_poly) == uk_full)
        x_end = sum(D[i] * x_all[i] for i in range(d + 1))
        opti.subject_to(X[:, k + 1] == x_end)

    # Bounds at grid points on tangent state
    for k in range(n + 1):
        opti.subject_to(opti.bounded(q_lb, X[Q_J, k], q_ub))
        opti.subject_to(opti.bounded(v_lb, X[V_J, k], v_ub))
        opti.subject_to(opti.bounded(-2.0, X[V_B, k], 2.0))
        for side_idx in range(6, 6 + n_act, 3):
            opti.subject_to(X[side_idx, k] == 0.0)

    # Bounds at collocation points
    for k in range(n):
        for j in range(d):
            xc_kj = Xc[:, k * d + j]
            opti.subject_to(opti.bounded(q_lb, xc_kj[Q_J], q_ub))
            opti.subject_to(opti.bounded(v_lb, xc_kj[V_J], v_ub))
            opti.subject_to(opti.bounded(-2.0, xc_kj[V_B], 2.0))

    # Bounds on controls at grid points
    for k in range(n):
        opti.subject_to(opti.bounded(tau_lb, U[:, k], tau_ub))

    # Torque-rate (bandwidth) constraints — first-order filter, symbolic in T.
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

    # Periodicity + phi anchor + average-speed floor + heading bound.
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

    # Warm start
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
        "Q_J": Q_J, "V_B": V_B, "V_J": V_J,
        "power_cost": power_cost,
        "vel_smooth_cost": vel_smooth_cost,
        "drift_cost": drift_cost,
    }
