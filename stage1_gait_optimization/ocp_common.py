from __future__ import annotations

import casadi as ca
import mlflow
import numpy as np
import pinocchio as pin
from hydro_model.trajectory import save_solution

# Grid size W_VEL_SMOOTH was tuned at.  vel_smooth_cost is calibrated to keep
# its old magnitude here, so the tuned weight still means what it did; see the
# term's construction in build_collocation_nlp.  Changing this rescales the
# smoothness penalty for every robot and every N, so it is a retune, not a knob.
VEL_SMOOTH_REF_N = 32


def _base_ref(model: pin.Model, q_ref_quat: np.ndarray) -> np.ndarray:
    """Full-size configuration holding only the reference base orientation.

    ``pin.neutral`` rather than zeros so that a tree with continuous joints
    (whose configuration is a ``(cos, sin)`` pair) stays valid.  For a purely
    revolute tree neutral is zeros, so this changes nothing there.  Only the
    free-flyer block is ever read back out.
    """
    q = pin.neutral(model)
    q[3:7] = q_ref_quat
    return q


def legacy_to_tangent(
    X_legacy: np.ndarray, q_ref_quat: np.ndarray, model: pin.Model,
    *, nq: int | None = None, nv: int | None = None,
) -> np.ndarray:
    """Convert (nq+nv, K) state with base quaternion to (2*nv, K) with a
    3-vector base tangent ``phi`` around ``q_ref_quat`` (scalar-last).

    ``nq``/``nv`` default to the model's dimensions.  Pass the robot's *reduced*
    dimensions for a closed-chain robot, whose state is expressed in the
    actuated coordinates rather than in the larger tree.  ``model`` is used
    only for the base SE(3) operations, so it is always the tree model.
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
    #
    # This stays a node sum while the OCP's power_cost integrates on the
    # collocation points: a guess has grid states only, no Xc to integrate
    # over.  So the number below is a first-order estimate of the term the
    # solver will actually minimise — fine for the weight-balance check it
    # exists for, but do not read it as the objective's starting value, and
    # do not compare it across N (the rectangle rule's error shrinks with N,
    # which on its own moves this number).
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
    """Lagrange basis derivative matrix C, endpoint vector D and quadrature
    weights B for Radau collocation.

    tau_root = [0, tau_1, ..., tau_d]  (d+1 points)
    C[i, r] = d/dtau L_i(tau_root[r+1])   (r = 0..d-1, the d Radau points)
    D[i]    = L_i(1)
    B[r]    = int_0^1 L_r(tau) dtau, over the d Radau points; sums to 1 and
              integrates polynomials up to degree 2d-2 exactly.  This is what
              makes an integral cost consistent with the dynamics it is solved
              against — a sum over grid nodes instead is only first order.
    """
    tau_root = np.concatenate([[0.0], np.array(ca.collocation_points(d, "radau"))])
    C, D, B = ca.collocation_coeff(ca.collocation_points(d, "radau"))
    return tau_root, np.array(C), np.array(D).flatten(), np.array(B).flatten()


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
    # Reduced coordinates: equal to the tree dimensions for a serial robot.
    nq, nv = robot.nq_reduced, robot.nv_reduced
    n_act = robot.n_actuated
    nx = 2 * nv
    Q_J = slice(6, 6 + n_act)
    V_B = slice(6 + n_act, 12 + n_act)
    V_J = slice(12 + n_act, nx)

    # Limits come from the spec when it supplies them; otherwise from the URDF,
    # which is what the pipeline always did.  A closed-chain robot must supply
    # them: its tree limits are indexed over joints it does not control, and
    # continuous joints carry no limits at all.
    spec = robot.spec

    def _limit(given, fallback):
        return fallback if given is None else np.asarray(given, dtype=float)

    q_lb = _limit(spec.theta_lower, robot.model.lowerPositionLimit[7:])
    q_ub = _limit(spec.theta_upper, robot.model.upperPositionLimit[7:])
    v_ub = _limit(spec.theta_vel_limit, robot.model.velocityLimit[6:])
    tau_ub = _limit(spec.theta_effort_limit, robot.model.effortLimit[6:])
    v_lb, tau_lb = -v_ub, -tau_ub
    for label, arr in (("position", q_lb), ("velocity", v_ub), ("effort", tau_ub)):
        if len(arr) != n_act:
            raise ValueError(
                f"{spec.name}: {label} limits have length {len(arr)}, expected "
                f"{n_act} — set RobotSpec.theta_* for this robot"
            )

    tau_root, C, D, B = collocation_coefficients(d_colloc)
    d = d_colloc

    # Reference quaternion anchors the tangent representation at the guess's
    # initial base orientation, so phi(t=0) = 0 by construction.
    q_ref_quat = X_guess[3:7, 0].copy()
    f_kin, f_inv_dyn = dyn.build_tangent_dynamics(q_ref_quat)
    Xt_guess = legacy_to_tangent(X_guess, q_ref_quat, robot.model, nq=nq, nv=nv)
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
    #
    # Power is integrated on the collocation points with the Radau weights B,
    # not sampled at the grid nodes.  The node sum this replaces was a
    # left-rectangle rule: first order, and blind to everything between nodes.
    # The optimiser exploited that — in an N=64 solution the penalised value
    # was 15.4 while the same trajectory's interval mean was 61.8, because
    # large torques had been parked at nodes where the sampled velocity
    # happened to be small.  U is piecewise constant, so the integrand is a
    # polynomial and B integrates it to the order of the scheme.
    #
    # T cancels, leaving the same normalisation (a mean, so W_POWER keeps its
    # scale):  (1/T) * sum_k sum_i B_i * dt * f  ==  (1/n) * sum_k sum_i B_i * f.
    power_cost = sum(
        B[i] * ca.sumsqr(U[:, k] * Xc[V_J, k * d + i])
        for k in range(n) for i in range(d)
    ) / n
    drift_cost = sum(X[1, k] ** 2 + (X[2, k] - X[2, 0]) ** 2 for k in range(n + 1)) / (n + 1)

    # Collocation constraints (kinematic + inverse-dynamics split — no M⁻¹)
    a_base = []          # (weight, base acceleration) at every collocation point
    for k in range(n):
        uk_full = ca.vertcat(ca.DM.zeros(6, 1), U[:, k])
        x_all = [X[:, k]] + [Xc[:, k * d + j] for j in range(d)]
        for j in range(1, d + 1):
            xp = sum(C[i, j - 1] * x_all[i] for i in range(d + 1))
            opti.subject_to(dt * f_kin(x_all[j]) == xp[:n_kin])
            a_poly = xp[n_kin:] / dt
            opti.subject_to(f_inv_dyn(x_all[j], a_poly) == uk_full)
            # a_poly is d(v)/dt of the state polynomial; V_B's rows are the
            # first six of the velocity block, so a_poly[:6] is the base.
            a_base.append((B[j - 1], a_poly[:6]))
        x_end = sum(D[i] * x_all[i] for i in range(d + 1))
        opti.subject_to(X[:, k + 1] == x_end)

    # Base-motion smoothness as (1/T)∫‖v̇_B‖²dt, quadrature on the collocation
    # points, replacing sum‖v_{k+1} − v_k‖²/n.  That difference sum is dt²
    # times a mean square acceleration, so its weighted share of the objective
    # fell 1.52 -> 0.29 of the power term across an N = 16..64 ladder: the
    # trade-off being optimised changed with the mesh, which a refinement study
    # cannot tolerate.  The mean square acceleration itself is mesh-independent.
    #
    # VEL_SMOOTH_REF_N restores the old magnitude at one N so W_VEL_SMOOTH keeps
    # the scale it was tuned at; the difference sum equals dt² · mean‖a‖² there,
    # with dt = t_init / VEL_SMOOTH_REF_N.  "Roughly": the old sum measured that
    # mean by finite difference and under-read it exactly as the power node sum
    # did, so expect the term to sit somewhat above its old value even at the
    # reference N.  Away from it the drift with N is gone, which is the point.
    vel_smooth_cost = (t_init / VEL_SMOOTH_REF_N) ** 2 * sum(
        b * ca.sumsqr(a) for b, a in a_base
    ) / n

    # Per-configuration constraints supplied by the robot: equalities (amph
    # pins its side joints to zero) and inequalities (a closed-chain robot
    # must stay assemblable).  Equalities are applied at grid points only,
    # matching where the side-joint pinning has always lived; inequalities go
    # everywhere, because a branch flip mid-interval would corrupt the solve.
    def _pose(theta):
        if spec.pose_constraints is None:
            return [], []
        return spec.pose_constraints(theta)

    # Bounds at grid points on tangent state
    for k in range(n + 1):
        opti.subject_to(opti.bounded(q_lb, X[Q_J, k], q_ub))
        opti.subject_to(opti.bounded(v_lb, X[V_J, k], v_ub))
        opti.subject_to(opti.bounded(-2.0, X[V_B, k], 2.0))
        eqs, ineqs = _pose(X[Q_J, k])
        for expr in eqs:
            opti.subject_to(expr == 0.0)
        for expr in ineqs:
            opti.subject_to(expr >= 0.0)

    # Bounds at collocation points.  Radau puts its last point at tau = 1, so
    # D = [0, ..., 0, 1] and the defect above reads X[:, k+1] == Xc[:, k*d+d-1]:
    # the interval endpoint is held in two variables tied by an equality.
    # Constraining both copies duplicates every row there, and once a duplicated
    # pair goes active its two gradients differ by exactly the equality's own
    # gradient — LICQ fails, so no KKT point exists for IPOPT to certify and it
    # regularises indefinitely instead of converging.  The grid loop above
    # already covers those points (the equality carries the bounds across), so
    # stop one short.  This holds because collocation_coefficients fixes the
    # scheme to Radau; under one whose last point is interior (Legendre, at
    # tau = 0.887 for d = 3) that point is a state no grid node covers, and
    # skipping it would leave it silently unbounded.
    for k in range(n):
        for j in range(d - 1):
            xc_kj = Xc[:, k * d + j]
            opti.subject_to(opti.bounded(q_lb, xc_kj[Q_J], q_ub))
            opti.subject_to(opti.bounded(v_lb, xc_kj[V_J], v_ub))
            opti.subject_to(opti.bounded(-2.0, xc_kj[V_B], 2.0))
            for expr in _pose(xc_kj[Q_J])[1]:
                opti.subject_to(expr >= 0.0)

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
        # Quadrature weights and degree, so a caller measuring energy off the
        # solution integrates it the same way the objective did.
        "B": B, "d": d,
        "Q_J": Q_J, "V_B": V_B, "V_J": V_J,
        "power_cost": power_cost,
        "vel_smooth_cost": vel_smooth_cost,
        "drift_cost": drift_cost,
    }
