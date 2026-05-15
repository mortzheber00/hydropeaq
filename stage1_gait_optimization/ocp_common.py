import casadi as ca
import mlflow
import numpy as np


def diagnose_initial_guess(
    X_guess: np.ndarray,
    U_guess: np.ndarray,
    nq: int,
    N: int,
    T_FIXED: float,
    W_TORQUE: float,
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
    torque_cost = float(np.sum(U_guess ** 2)) / N
    forward = float(X_guess[0, -1] - X_guess[0, 0])
    dist_cost_raw = -forward / T_FIXED                       # negated: reward
    dv = X_guess[nq : nq + 6, 1:] - X_guess[nq : nq + 6, :-1]
    vel_smooth = float(np.sum(dv ** 2)) / N
    drift_y = float(np.sum(X_guess[1, :] ** 2))
    drift_z = float(np.sum((X_guess[2, :] - X_guess[2, 0]) ** 2))
    drift = (drift_y + drift_z) / (N + 1)

    terms = [
        ("torque",    torque_cost,    W_TORQUE),
        ("dist",      dist_cost_raw,  W_DIST),
        ("vel_smooth", vel_smooth,    W_VEL_SMOOTH),
        ("drift",     drift,          W_DRIFT),
    ]
    print(f"    {'term':<11s}{'unweighted':>14s}{'W':>10s}{'weighted':>14s}")
    for name, raw, w in terms:
        print(f"    {name:<11s}{raw:>14.4e}{w:>10.2f}{w * raw:>14.4e}")

    # Flag terms whose weighted magnitude is more than 5x off from torque.
    # Skip when W_TORQUE == 0 (feasibility stage) — reference is meaningless.
    if W_TORQUE > 0:
        ref = abs(terms[0][2] * terms[0][1]) + 1e-30
        for name, raw, w in terms[1:]:
            ratio = abs(w * raw) / ref
            if ratio > 5.0 or ratio < 0.05:
                print(
                    f"    (warn) '{name}' weighted contribution is {ratio:.1f}x "
                    f"torque — consider rebalancing W_{name.upper()}"
                )

    mlflow.log_metrics({
        "guess_cost_torque":      torque_cost,
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


def extract_solution(sol, X, U, nq: int, N: int, T_FIXED: float) -> None:
    X_val = sol.value(X)
    U_val = sol.value(U)

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
