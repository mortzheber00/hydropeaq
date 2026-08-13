"""Sinusoid-search initial guess (Cui et al. 2026, imitation-learning stage).

Adapts the front end of Cui et al.'s Imitation Learning stage: instead of
encoding one fixed gait shape (cf. ``paper.py`` / ``firmware.py``), this
parameterises a single representative leg as joint-space sinusoids,
Latin-Hypercube-samples the parameter space, and keeps the candidate that
scores best under the robot's own hydro model.

What transfers from the paper: the sinusoidal parameterisation (their Eq. 4)
and the LHS-over-parameters search ranked by forward thrust with a lift
penalty.  What does NOT transfer: the imitation-learning / Transformer / RL
machinery — that seeds a control *policy*, whereas here we only need a warm
start for the trajectory OCP.

Fitness is a *model-based proxy*, not a hardware force measurement: each
candidate's leg sinusoid is mapped to all four legs (diagonal coordination,
matching ``firmware.py``), the base DOF are simulated with prescribed joint
kinematics, and the candidate is scored by

    score = mean forward velocity  −  lift_weight · heave-oscillation amplitude

where heave oscillation (std of base z over a cycle) is the embodied
stand-in for the paper's lift-non-cancellation cost.

The single representative leg is centred on the trim joint posture, so the
search only ranges over amplitudes and phase — mirroring the paper's
reduction of the four-legged problem to one limb.

Returns ``(X_guess (nx, N+1), U_guess (n_act, N))``, the same interface as
``build_initial_guess`` and ``build_robot_ik_initial_guess``.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np
from hydro_model import SymbolicDynamics

from .base_sim import simulate_base_kinematics


def _lhs(n_samples: int, n_dim: int, rng: np.random.Generator) -> np.ndarray:
    """Latin Hypercube samples in the unit cube, shape ``(n_samples, n_dim)``.

    Each column is partitioned into ``n_samples`` equal strata with one
    sample drawn per stratum, then the strata are shuffled independently per
    dimension — the standard LHS construction used in the paper.
    """
    cut = np.linspace(0.0, 1.0, n_samples + 1)
    pts = cut[:n_samples, None] + rng.random((n_samples, n_dim)) / n_samples
    for j in range(n_dim):
        rng.shuffle(pts[:, j])
    return pts


def _leg_sinusoid_kinematics(
    a_thigh: float,
    a_calf: float,
    phi: float,
    diag_offset: float,
    q_trim: np.ndarray,
    N: int,
    n_act: int,
    T_FIXED: float,
    n_cycles: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Joint position/velocity/acceleration for the sinusoid gait.

    The representative leg's thigh and calf oscillate about their trim
    angles, the calf lagging the thigh by ``phi`` (cf. paper Eq. 4).  The
    same motion drives all four legs as two diagonal pairs — (FL, HR) at
    phase 0 and (FR, HL) lagging by ``diag_offset`` cycles — matching the
    diagonal coordination in ``firmware.py``.  Side joints are held at trim.

    Returns ``(q_joints (n_act, N+1), v_joints (n_act, N+1),
    a_joints (n_act, N))``; velocities are analytic, accelerations the
    forward difference of the analytic velocity (the form the base simulator
    integrates).
    """
    phase_offsets = [0.0, diag_offset, diag_offset, 0.0]  # FL, FR, HL, HR
    w = 2.0 * np.pi * n_cycles
    dtl_dt = 1.0 / T_FIXED  # cycle-fraction time -> real time

    q_joints = np.zeros((n_act, N + 1))
    v_joints = np.zeros((n_act, N + 1))

    for k in range(N + 1):
        t_norm = k / N
        for i, off in enumerate(phase_offsets):
            t_leg = (t_norm - off) % 1.0
            b = i * 3
            thigh0 = q_trim[7 + b + 1]
            calf0 = q_trim[7 + b + 2]

            q_joints[b, k] = q_trim[7 + b]  # side joint held at trim
            q_joints[b + 1, k] = thigh0 + a_thigh * np.sin(w * t_leg)
            q_joints[b + 2, k] = calf0 + a_calf * np.sin(w * t_leg + phi)

            v_joints[b + 1, k] = a_thigh * w * np.cos(w * t_leg) * dtl_dt
            v_joints[b + 2, k] = a_calf * w * np.cos(w * t_leg + phi) * dtl_dt

    dt_val = T_FIXED / N
    a_joints = (v_joints[:, 1:] - v_joints[:, :-1]) / dt_val
    return q_joints, v_joints, a_joints


def _score_candidate(
    dyn: SymbolicDynamics,
    a_thigh: float,
    a_calf: float,
    phi: float,
    diag_offset: float,
    q_trim: np.ndarray,
    N: int,
    n_act: int,
    T_FIXED: float,
    n_cycles: int,
    lift_weight: float,
    score_cycles: int,
) -> float:
    """Forward-thrust-minus-lift score for one sinusoid candidate.

    Simulates the base DOF under the candidate's prescribed joint kinematics
    and returns ``mean(vx) - lift_weight * std(z)`` over the (converged)
    cycle.  Returns ``-inf`` if the simulation diverges.
    """
    q_j, v_j, a_j = _leg_sinusoid_kinematics(
        a_thigh, a_calf, phi, diag_offset, q_trim, N, n_act, T_FIXED, n_cycles
    )
    with contextlib.redirect_stdout(io.StringIO()):
        q_cycle, v_cycle = simulate_base_kinematics(
            dyn, q_j, v_j, a_j, q_trim, T_FIXED / N, n_cycles=score_cycles
        )
    if not np.all(np.isfinite(q_cycle)) or not np.all(np.isfinite(v_cycle)):
        return -np.inf
    mean_vx = float(np.mean(v_cycle[0]))
    heave = float(np.std(q_cycle[2]))
    return mean_vx - lift_weight * heave


def search_sinusoid_params(
    dyn: SymbolicDynamics,
    N: int,
    T_FIXED: float,
    *,
    n_samples: int = 100,
    seed: int = 0,
    lift_weight: float = 0.5,
    amp_thigh_range: tuple[float, float] = (0.10, 0.55),
    amp_calf_range: tuple[float, float] = (0.10, 0.55),
    phase_range: tuple[float, float] = (0.0, np.pi),
    diag_offset_range: tuple[float, float] = (0.0, 0.5),
    n_cycles: int = 1,
    score_cycles: int = 8,
) -> tuple[dict, float]:
    """Latin-Hypercube-search the sinusoid gait parameters.

    Samples ``n_samples`` candidates, scores each by simulated forward thrust
    penalised by heave oscillation (see module docstring), and returns the
    best-scoring parameter set.  No OCP trajectory is built — feed the
    returned dict to ``build_sinusoid_guess`` to construct ``(X, U)``.

    Returns
    -------
    params : dict
        ``{"a_thigh", "a_calf", "phi", "diag_offset", "n_cycles"}`` — the
        winning sinusoid parameters (``build_sinusoid_guess`` consumes this).
    score : float
        The winning candidate's thrust-minus-lift score.

    Parameters
    ----------
    n_samples : int
        Number of LHS candidates evaluated.  Each evaluation runs the base
        simulator for up to ``score_cycles`` cycles, so wall-clock cost grows
        with ``n_samples * score_cycles``.
    seed : int
        RNG seed for the Latin Hypercube sampling.
    lift_weight : float
        Weight on heave-oscillation amplitude (std of base z) in the score.
        ``0`` ranks by pure forward thrust; larger values bias toward the
        paper's low-lift demonstration subset.
    amp_thigh_range / amp_calf_range : (float, float)
        Min/max sinusoid amplitude for the thigh / calf joint [rad], about
        the trim posture.
    phase_range : (float, float)
        Min/max calf-vs-thigh phase lag ``phi`` [rad] (paper Eq. 4 phase).
    diag_offset_range : (float, float)
        Min/max phase offset (cycle fraction) of the (FR, HL) diagonal pair
        relative to (FL, HR).
    n_cycles : int
        Integer paddling cycles contained in ``T_FIXED`` (keeps the gait
        periodic over the OCP horizon).
    score_cycles : int
        Base-simulator cycles used when ranking candidates.
    """
    n_act = dyn.robot.n_actuated
    q_trim = dyn.find_trim_state()
    rng = np.random.default_rng(seed)

    lo = np.array(
        [amp_thigh_range[0], amp_calf_range[0], phase_range[0], diag_offset_range[0]]
    )
    hi = np.array(
        [amp_thigh_range[1], amp_calf_range[1], phase_range[1], diag_offset_range[1]]
    )
    samples = lo + _lhs(n_samples, 4, rng) * (hi - lo)

    print(f"  Searching {n_samples} sinusoid candidates ({score_cycles} cycles each)...")
    best_score = -np.inf
    best = None
    for idx, (a_thigh, a_calf, phi, diag) in enumerate(samples):
        score = _score_candidate(
            dyn, a_thigh, a_calf, phi, diag, q_trim,
            N, n_act, T_FIXED, n_cycles, lift_weight, score_cycles,
        )
        if score > best_score:
            best_score = score
            best = (a_thigh, a_calf, phi, diag)
            print(
                f"    [{idx + 1:3d}/{n_samples}] new best score={score:.4f}  "
                f"A_thigh={a_thigh:.3f} A_calf={a_calf:.3f} "
                f"phi={phi:.3f} diag={diag:.3f}"
            )

    if best is None:
        raise RuntimeError("Sinusoid search found no finite-scoring candidate.")
    a_thigh, a_calf, phi, diag = best
    print(f"  Best score {best_score:.4f}.")

    params = {
        "a_thigh": float(a_thigh),
        "a_calf": float(a_calf),
        "phi": float(phi),
        "diag_offset": float(diag),
        "n_cycles": int(n_cycles),
    }
    return params, float(best_score)


def build_sinusoid_guess(
    dyn: SymbolicDynamics,
    params: dict,
    N: int,
    T_FIXED: float,
    TAU_MAX: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (X_guess, U_guess) from a fixed sinusoid parameter set.

    ``params`` is a dict as returned by ``search_sinusoid_params``:
    ``{"a_thigh", "a_calf", "phi", "diag_offset", "n_cycles"}``.  This is the
    entry point for OCP scripts to reuse a previously searched gait without
    re-running the search.
    """
    robot = dyn.robot
    nq = robot.nq
    nx = nq + robot.nv
    n_act = robot.n_actuated
    dt_val = T_FIXED / N

    q_trim = dyn.find_trim_state()

    # ── Step 1: joint kinematics from the sinusoid parameters ────────────
    q_joints, v_joints, a_joints = _leg_sinusoid_kinematics(
        params["a_thigh"], params["a_calf"], params["phi"], params["diag_offset"],
        q_trim, N, n_act, T_FIXED, params.get("n_cycles", 1),
    )

    # ── Step 2: simulate base DOF with prescribed joints ─────────────────
    print("  Simulating base DOF (sinusoid gait)...")
    q_base_traj, v_base_traj = simulate_base_kinematics(
        dyn, q_joints, v_joints, a_joints, q_trim, dt_val
    )

    # ── Step 3: assemble full state trajectory ───────────────────────────
    X_guess = np.zeros((nx, N + 1))
    for k in range(N + 1):
        q_k = np.concatenate([q_base_traj[:, k], q_joints[:, k]])
        v_k = np.concatenate([v_base_traj[:, k], v_joints[:, k]])
        X_guess[:, k] = np.concatenate([q_k, v_k])

    # ── Step 4: torque guess via inverse dynamics ────────────────────────
    U_guess = np.zeros((n_act, N))
    for k in range(N):
        a_k = (X_guess[nq:, k + 1] - X_guess[nq:, k]) / dt_val
        tau_id = dyn.eval_inverse_dynamics(X_guess[:nq, k], X_guess[nq:, k], a_k)
        U_guess[:, k] = np.clip(tau_id[6:], -TAU_MAX, TAU_MAX)

    return X_guess, U_guess


def build_search_initial_guess(
    dyn: SymbolicDynamics,
    N: int,
    T_FIXED: float,
    TAU_MAX: float,
    **search_kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Search the sinusoid gait family and build the best as an OCP warm start.

    Convenience wrapper: ``search_sinusoid_params`` then ``build_sinusoid_guess``.
    Accepts the same keyword arguments as ``search_sinusoid_params``.
    """
    params, _ = search_sinusoid_params(dyn, N, T_FIXED, **search_kwargs)
    return build_sinusoid_guess(dyn, params, N, T_FIXED, TAU_MAX)
