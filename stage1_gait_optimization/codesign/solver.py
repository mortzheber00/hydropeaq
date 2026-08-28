#!/usr/bin/env python3
"""
Importable collocation-OCP evaluator for the (gait, T) co-design sweep.

Wraps the shared collocation transcription (``ocp_common.build_collocation_nlp``)
into a pure function ``solve_gait_ocp`` that the outer loop (see
``run_codesign.py``) calls per design point.  Compared with the standalone
``trajopt/run_collocation.py`` driver:

  * No module-level mutable state, no MLflow, no interactive prompts — the
    function is callable from worker processes.
  * The cycle period T does a **narrow free-T refine**: IPOPT may move T within
    ``[t_center - free_T_band, t_center + free_T_band]`` (band = 0 pins it).
  * The objective is a clean effort/energy minimisation (no forward-distance
    reward); the forward-speed floor stays a hard constraint.  Speed and energy
    are returned as metrics so the outer loop owns the speed-vs-efficiency
    tradeoff.
"""

from __future__ import annotations

import sys
from pathlib import Path

STAGE1_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STAGE1_DIR))

import numpy as np
import pinocchio as pin
from hydro_model import SymbolicDynamics, load_robot
from initial_guess import build_initial_guess, build_robot_ik_initial_guess
from ocp_common import build_collocation_nlp, tangent_to_legacy

ROBOT = "amph"   # registered robot name; see hydro_model/robots/


PAPER_GAITS = ("LSPG25", "LSPG33", "TLPG50")
FIRMWARE_GAITS = ("Prototype",)
GAITS = PAPER_GAITS + FIRMWARE_GAITS

# ── OCP parameters (mirror run_collocation.py defaults) ─────────────────────
N = 32           # collocation intervals
D_TARGET = 0.2   # forward distance per nominal cycle [m] -> speed floor
V_TARGET = 0.2   # required average forward speed [m/s]
TAU_MAX = 3.5    # joint torque limit [Nm]
F_C = 20.0       # actuator bandwidth [Hz] — first-order filter cutoff
W_POWER = 2.0       # weight for sum-of-squared per-joint mechanical power (τ·q̇)²
W_VEL_SMOOTH = 20.0  # weight for velocity smoothing
W_DRIFT = 10.0      # weight for drift penalty
HEADING_TOL = 0.05  # max yaw angle at endpoint (radians)
D_COLLOC = 3        # polynomial degree (Radau collocation points)

GRAVITY = 9.81

# Per-process cache of the (expensive) robot + symbolic dynamics build.
_ROBOT_DYN = None


def get_robot_dyn():
    """Build (and cache per process) the robot and symbolic dynamics.

    Caching at module level means each multiprocessing worker pays the build
    cost once, not once per evaluation.
    """
    global _ROBOT_DYN
    if _ROBOT_DYN is None:
        robot = load_robot(ROBOT)
        dyn = SymbolicDynamics(robot)
        _ROBOT_DYN = (robot, dyn)
    return _ROBOT_DYN


def _build_guess(dyn, gait, n, t_center):
    if gait in PAPER_GAITS:
        return build_initial_guess(dyn, gait, n, t_center, TAU_MAX)
    if gait in FIRMWARE_GAITS:
        return build_robot_ik_initial_guess(dyn, n, t_center, TAU_MAX)
    raise ValueError(f"Unknown gait: {gait!r} (expected one of {GAITS})")


def solve_gait_ocp(
    gait: str,
    t_center: float,
    *,
    v_target: float = V_TARGET,
    free_T_band: float = 0.1,
    robot=None,
    dyn=None,
    n: int = N,
    weights: dict | None = None,
    print_level: int = 5,
) -> dict:
    """Solve the collocation OCP for one (gait, T, target-speed) design point.

    Minimises energy/effort subject to an average-speed floor ``v_target``
    (ε-constraint method): the floor binds, so the converged speed ≈ ``v_target``
    and sweeping ``v_target`` in the outer loop traces the speed axis of the
    Pareto front.  The period T is free within
    ``[t_center - free_T_band, t_center + free_T_band]`` (band 0 pins it) and acts
    as the efficiency knob at each target speed.

    Returns a dict with ``feasible``, the converged ``speed`` and
    ``energy``/``cot`` metrics, the solved ``T``, and the legacy-state / control
    trajectories ``X`` / ``U``.
    """
    if robot is None or dyn is None:
        robot, dyn = get_robot_dyn()

    w = {
        "power": W_POWER,
        "vel_smooth": W_VEL_SMOOTH,
        "drift": W_DRIFT,
    }
    if weights:
        w.update(weights)

    X_guess, U_guess = _build_guess(dyn, gait, n, t_center)

    nlp = build_collocation_nlp(
        dyn, robot, X_guess, U_guess, n,
        t_lo=t_center - free_T_band,
        t_hi=t_center + free_T_band,
        t_init=t_center,
        v_target=v_target,
        f_c=F_C,
        heading_tol=HEADING_TOL,
        d_colloc=D_COLLOC,
    )
    opti = nlp["opti"]

    # Effort/energy minimisation; the speed floor lives inside the shared NLP.
    opti.minimize(
        w["power"] * nlp["power_cost"]
        + w["vel_smooth"] * nlp["vel_smooth_cost"]
        + w["drift"] * nlp["drift_cost"]
    )

    opti.solver(
        "ipopt",
        {"expand": False},
        {
            "max_iter": 1200,
            "tol": 1e-4,
            "acceptable_tol": 1e-3,
            "acceptable_iter": 15,
            "print_level": print_level,
            "linear_solver": "ma97",
            "hsllib": "/usr/local/lib/libcoinhsl.so",
            "ma97_order": "metis",
            "mu_strategy": "adaptive",
            "nlp_scaling_method": "gradient-based",
            "ma97_scaling": "mc64",
        },
    )

    try:
        sol = opti.solve()
        feasible = True
        src = sol
    except RuntimeError:
        feasible = False
        src = opti.debug

    return _extract(src, nlp["X"], nlp["Xc"], nlp["U"], nlp["T"], nlp["V_J"],
                    nlp["B"], nlp["d"], nlp["q_ref_quat"], robot, robot.nq, n,
                    gait, t_center, v_target, feasible)


def _extract(src, X, Xc, U, T, V_J, B, d, q_ref_quat, robot, nq, n,
             gait, t_center, v_target, feasible):
    """Read metrics off the (possibly failed) iterate and return them."""
    T_val = float(src.value(T))
    Xt_val = src.value(X)
    U_val = src.value(U)
    X_val = tangent_to_legacy(Xt_val, q_ref_quat, robot.model,
                              nq=robot.nq_reduced, nv=robot.nv_reduced)

    forward = float(X_val[0, -1] - X_val[0, 0])
    speed = forward / T_val

    # Mechanical work over the cycle, ∫Σ_j|τ_j·q̇_j|dt, integrated the same way
    # the objective is: on the collocation points with the Radau weights B.
    # The grid-node sum this replaces was a left-rectangle rule and came out
    # 30-40% low on solved trajectories, so every COT it produced was too.
    #
    # One caveat this does not remove: |·| kinks wherever a joint velocity
    # crosses zero inside an interval, and B is exact only for polynomials.
    # It is high order between sign changes and first order across them; an
    # exact figure would split each interval at the roots of q̇.
    dt = T_val / n
    vc = src.value(Xc)[V_J, :]             # joint velocities at the collocation points
    energy = float(sum(
        B[i] * dt * np.sum(np.abs(U_val[:, k] * vc[:, k * d + i]))
        for k in range(n) for i in range(d)
    ))

    mass = pin.computeTotalMass(robot.model)
    cot = energy / (mass * GRAVITY * abs(forward)) if abs(forward) > 1e-9 else np.inf

    return {
        "gait": gait,
        "t_center": float(t_center),
        "v_target": float(v_target),
        "feasible": feasible,
        "T": T_val,
        "speed": speed,
        "forward_dist": forward,
        "energy": energy,
        "cot": cot,
        "X": X_val,
        "U": U_val,
        "nq": nq,
        "N": n,
    }
