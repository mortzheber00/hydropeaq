#!/usr/bin/env python3
"""Single-point OCP evaluator for the co-design sweep (``solve_gait_ocp``).

Solves the same NLP as ``trajopt/run_collocation.py``, except that
  - it has no MLflow logging or prompts, so it can run in worker processes,
  - T is only free within ``t_center +- free_T_band``,
  - the objective has no distance reward.
Keep everything else in sync with the standalone driver.
"""

from __future__ import annotations

import sys
from pathlib import Path

STAGE1_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STAGE1_DIR))

import numpy as np
from hydro_model import SymbolicDynamics, load_robot
from hydro_model.robots import get_spec
from initial_guess import build_initial_guess, build_robot_ik_initial_guess
from ocp_common import (
    add_symmetry_constraints,
    build_collocation_nlp,
    cost_of_transport,
    cycle_energy,
    tangent_to_legacy,
)

ROBOT = "amph"   # registered robot name; see hydro_model/robots/


PAPER_GAITS = ("LSPG25", "LSPG33", "TLPG50")
FIRMWARE_GAITS = ("Prototype",)
GAITS = PAPER_GAITS + FIRMWARE_GAITS

# Same OCP settings as run_collocation.py (get_spec does not build the model).
CFG = get_spec(ROBOT).ocp

_ROBOT_DYN = None  # per-process cache


def get_robot_dyn():
    """Robot and symbolic dynamics, built once per process."""
    global _ROBOT_DYN
    if _ROBOT_DYN is None:
        robot = load_robot(ROBOT)
        dyn = SymbolicDynamics(robot)
        _ROBOT_DYN = (robot, dyn)
    return _ROBOT_DYN


def _build_guess(dyn, gait, n, t_center):
    if gait in PAPER_GAITS:
        return build_initial_guess(dyn, gait, n, t_center, CFG.tau_max)
    if gait in FIRMWARE_GAITS:
        return build_robot_ik_initial_guess(dyn, n, t_center, CFG.tau_max)
    raise ValueError(f"Unknown gait: {gait!r} (expected one of {GAITS})")


def solve_gait_ocp(
    gait: str,
    t_center: float,
    *,
    v_target: float = CFG.v_target,
    free_T_band: float = 0.1,
    robot=None,
    dyn=None,
    n: int = CFG.n,
    weights: dict | None = None,
    print_level: int = 5,
    warm_start: np.ndarray | None = None,
) -> dict:
    """Solve the OCP for one (gait, T centre, target speed) point.

    Minimises effort subject to ``speed >= v_target`` (epsilon-constraint), with
    T free within ``t_center +- free_T_band`` (0 fixes it). ``warm_start`` is
    the primal vector of a previous solve with the same gait, T centre and n.

    Returns a dict with ``feasible``, ``speed``, ``energy``, ``cot``, ``T``,
    ``X``/``U`` (quaternion states), solver stats and ``warm_start`` (None if
    the solve failed).
    """
    if robot is None or dyn is None:
        robot, dyn = get_robot_dyn()

    w = {
        "power": CFG.w_power,
        "vel_smooth": CFG.w_vel_smooth,
        "joint_smooth": CFG.w_joint_smooth,
        "drift": CFG.w_drift,
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
        f_c=CFG.f_c,
        heading_tol=CFG.heading_tol,
        d_colloc=CFG.d_colloc,
    )
    opti = nlp["opti"]

    opti.minimize(
        w["power"] * nlp["power_cost"]
        + w["vel_smooth"] * nlp["vel_smooth_cost"]
        + w["joint_smooth"] * nlp["joint_smooth_cost"]
        + w["drift"] * nlp["drift_cost"]
    )

    if CFG.enforce_symmetry:
        add_symmetry_constraints(opti, nlp["X"], X_guess, robot, n,
                                 phases=CFG.symmetry_phase,
                                 verbose=print_level > 0)

    # Primal warm start only; IPOPT's warm_start_init_point (needed for duals)
    # turned out slower than cold solves.
    if warm_start is not None:
        if warm_start.shape[0] != opti.nx:
            raise ValueError(
                f"warm start has {warm_start.shape[0]} primals, this NLP has "
                f"{opti.nx} — a chain must hold gait, t_center and n fixed"
            )
        opti.set_initial(opti.x, warm_start)

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

    stats = opti.stats()
    row = _extract(src, nlp["X"], nlp["Xc"], nlp["U"], nlp["T"], nlp["V_J"],
                   nlp["B"], nlp["d"], nlp["q_ref_quat"], robot, robot.nq, n,
                   gait, t_center, v_target, feasible)
    row["iterations"] = int(stats.get("iter_count", 0))
    row["wall_time_s"] = float(stats.get("t_wall_total", 0.0))
    row["status"] = str(stats.get("return_status", ""))
    row["warm_started"] = warm_start is not None
    # Pass on converged points only, so failures do not propagate along a chain.
    row["warm_start"] = np.array(src.value(opti.x)).ravel() if feasible else None
    return row


def _extract(src, X, Xc, U, T, V_J, B, d, q_ref_quat, robot, nq, n,
             gait, t_center, v_target, feasible):
    """Metrics and trajectories of a (possibly failed) iterate."""
    T_val = float(src.value(T))
    Xt_val = src.value(X)
    U_val = src.value(U)
    X_val = tangent_to_legacy(Xt_val, q_ref_quat, robot.model,
                              nq=robot.nq_reduced, nv=robot.nv_reduced)

    forward = float(X_val[0, -1] - X_val[0, 0])
    speed = forward / T_val

    vc = src.value(Xc)[V_J, :]             # joint velocities at collocation points
    energy = cycle_energy(U_val, vc, B, d, n, T_val)
    cot = cost_of_transport(energy, robot, forward)

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
