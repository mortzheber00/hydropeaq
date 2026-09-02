#!/usr/bin/env python3
"""
Importable collocation-OCP evaluator for the (gait, T) co-design sweep.

Wraps the shared collocation transcription (``ocp_common.build_collocation_nlp``)
into a pure function ``solve_gait_ocp`` that the outer loop (see
``run_codesign.py``) calls per design point.  Same NLP the standalone
``trajopt/run_collocation.py`` driver builds — same transcription, the same
``RobotSpec.ocp`` settings, the same left–right symmetry constraints and the
same solver options — with exactly three deliberate differences:

  * No module-level mutable state, no MLflow, no interactive prompts — the
    function is callable from worker processes.
  * The cycle period T does a **narrow free-T refine**: IPOPT may move T within
    ``[t_center - free_T_band, t_center + free_T_band]`` (band = 0 pins it),
    rather than over the spec's ``[t_min, t_max]``.
  * The objective is a clean effort/energy minimisation (no forward-distance
    reward); the forward-speed floor stays a hard constraint.  Speed and energy
    are returned as metrics so the outer loop owns the speed-vs-efficiency
    tradeoff.

Anything else that differs is drift and should be fixed here, not worked around.
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

# ── OCP parameters ──────────────────────────────────────────────────────────
# Read off the robot's spec, the same block run_collocation.py reads, so the two
# drivers keep solving the same problem.  They were copied constants here, which
# silently went stale the first time OCPSettings was retuned.  Cheap at import:
# get_spec only imports the robot's module, it does not build the model.
CFG = get_spec(ROBOT).ocp

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
    """Solve the collocation OCP for one (gait, T, target-speed) design point.

    Minimises energy/effort subject to an average-speed floor ``v_target``
    (ε-constraint method): the floor binds, so the converged speed ≈ ``v_target``
    and sweeping ``v_target`` in the outer loop traces the speed axis of the
    Pareto front.  The period T is free within
    ``[t_center - free_T_band, t_center + free_T_band]`` (band 0 pins it) and acts
    as the efficiency knob at each target speed.

    ``warm_start`` is the decision vector returned by a previous solve of the
    *same* (gait, T centre, n) at a neighbouring target speed — see
    ``run_codesign.build_chains``.  It seeds the iterate; the solver options are
    the same either way.

    Returns a dict with ``feasible``, the converged ``speed`` and
    ``energy``/``cot`` metrics, the solved ``T``, the legacy-state / control
    trajectories ``X`` / ``U``, solver stats, and a ``warm_start`` vector to seed
    the next link (``None`` if this solve did not converge).
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

    # Effort/energy minimisation; the speed floor lives inside the shared NLP.
    opti.minimize(
        w["power"] * nlp["power_cost"]
        + w["vel_smooth"] * nlp["vel_smooth_cost"]
        + w["joint_smooth"] * nlp["joint_smooth_cost"]
        + w["drift"] * nlp["drift_cost"]
    )

    # Same left–right symmetry the driver imposes — without it the co-design
    # sweep would be scoring a different problem than the one that gets solved.
    if CFG.enforce_symmetry:
        add_symmetry_constraints(opti, nlp["X"], X_guess, robot, n,
                                 phases=CFG.symmetry_phase,
                                 verbose=print_level > 0)

    # Seeding the whole vector replaces the per-variable seeds that
    # build_collocation_nlp took from the analytic guess.  It carries more than
    # the N ladder's resampled (X, U) does: Xc gets the converged collocation
    # states rather than a linear interpolation, and T the previously solved
    # period rather than t_center.
    #
    # Primal only, and the solver options below are left exactly as a cold solve
    # has them.  Restoring the multipliers too would need warm_start_init_point
    # — IPOPT ignores supplied duals without it — and that flag drags its
    # barrier settings along with it.  That package came out slower than a cold
    # sweep on an n=12 chain, so it is not used.
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
    # Only a converged point is worth passing on: seeding the next link from a
    # failed iterate would propagate the failure down the rest of the chain.
    row["warm_start"] = np.array(src.value(opti.x)).ravel() if feasible else None
    return row


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

    vc = src.value(Xc)[V_J, :]             # joint velocities at the collocation points
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
