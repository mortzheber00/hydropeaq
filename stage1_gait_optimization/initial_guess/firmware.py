"""Firmware IK swim gait, a port of ``Robot_Swim_Task_IK``.

Each foot follows a 4-phase Cartesian path: recovery (forward swing near the
surface), strike (descend), power (backward sweep at depth), lift (ascend).
The waypoints match the firmware, but phases are interpolated with a quintic
smootherstep instead of linearly, so foot accelerations stay finite for inverse
dynamics. Foot y is taken from the trim pose.
"""

from __future__ import annotations

import numpy as np
from hydro_model import SymbolicDynamics
from hydro_model.coordinate_map import IdentityMap

from .assemble import assemble_guess
from .foot_ik import solve_leg_theta

# Stroke shape for amph [m]; other robots override via RobotSpec.firmware_gait.
_DEFAULT_GAIT = {
    "ratio_recovery": 0.55,
    "ratio_strike": 0.1,
    "ratio_power": 0.15,
    "ratio_lift": 0.2,
    "stroke_len": 0.05,
    "stand_h": 0.14,
    "depth_surface": 0.15,
    "depth_deep": 0.2,
    "center_x_front": 0.0145,
    "center_x_rear": -0.02,
}

# Foot-tracking error above which the stroke counts as unreachable.
_TRACK_TOL = 1e-3  # [m]


def _smootherstep(p: float) -> float:
    """Quintic smoothstep ``6p⁵ − 15p⁴ + 10p³`` on [0, 1] (zero 1st/2nd derivative at the ends)."""
    return p * p * p * (p * (6.0 * p - 15.0) + 10.0)


def _firmware_foot_target(
    t_in_cycle: float,
    T_c: float,
    r_rec: float,
    r_str: float,
    r_pow: float,
    x_front: float,
    x_back: float,
    dz_surface: float,
    dz_deep: float,
) -> tuple[float, float]:
    """Foot offset ``(dx, dz)`` from trim at ``t_in_cycle`` in ``[0, T_c)``; z is up."""
    t_r = r_rec * T_c
    t_s = r_str * T_c
    t_p = r_pow * T_c

    if t_in_cycle < t_r:
        p = _smootherstep(t_in_cycle / t_r) if t_r > 0 else 1.0
        return x_back + (x_front - x_back) * p, dz_surface
    elif t_in_cycle < t_r + t_s:
        p = _smootherstep((t_in_cycle - t_r) / t_s) if t_s > 0 else 1.0
        return x_front, dz_surface + (dz_deep - dz_surface) * p
    elif t_in_cycle < t_r + t_s + t_p:
        p = _smootherstep((t_in_cycle - t_r - t_s) / t_p) if t_p > 0 else 1.0
        return x_front + (x_back - x_front) * p, dz_deep
    else:
        t_l = T_c - (t_r + t_s + t_p)
        p = _smootherstep((t_in_cycle - t_r - t_s - t_p) / t_l) if t_l > 0 else 1.0
        return x_back, dz_deep + (dz_surface - dz_deep) * p


def _solve_leg_ik(
    robot,
    q_context: np.ndarray,
    leg_idx: int,
    foot_target: np.ndarray,
    q_trim: np.ndarray | None = None,
    n_per_leg: int = 3,
    leg_names: list[str] | None = None,
) -> np.ndarray:
    """Serial-leg IK for (side, thigh, calf) placing the foot at ``foot_target``.

    Solves from both the previous and the trim configuration and keeps the
    converged solution closest to the previous one, avoiding branch flips.
    """
    from scipy.optimize import least_squares

    _IK_TOL = 1e-5  # acceptable foot-position error [m]

    js = 7 + leg_idx * n_per_leg
    name = (leg_names or list(robot.spec.leg_names))[leg_idx]
    q_prev = q_context[js : js + n_per_leg].copy()

    def residual(q_leg: np.ndarray) -> np.ndarray:
        q = q_context.copy()
        q[js : js + n_per_leg] = q_leg
        robot.forward_kinematics(q)
        return np.array(robot.foot_positions()[name]) - foot_target

    def _solve(q0: np.ndarray) -> tuple[np.ndarray, float]:
        res = least_squares(residual, q0, method="lm")
        return res.x, float(np.linalg.norm(residual(res.x)))

    candidates = [_solve(q_prev)]
    if q_trim is not None:
        candidates.append(_solve(q_trim[js : js + n_per_leg].copy()))

    feasible = [(q, err) for q, err in candidates if err < _IK_TOL]
    if feasible:
        return min(feasible, key=lambda c: np.linalg.norm(c[0] - q_prev))[0]

    return min(candidates, key=lambda c: c[1])[0]


def build_robot_ik_initial_guess(
    dyn: SymbolicDynamics,
    N: int,
    T_FIXED: float,
    TAU_MAX: float,
    *,
    n_cycles: float = 1.0,
    diagonal_phase_offset: float | None = None,
    **gait: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build ``(X_guess, U_guess)`` from the firmware swim gait.

    Gait parameters come from keyword arguments, then
    ``RobotSpec.firmware_gait``, then the amph defaults.

    Parameters
    ----------
    ratio_recovery / ratio_strike / ratio_power / ratio_lift : float
        Fraction of the cycle spent in each phase; must sum to 1.
    stroke_len : float
        Half-stroke length along x [m].
    stand_h : float
        Nominal foot depth below the hip [m].
    depth_surface, depth_deep : float
        Foot depth during recovery and during the power stroke [m].
    center_x_front / center_x_rear : float
        Stroke centre offset from the trim foot position [m].
    n_cycles : float
        Gait cycles within ``T_FIXED``.
    diagonal_phase_offset : float, optional
        Phase of FR/HL relative to FL/HR in cycles (negative = lag). Defaults
        to the firmware's ``-ratio_recovery / 2``.
    """
    robot = dyn.robot
    n_act = robot.n_actuated
    dt_val = T_FIXED / N

    unknown = set(gait) - set(_DEFAULT_GAIT)
    if unknown:
        raise TypeError(
            f"unknown gait parameter(s) {sorted(unknown)}; "
            f"choose from {sorted(_DEFAULT_GAIT)}"
        )
    p = {**_DEFAULT_GAIT, **robot.spec.firmware_gait, **gait}
    ratio_recovery = p["ratio_recovery"]
    ratio_strike, ratio_power = p["ratio_strike"], p["ratio_power"]
    stroke_len, stand_h = p["stroke_len"], p["stand_h"]
    depth_surface, depth_deep = p["depth_surface"], p["depth_deep"]
    center_x_front, center_x_rear = p["center_x_front"], p["center_x_rear"]

    leg_names = list(robot.spec.leg_names)
    n_legs = len(leg_names)
    if n_legs != 4:
        raise ValueError(
            f"the firmware gait phases two diagonal pairs, so it needs 4 legs; "
            f"{robot.spec.name} has {n_legs}"
        )
    n_per_leg = n_act // n_legs
    serial = isinstance(robot.coord_map, IdentityMap)

    q_trim = dyn.find_trim_state()
    robot.forward_kinematics(q_trim)
    trim_feet = robot.foot_positions()  # leg_name -> (3,) world-frame position

    T_c = T_FIXED / n_cycles

    if diagonal_phase_offset is None:
        diagonal_phase_offset = -ratio_recovery / 2.0

    # Leg order FL, FR, HL, HR
    phase_offsets = [0.0, diagonal_phase_offset, diagonal_phase_offset, 0.0]

    # Firmware depths are positive downward; world z is up.
    dz_surface = -(depth_surface - stand_h)
    dz_deep = -(depth_deep - stand_h)

    # --- IK for all legs at every node ---
    # The base is held at trim, since the targets are relative to the trim feet.
    q_joints = np.zeros((n_act, N + 1))
    act_ctx = q_trim[7:].copy() if serial else np.zeros(n_act)
    q_base_trim = q_trim[:7]
    worst_err = 0.0

    for k in range(N + 1):
        t = k * dt_val
        for i in range(n_legs):
            cx = center_x_front if i < 2 else center_x_rear
            leg = leg_names[i]
            p_ref = trim_feet[leg]

            t_cyc = (t + phase_offsets[i] * T_c) % T_c
            dx, dz = _firmware_foot_target(
                t_cyc, T_c,
                ratio_recovery, ratio_strike, ratio_power,
                cx + stroke_len, cx - stroke_len,
                dz_surface, dz_deep,
            )

            target = np.array([p_ref[0] + dx, p_ref[1], p_ref[2] + dz])
            sl = slice(i * n_per_leg, (i + 1) * n_per_leg)

            if serial:
                q_ctx = np.concatenate([q_base_trim, act_ctx])
                sol = _solve_leg_ik(robot, q_ctx, i, target, q_trim,
                                    n_per_leg=n_per_leg, leg_names=leg_names)
            else:
                sol, err = solve_leg_theta(
                    robot, leg, act_ctx, sl, target[[0, 2]], q_base_trim,
                )
                worst_err = max(worst_err, err)

            act_ctx[sl] = sol            # warm-start next leg / timestep
            q_joints[sl, k] = sol

    if worst_err > _TRACK_TOL:
        raise ValueError(
            f"the requested stroke leaves {robot.spec.name}'s reachable set: "
            f"worst foot-tracking error {worst_err * 1e3:.1f} mm exceeds "
            f"{_TRACK_TOL * 1e3:.1f} mm — reduce stroke_len or the depth swing"
        )

    # --- Periodic finite differences (q[:, 0] == q[:, N]) ---
    v_joints = np.zeros((n_act, N + 1))
    v_joints[:, 1:-1] = (q_joints[:, 2:] - q_joints[:, :-2]) / (2.0 * dt_val)
    v_joints[:, 0] = (q_joints[:, 1] - q_joints[:, -2]) / (2.0 * dt_val)
    v_joints[:, -1] = v_joints[:, 0]

    a_joints = np.zeros((n_act, N + 1))
    a_joints[:, :-2] = (v_joints[:, 1:-1] - v_joints[:, :-2]) / dt_val
    a_joints[:, -2] = (v_joints[:, 0] - v_joints[:, -2]) / dt_val
    a_joints[:, -1] = a_joints[:, 0]

    return assemble_guess(
        dyn, q_joints, v_joints, a_joints, T_FIXED, N, TAU_MAX,
        label="firmware IK trajectory",
    )
