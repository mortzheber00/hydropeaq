"""Robot firmware IK swim gait — mirrors ``Robot_Swim_Task_IK``.

The firmware drives each leg through a 4-phase state machine:
    recovery (forward swing, foot near surface)
    strike   (descend to power depth)
    power    (backward sweep at depth)
    lift     (ascend back to surface)

This module mirrors the C firmware's waypoints and phase semantics.  The
*only* deliberate departure is the in-phase interpolation: the C code uses
linear interpolation (piecewise-constant foot velocity, discontinuous at
phase boundaries — pathological for inverse dynamics), while here we use
a quintic smootherstep ``6p⁵ − 15p⁴ + 10p³`` (C² at both endpoints).  The
waypoints — and therefore the gait shape — are unchanged.

Lateral ``target_y``: the C code sets ``±side_w`` explicitly; here it is
inherited from the trim FK (URDF symmetry ⇒ trim foot y equals firmware
``side_w`` to within URDF tolerance).

The gait itself is a foot *Cartesian* path, so it is not tied to any particular
leg mechanism and retargets onto a closed-chain robot unchanged.  Only two
things are: the joint solve, which goes through ``foot_ik`` for a robot with a
coordinate map because such a robot has no tree joints to solve for; and the
stroke dimensions, which are in metres and therefore live in ``RobotSpec``.
"""

from __future__ import annotations

import numpy as np
from hydro_model import SymbolicDynamics
from hydro_model.coordinate_map import IdentityMap

from .assemble import assemble_guess
from .foot_ik import solve_leg_theta

# Stroke shape as calibrated on amph, in metres of foot travel.  A robot whose
# legs are a different size overrides these through ``RobotSpec.firmware_gait``.
_DEFAULT_GAIT = {
    "ratio_recovery": 0.55,
    "ratio_strike": 0.1,
    "ratio_power": 0.15,
    "ratio_lift": 0.2,
    "stroke_len": 0.05,
    "stand_h": 0.14,
    "depth_surface": 0.14,
    "depth_deep": 0.2,
    "center_x_front": -0.04,
    "center_x_rear": 0.0,
}

# Foot-tracking error above which the requested stroke is reported as not
# reachable.  Well below the smallest stroke worth running (BODY2's usable box
# is +-20 mm), so it separates "off the workspace" from IK round-off.
_TRACK_TOL = 1e-3  # [m]


def _smootherstep(p: float) -> float:
    """Quintic C² smoothstep on [0, 1]: ``6p⁵ − 15p⁴ + 10p³``.

    smootherstep(0) = 0, smootherstep(1) = 1, with first and second
    derivatives vanishing at both endpoints.
    """
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
    """Compute (Δx, Δz) foot offsets from trim at time t_in_cycle ∈ [0, T_c).

    z convention: positive = higher in world frame (= less deep in water).
    Phases: recovery (forward swing) → strike (descend) → power (backward) → lift (ascend).
    Within each phase, ``progress`` is run through a quintic smootherstep so
    foot velocity and acceleration are continuous at every phase boundary.
    """
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
    """Find the 3 joint angles (side, thigh, calf) that place the foot at foot_target.

    Uses Levenberg-Marquardt (least_squares) and picks the IK branch closest
    to the previous timestep's joint configuration — both the primary
    warm-start and the q_trim warm-start are always tried, and the candidate
    that achieves the IK tolerance with the smallest joint-space jump from
    the warm-start is kept.  This suppresses elbow-up/elbow-down branch flips
    that would otherwise inject step jumps into q_joints.
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
    """Build (X_guess, U_guess) from the robot firmware IK-based swim gait.

    Mirrors Robot_Swim_Task_IK from the embedded C firmware.  The gait is a
    4-phase state machine (recovery → strike → power → lift) timed by the
    four ratio_* parameters (which must sum to 1).  Foot Cartesian targets are
    converted to actuated coordinates by IK at each shooting node.

    The stroke-shape parameters below are resolved in order: an explicit
    keyword argument, then ``RobotSpec.firmware_gait``, then the amph-calibrated
    default.  They are absolute foot travel in metres, which is why a robot of
    a different size must override them -- amph's 50 mm stroke and 60 mm depth
    swing are several times BODY2's whole reachable box.

    Gait phasing: FL and HR start at the beginning of recovery; FR and HL
    lag by ``ratio_recovery / 2`` cycles — matching the firmware's
    ``swim_timer[FR/HL] = -t_recovery / 2`` diagonal phase offset (negative
    ``diagonal_phase_offset`` ⇒ FR/HL lag).

    Parameters
    ----------
    ratio_recovery / ratio_strike / ratio_power / ratio_lift : float
        Fraction of the cycle period spent in each phase.  Must sum to 1.
    stroke_len : float
        Half-stroke length in the forward (x) direction [m].
    stand_h : float
        Nominal foot depth below the hip at rest [m].
        Equivalent to the C firmware's ``stand_h`` (14 cm → 0.14 m).
    depth_surface : float
        Foot depth during the recovery / surface phase [m].
        Use ``≈ stand_h`` to keep the foot at nominal height during swing.
    depth_deep : float
        Foot depth during the power stroke [m].  Must be ≥ stand_h; larger
        values push the foot deeper and increase hydrodynamic thrust.
    center_x_front / center_x_rear : float
        Forward offset of the stroke x-centre from the trim foot position,
        for front / rear legs [m].
    n_cycles : float
        Number of complete gait cycles contained in T_FIXED (default 1.0).
    diagonal_phase_offset : float, optional
        Fractional cycle offset between the two diagonal pairs (FL+HR vs
        FR+HL).  Negative ⇒ FR/HL lag FL/HR; positive ⇒ lead.  Defaults to
        ``-ratio_recovery / 2`` — exactly the firmware's
        ``swim_timer[FR/HL] = -t_recovery / 2`` diagonal phase offset.
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

    # Firmware diagonal offset: FR/HL lag FL/HR by half a recovery phase.
    if diagonal_phase_offset is None:
        diagonal_phase_offset = -ratio_recovery / 2.0

    # FR (i=1) and HL (i=2) are offset by diagonal_phase_offset relative to FL/HR.
    phase_offsets = [0.0, diagonal_phase_offset, diagonal_phase_offset, 0.0]

    # Vertical deviation from trim foot z (world frame, z-up):
    #   C code z is positive-downward so deeper → smaller world z → negative dz.
    dz_surface = -(depth_surface - stand_h)
    dz_deep = -(depth_deep - stand_h)

    # ── Solve IK for all legs at every shooting node ─────────────────────
    # ``act`` carries the actuated coordinates, which for a serial robot are the
    # tree joints and for a mapped one are theta.  The base is held at trim
    # throughout: the stroke is defined relative to the trim foot positions, and
    # letting the base move here would make the target chase itself.
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

    # ── Velocities via central differences (periodic at the seam) ────────
    # q_joints[:, 0] == q_joints[:, N] by construction (cyclic t_cyc), so
    # the natural neighbours of the seam are q[:, 1] and q[:, N-1].
    v_joints = np.zeros((n_act, N + 1))
    v_joints[:, 1:-1] = (q_joints[:, 2:] - q_joints[:, :-2]) / (2.0 * dt_val)
    v_joints[:, 0] = (q_joints[:, 1] - q_joints[:, -2]) / (2.0 * dt_val)
    v_joints[:, -1] = v_joints[:, 0]

    # a_joints needs N+1 columns for assemble_guess; the seam value is the
    # periodic wrap, matching how v_joints is closed above.
    a_joints = np.zeros((n_act, N + 1))
    a_joints[:, :-2] = (v_joints[:, 1:-1] - v_joints[:, :-2]) / dt_val
    a_joints[:, -2] = (v_joints[:, 0] - v_joints[:, -2]) / dt_val
    a_joints[:, -1] = a_joints[:, 0]

    # ── Base simulation, assembly and torques (shared with every builder) ─
    return assemble_guess(
        dyn, q_joints, v_joints, a_joints, T_FIXED, N, TAU_MAX,
        label="firmware IK trajectory",
    )
