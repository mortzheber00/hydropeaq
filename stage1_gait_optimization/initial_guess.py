"""
Initial guess for the gait OCP.

Two strategies are provided:

1. Fourier-series paddling trajectory (Qu et al. 2025):
   build_initial_guess(dyn, gait, N, T_FIXED, TAU_MAX)

2. Robot firmware IK gait — mirrors Robot_Swim_Task_IK from the embedded C
   firmware (4-phase state machine: recovery → strike → power → lift):
   build_robot_ik_initial_guess(dyn, N, T_FIXED, TAU_MAX, ...)

Both return  (X_guess (nx, N+1),  U_guess (n_act, N)).
"""

from __future__ import annotations

import numpy as np
from hydro_model import SymbolicDynamics

# Supported gaits and their power-phase ratios
GAITS = {
    "LSPG25": 0.25,  # lateral-sequence, 25 % power phase
    "LSPG33": 0.33,  # lateral-sequence, 33 % power phase (fastest in paper)
    "TLPG50": 0.50,  # trot-like,        50 % power phase (most stable)
}

# Phase offsets (cycle fraction) per leg [FL, FR, HL, HR].
# Paper notation: LF=FL, RF=FR, LH=HL, RH=HR.
# LSPG: LH=0%, LF=25%, RF=50%, RH=75%
# TLPG: diagonal pairs (LF,RH)=0%, (RF,LH)=50%
_LSPG_OFFSETS = np.array([0.25, 0.50, 0.00, 0.75])
_TLPG_OFFSETS = np.array([0.00, 0.50, 0.50, 0.00])

# Joint-angle mapping amplitudes — tune independently per leg type
_FRONT_THIGH_AMP = 0.35  # rad
_FRONT_CALF_AMP = 0.8  # rad
_HIND_THIGH_AMP = 0.55  # rad
_HIND_CALF_AMP = 0.4  # rad


# ── Fourier trajectory ───────────────────────────────────────────────────────


def paper_fourier_trajectory(pp_ratio: float, n_harmonics: int = 3):
    """Fourier-series leg trajectory from Qu et al. 2025.

    Parameters
    ----------
    pp_ratio : float
        Fraction of the cycle spent in the power phase (0.25 / 0.33 / 0.50).
    n_harmonics : int
        Number of Fourier harmonics (paper uses 3).

    Returns
    -------
    alpha1, alpha2 : callables  t -> float|array, normalised angle in [-1, 1]
    dalpha1, dalpha2 : callables  t -> float|array, d(alpha)/d(t_norm)
    All callables accept normalised cycle time t in [0, 1).
    """
    theta1_kp = np.array(
        [100, 90, 80, 70, 60, 50, 40, 50, 60, 70, 80, 90], dtype=float
    )
    theta2_kp = np.array(
        [80, 60, 40, 20, 38.33, 56.67, 75, 86.67, 98.33, 110, 100, 90], dtype=float
    )

    theta1_mean, theta1_half = 70.0, 30.0
    theta2_mean, theta2_half = 65.0, 45.0

    alpha1_kp = (theta1_kp - theta1_mean) / theta1_half
    alpha2_kp = (theta2_kp - theta2_mean) / theta2_half

    n_half = 6
    t_kp = np.concatenate(
        [
            np.linspace(0, pp_ratio, n_half + 1)[:-1],
            np.linspace(pp_ratio, 1.0, n_half + 1)[:-1],
        ]
    )

    n = len(t_kp)
    A = np.ones((n, 1 + 2 * n_harmonics))
    for h in range(1, n_harmonics + 1):
        A[:, 2 * h - 1] = np.cos(2 * np.pi * h * t_kp)
        A[:, 2 * h] = np.sin(2 * np.pi * h * t_kp)

    c1, _, _, _ = np.linalg.lstsq(A, alpha1_kp, rcond=None)
    c2, _, _, _ = np.linalg.lstsq(A, alpha2_kp, rcond=None)

    def _eval(t, c):
        t = np.asarray(t, dtype=float)
        out = np.full_like(t, c[0])
        for h in range(1, n_harmonics + 1):
            out += c[2 * h - 1] * np.cos(2 * np.pi * h * t)
            out += c[2 * h] * np.sin(2 * np.pi * h * t)
        return out

    def _deval(t, c):
        t = np.asarray(t, dtype=float)
        out = np.zeros_like(t)
        for h in range(1, n_harmonics + 1):
            w = 2 * np.pi * h
            out += -c[2 * h - 1] * w * np.sin(w * t)
            out += c[2 * h] * w * np.cos(w * t)
        return out

    return (
        lambda t: _eval(t, c1),
        lambda t: _eval(t, c2),
        lambda t: _deval(t, c1),
        lambda t: _deval(t, c2),
    )


# ── Main entry point ─────────────────────────────────────────────────────────


def build_initial_guess(
    dyn: SymbolicDynamics,
    gait: str,
    N: int,
    T_FIXED: float,
    TAU_MAX: float,
    hind_thigh_offset: float = 0.0,
    hind_calf_offset: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (X_guess, U_guess) for the gait OCP.

    Kinematics follow the paddling trajectory of Qu et al. 2025.
    Torques are computed via inverse dynamics on the kinematic trajectory.
    """
    if gait not in GAITS:
        raise ValueError(f"Unknown gait '{gait}'. Choose from {list(GAITS)}")

    robot = dyn.robot
    nq, nv = robot.nq, robot.nv
    nx = nq + nv
    n_act = robot.n_actuated
    dt_val = T_FIXED / N

    q_trim = dyn.find_trim_state()

    pp_ratio = GAITS[gait]
    phase_offsets = _TLPG_OFFSETS if gait == "TLPG50" else _LSPG_OFFSETS
    alpha1, alpha2, dalpha1, dalpha2 = paper_fourier_trajectory(pp_ratio)

    # ── Step 1: build joint trajectory ──────────────────────────────────
    q_joints = np.zeros((n_act, N + 1))
    v_joints = np.zeros((n_act, N + 1))

    for k in range(N + 1):
        t_norm = k / N
        for i, phi_off in enumerate(phase_offsets):
            t_leg = (t_norm + phi_off) % 1.0
            a1 = float(alpha1(t_leg))
            a2 = float(alpha2(t_leg))
            da1 = float(dalpha1(t_leg)) / T_FIXED
            da2 = float(dalpha2(t_leg)) / T_FIXED

            is_hind = i >= 2
            s = -1 if is_hind else 1
            z = 1 if is_hind else -1
            thigh_amp = _HIND_THIGH_AMP if is_hind else _FRONT_THIGH_AMP
            calf_amp = _HIND_CALF_AMP if is_hind else _FRONT_CALF_AMP
            b = i * 3

            q_joints[b, k] = q_trim[7 + b]
            q_joints[b + 1, k] = (
                q_trim[8 + b]
                + z * s * a1 * thigh_amp
                + (hind_thigh_offset if is_hind else 0.0)
            )
            q_joints[b + 2, k] = (
                q_trim[9 + b]
                + s * a2 * calf_amp
                + (hind_calf_offset if is_hind else 0.0)
            )

            v_joints[b, k] = 0.0
            v_joints[b + 1, k] = s * da1 * thigh_amp
            v_joints[b + 2, k] = z * s * da2 * calf_amp

    a_joints = (v_joints[:, 1:] - v_joints[:, :-1]) / dt_val

    # ── Step 2: simulate base DOF with prescribed joints ─────────────────
    print("  Simulating base DOF (prescribed joint kinematics)...")
    q_base_traj, v_base_traj = _simulate_base_kinematics(
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


# ── Robot firmware IK gait ────────────────────────────────────────────────────

_FW_LEG_NAMES = ["Front_Left", "Front_Right", "Hind_Left", "Hind_Right"]


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
    """
    t_r = r_rec * T_c
    t_s = r_str * T_c
    t_p = r_pow * T_c

    if t_in_cycle < t_r:
        progress = t_in_cycle / t_r if t_r > 0 else 1.0
        return x_back + (x_front - x_back) * progress, dz_surface
    elif t_in_cycle < t_r + t_s:
        progress = (t_in_cycle - t_r) / t_s if t_s > 0 else 1.0
        return x_front, dz_surface + (dz_deep - dz_surface) * progress
    elif t_in_cycle < t_r + t_s + t_p:
        progress = (t_in_cycle - t_r - t_s) / t_p if t_p > 0 else 1.0
        return x_front + (x_back - x_front) * progress, dz_deep
    else:
        t_l = T_c - (t_r + t_s + t_p)
        progress = (t_in_cycle - t_r - t_s - t_p) / t_l if t_l > 0 else 1.0
        return x_back, dz_deep + (dz_surface - dz_deep) * progress


def _solve_leg_ik(
    robot,
    q_context: np.ndarray,
    leg_idx: int,
    foot_target: np.ndarray,
    q_trim: np.ndarray | None = None,
) -> np.ndarray:
    """Find the 3 joint angles (side, thigh, calf) that place the foot at foot_target.

    Uses Levenberg-Marquardt (least_squares) which handles near-singular
    Jacobians via adaptive damping.  Falls back to q_trim when the primary
    warm-start fails, so phase transitions are rescued without causing branch
    switches during smooth within-phase motion.
    """
    from scipy.optimize import least_squares

    _IK_TOL = 1e-4  # acceptable foot-position error [m]

    js = 7 + leg_idx * 3
    name = _FW_LEG_NAMES[leg_idx]

    def residual(q_leg: np.ndarray) -> np.ndarray:
        q = q_context.copy()
        q[js : js + 3] = q_leg
        robot.forward_kinematics(q)
        return np.array(robot.foot_positions()[name]) - foot_target

    def _solve(q0: np.ndarray) -> tuple[np.ndarray, float]:
        res = least_squares(residual, q0, method="lm")
        return res.x, float(np.linalg.norm(residual(res.x)))

    q0_primary = q_context[js : js + 3].copy()
    sol, err = _solve(q0_primary)
    if err < _IK_TOL:
        return sol

    # Primary warm-start failed — try neutral (trim) configuration.
    if q_trim is not None:
        q0_trim = q_trim[js : js + 3].copy()
        sol_trim, err_trim = _solve(q0_trim)
        if err_trim < err:
            return sol_trim

    return sol


def build_robot_ik_initial_guess(
    dyn: SymbolicDynamics,
    N: int,
    T_FIXED: float,
    TAU_MAX: float,
    *,
    ratio_recovery: float = 0.4,
    ratio_strike: float = 0.1,
    ratio_power: float = 0.4,
    ratio_lift: float = 0.1,
    stroke_len: float = 0.05,
    stand_h: float = 0.14,
    depth_surface: float = 0.14,
    depth_deep: float = 0.20,
    center_x_front: float = 0.0,
    center_x_rear: float = 0.0,
    n_cycles: float = 1.0,
    diagonal_phase_offset: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (X_guess, U_guess) from the robot firmware IK-based swim gait.

    Mirrors Robot_Swim_Task_IK from the embedded C firmware.  The gait is a
    4-phase state machine (recovery → strike → power → lift) timed by the
    four ratio_* parameters (which must sum to 1).  Foot Cartesian targets are
    converted to joint angles via Pinocchio IK at each shooting node.

    Gait phasing: FL and HR start at the beginning of recovery; FR and HL are
    shifted back by ``ratio_recovery / 2`` cycles — this matches the firmware's
    ``swim_timer[FR/HL] = -t_recovery / 2`` diagonal phase offset.

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
    diagonal_phase_offset : float
        Fractional cycle offset between the two diagonal pairs (FL+HR vs
        FR+HL).  Default 0.5 gives a trot-like gait where each pair is in
        its power stroke while the other is recovering, keeping at least one
        pair producing thrust throughout 80 % of the cycle.

        The C firmware uses ``ratio_recovery / 2`` (≈ 0.2 with default
        ratios), which creates a 40 % dead zone per cycle — fine for
        continuous multi-cycle swimming but causes the robot to coast to a
        stop halfway through a single-period OCP initial guess.
    """
    robot = dyn.robot
    nq, nv = robot.nq, robot.nv
    nx = nq + nv
    n_act = robot.n_actuated
    dt_val = T_FIXED / N

    q_trim = dyn.find_trim_state()
    robot.forward_kinematics(q_trim)
    trim_feet = robot.foot_positions()  # leg_name -> (3,) world-frame position

    T_c = T_FIXED / n_cycles

    # FR (i=1) and HL (i=2) are offset by diagonal_phase_offset relative to FL/HR.
    phase_offsets = [0.0, diagonal_phase_offset, diagonal_phase_offset, 0.0]

    # Vertical deviation from trim foot z (world frame, z-up):
    #   C code z is positive-downward so deeper → smaller world z → negative dz.
    dz_surface = -(depth_surface - stand_h)
    dz_deep = -(depth_deep - stand_h)

    # ── Solve IK for all legs at every shooting node ─────────────────────
    q_joints = np.zeros((n_act, N + 1))
    q_ctx = q_trim.copy()  # IK context — base stays at trim throughout

    for k in range(N + 1):
        t = k * dt_val
        for i in range(4):
            cx = center_x_front if i < 2 else center_x_rear
            p_ref = trim_feet[_FW_LEG_NAMES[i]]

            t_cyc = (t + phase_offsets[i] * T_c) % T_c
            dx, dz = _firmware_foot_target(
                t_cyc, T_c,
                ratio_recovery, ratio_strike, ratio_power,
                cx + stroke_len, cx - stroke_len,
                dz_surface, dz_deep,
            )

            target = np.array([p_ref[0] + dx, p_ref[1], p_ref[2] + dz])
            q_sol = _solve_leg_ik(robot, q_ctx, i, target, q_trim)

            js = 7 + i * 3
            q_ctx[js : js + 3] = q_sol           # warm-start next leg / timestep
            q_joints[i * 3 : (i + 1) * 3, k] = q_sol

    # ── Velocities via central differences ───────────────────────────────
    v_joints = np.zeros((n_act, N + 1))
    v_joints[:, 1:-1] = (q_joints[:, 2:] - q_joints[:, :-2]) / (2.0 * dt_val)
    v_joints[:, 0] = (q_joints[:, 1] - q_joints[:, 0]) / dt_val
    v_joints[:, -1] = (q_joints[:, -1] - q_joints[:, -2]) / dt_val

    a_joints = (v_joints[:, 1:] - v_joints[:, :-1]) / dt_val

    # ── Simulate base DOF with prescribed joint kinematics ───────────────
    print("  Simulating base DOF (firmware IK trajectory)...")
    q_base_traj, v_base_traj = _simulate_base_kinematics(
        dyn, q_joints, v_joints, a_joints, q_trim, dt_val
    )

    # ── Assemble full state trajectory ───────────────────────────────────
    X_guess = np.zeros((nx, N + 1))
    for k in range(N + 1):
        q_k = np.concatenate([q_base_traj[:, k], q_joints[:, k]])
        v_k = np.concatenate([v_base_traj[:, k], v_joints[:, k]])
        X_guess[:, k] = np.concatenate([q_k, v_k])

    # ── Torque guess via inverse dynamics ────────────────────────────────
    U_guess = np.zeros((n_act, N))
    for k in range(N):
        a_k = (X_guess[nq:, k + 1] - X_guess[nq:, k]) / dt_val
        tau_id = dyn.eval_inverse_dynamics(X_guess[:nq, k], X_guess[nq:, k], a_k)
        U_guess[:, k] = np.clip(tau_id[6:], -TAU_MAX, TAU_MAX)

    return X_guess, U_guess


def _simulate_base_kinematics(
    dyn: SymbolicDynamics,
    q_joints: np.ndarray,
    v_joints: np.ndarray,
    a_joints: np.ndarray,
    q_trim: np.ndarray,
    dt: float,
    n_cycles: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate base DOF with prescribed joint trajectory.

    Runs for up to n_cycles periods. Returns the last cycle's base trajectory.
    """
    N = q_joints.shape[1] - 1

    q_base = q_trim[:7].copy()
    q_base[0] = 0.0
    v_base = np.zeros(6)

    dist_prev = np.nan
    q_cycle = v_cycle = None

    for cycle in range(n_cycles):
        q_base[1:7] = q_trim[1:7]
        q_base[0] = 0.0
        v_base[1:] = 0.0

        q_cycle = np.zeros((7, N + 1))
        v_cycle = np.zeros((6, N + 1))
        q_cycle[:, 0] = q_base.copy()
        v_cycle[:, 0] = v_base.copy()

        for k in range(N):
            q = np.concatenate([q_base, q_joints[:, k]])
            v = np.concatenate([v_base, v_joints[:, k]])

            M = np.array(dyn.f_M_rb(q)) + np.array(dyn.f_M_added(q))
            C = np.array(dyn.f_C_rb(q, v))
            g = np.array(dyn.f_g_rb(q)).flatten()
            tb = np.array(dyn.f_tau_buoyancy(q)).flatten()
            td = np.array(dyn.f_tau_drag(q, v)).flatten()

            rhs = tb[:6] + td[:6] - C[:6, :] @ v - g[:6] - M[:6, 6:] @ a_joints[:, k]
            a_base = np.linalg.solve(M[:6, :6], rhs)

            v_base = v_base + a_base * dt

            qx, qy, qz, qw = q_base[3], q_base[4], q_base[5], q_base[6]
            vx, vy, vz = v_base[0], v_base[1], v_base[2]
            wx, wy, wz = v_base[3], v_base[4], v_base[5]

            q_base[0] += (
                (1 - 2 * (qy**2 + qz**2)) * vx
                + 2 * (qx * qy - qw * qz) * vy
                + 2 * (qx * qz + qw * qy) * vz
            ) * dt
            q_base[1] += (
                2 * (qx * qy + qw * qz) * vx
                + (1 - 2 * (qx**2 + qz**2)) * vy
                + 2 * (qy * qz - qw * qx) * vz
            ) * dt
            q_base[2] += (
                2 * (qx * qz - qw * qy) * vx
                + 2 * (qy * qz + qw * qx) * vy
                + (1 - 2 * (qx**2 + qy**2)) * vz
            ) * dt
            q_base[3] += 0.5 * (qw * wx + qy * wz - qz * wy) * dt
            q_base[4] += 0.5 * (qw * wy + qz * wx - qx * wz) * dt
            q_base[5] += 0.5 * (qw * wz + qx * wy - qy * wx) * dt
            q_base[6] += 0.5 * (-qx * wx - qy * wy - qz * wz) * dt
            q_base[3:7] /= np.linalg.norm(q_base[3:7])

            q_cycle[:, k + 1] = q_base.copy()
            v_cycle[:, k + 1] = v_base.copy()

        dist = float(q_base[0])
        print(
            f"    cycle {cycle + 1:2d}/{n_cycles}: dx = {dist:.4f} m, "
            f"mean vx = {np.mean(v_cycle[0]):.4f} m/s"
        )

        if not np.isnan(dist_prev) and abs(dist - dist_prev) < 1e-5:
            print("    Converged.")
            break
        dist_prev = dist

    return q_cycle, v_cycle
