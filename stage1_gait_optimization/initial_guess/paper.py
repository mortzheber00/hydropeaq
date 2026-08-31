"""Paddling-gait initial guess (Qu et al. 2025).

Builds the paper's Fourier paddling trajectory, drives the front-leg joints
with it directly, and mirrors the motion onto the hind legs by inverse
kinematics so that every foot traces the same hip-relative path.

Front-leg joints map from the paper angles by an affine relation calibrated
on the hardware (see ``_THIGH_OFFSET_DEG`` / ``_CALF_OFFSET_DEG``).  The hind
legs are mounted rotated 180 deg about the vertical axis, so the same foot
path requires different joint angles; these are solved per timestep.

Because of that mounting the hind reachable set is the front one mirrored, and
the shared path need not lie inside it.  ``RobotSpec.paper_gait`` may therefore
carry a rigid transform that places the path into the hind legs' reach; it is
per robot, since the right placement depends on the geometry.
"""

from __future__ import annotations

import numpy as np
from hydro_model import SymbolicDynamics

from .base_sim import simulate_base_kinematics

# Supported gaits and their power-phase ratios (Qu et al. 2025, sec. 3.1).
GAITS = {
    "LSPG25": 0.25,  # lateral-sequence, 25 % power phase
    "LSPG33": 0.33,  # lateral-sequence, 33 % power phase (fastest in paper)
    "TLPG50": 0.50,  # trot-like,        50 % power phase (most stable)
}

# Phase offsets (cycle fraction) per leg [FL, FR, HL, HR].
# Paper notation: LF=FL, RF=FR, LH=HL, RH=HR.
# LSPG: LH=0%, RF=25%, RH=50%, LF=75%  (Qu et al. fig. 5(g),(h))
# TLPG: diagonal pairs (LF,RH)=0%, (RF,LH)=50%
_LSPG_OFFSETS = np.array([0.75, 0.25, 0.00, 0.50])
_TLPG_OFFSETS = np.array([0.00, 0.50, 0.50, 0.00])

# Front-leg paper-angle -> robot-joint map (degrees), calibrated on hardware:
#   thigh = theta1 - 30,   calf = 115 - theta2
_THIGH_OFFSET_DEG = -30.0
_CALF_OFFSET_DEG = 115.0

# Stroke keyframes (paper angles, degrees).  Each stroke is specified by only
# three key positions — initial / midpoint / end as (init, mid, end) tuples —
# and the intermediate key positions are filled in by uniform piecewise-linear
# interpolation (``_stroke_keypoints``).  theta1 = thigh, theta2 = shank.
# For a closed cycle, power "end" matches recovery "init" and recovery "end"
# matches power "init".  Retune the paddling gait by editing these.
_PADDLE_KEYFRAMES_DEG = {
    "theta1": {  # thigh
        "power":    (100.0, 70.0, 25.0),
        "recovery": (25.0, 70.0, 100.0),
    },
    "theta2": {  # shank
        "power":    (80.0, 20.0, 80.0),
        "recovery": (80.0, 115.0, 80.0),
    },
}


def _stroke_keypoints(theta: str, phase: str, fractions: np.ndarray) -> np.ndarray:
    """Sample a stroke's paper angle at the given fractions in [0, 1].

    Piecewise-linear through the (init, mid, end) keyframes at fractions
    (0.0, 0.5, 1.0).  ``theta`` is "theta1" or "theta2", ``phase`` is "power"
    or "recovery".
    """
    init, mid, end = _PADDLE_KEYFRAMES_DEG[theta][phase]
    return np.interp(fractions, [0.0, 0.5, 1.0], [init, mid, end])


def paper_fourier_trajectory(pp_ratio: float, n_harmonics: int = 3):
    """Fourier-series leg trajectory from Qu et al. 2025.

    One paddling cycle is divided into 12 key positions — 6 in the power phase,
    6 in the recovery phase — and a 3-harmonic Fourier series is fitted through
    them (paper eq. 1).  The key positions come from the init/mid/end keyframes
    in ``_PADDLE_KEYFRAMES_DEG``, with intermediates filled in automatically.
    ``theta1`` is the thigh angle and ``theta2`` the shank angle, in degrees.

    Parameters
    ----------
    pp_ratio : float
        Fraction of the cycle spent in the power phase (0.25 / 0.33 / 0.50).
    n_harmonics : int
        Number of Fourier harmonics (paper uses 3).

    Returns
    -------
    theta1, theta2 : callables  t -> float|array, joint angle in degrees.
        Both accept normalised cycle time t in [0, 1).
    """
    n_half = 6
    # Per-phase sample fractions are endpoint-exclusive: the stroke "end"
    # coincides with the next stroke's "init" (the t = pp_ratio / t = 1 knots).
    fr = np.arange(n_half) / n_half
    theta1_kp = np.concatenate(
        [_stroke_keypoints("theta1", "power", fr), _stroke_keypoints("theta1", "recovery", fr)]
    )
    theta2_kp = np.concatenate(
        [_stroke_keypoints("theta2", "power", fr), _stroke_keypoints("theta2", "recovery", fr)]
    )

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

    # Ridge penalty ∝ harmonic² damps the high-frequency coefficients that
    # ring near the keyframe corners (DC term a0 left unpenalized).
    lam = 0.6  # smoothing strength; raise for less ringing, lower to track corners
    penalty = np.zeros(1 + 2 * n_harmonics)
    for h in range(1, n_harmonics + 1):
        penalty[2 * h - 1] = penalty[2 * h] = lam * h**2
    ATA = A.T @ A + np.diag(penalty)
    c1 = np.linalg.solve(ATA, A.T @ theta1_kp)
    c2 = np.linalg.solve(ATA, A.T @ theta2_kp)

    def _eval(t, c):
        t = np.asarray(t, dtype=float)
        out = np.full_like(t, c[0])
        for h in range(1, n_harmonics + 1):
            out += c[2 * h - 1] * np.cos(2 * np.pi * h * t)
            out += c[2 * h] * np.sin(2 * np.pi * h * t)
        return out

    return (lambda t: _eval(t, c1), lambda t: _eval(t, c2))


# Identity: no robot is required to declare a transform, and one that does not
# gets exactly the behaviour this module had before the field existed.
_PAPER_GAIT_DEFAULTS = {"hind_rotation_deg": 0.0, "hind_dx": 0.0, "hind_dz": 0.0}


def hind_target_path(front_xz: np.ndarray, spec) -> np.ndarray:
    """The front foot path placed where the hind legs can actually reach it.

    ``RobotSpec.paper_gait`` carries the transform because it is a property of
    how a particular robot's hind legs are mounted, not of the paper gait: a
    different quadruped running the same gait needs a different placement, or
    none.  Rotation is about the hip and is applied before the translation.
    """
    p = {**_PAPER_GAIT_DEFAULTS, **spec.paper_gait}
    unknown = set(spec.paper_gait) - set(_PAPER_GAIT_DEFAULTS)
    if unknown:
        raise TypeError(
            f"{spec.name}: unknown paper_gait key(s) {sorted(unknown)}; "
            f"choose from {sorted(_PAPER_GAIT_DEFAULTS)}"
        )
    # Clockwise in the sagittal view (+x forward, +z up) is a negative rotation.
    c, s_ = np.cos(np.radians(-p["hind_rotation_deg"])), np.sin(np.radians(-p["hind_rotation_deg"]))
    R = np.array([[c, -s_], [s_, c]])
    return front_xz @ R.T + np.array([p["hind_dx"], p["hind_dz"]])


def _leg_foot_xz(robot, leg: str, q_thigh: float, q_calf: float) -> np.ndarray:
    """Foot position relative to the leg's hip, in the sagittal (x, z) plane.

    Side joint is held at zero; the leg is planar in body x-z, so the (x, z)
    offset of the foot from the side joint fully describes the foot path.
    """
    q = robot.neutral_config()
    base = robot.n_base_q
    names = robot.actuated_joint_names
    q[base + names.index(f"{leg}_Thigh_joint")] = q_thigh
    q[base + names.index(f"{leg}_Calf_joint")] = q_calf
    robot.forward_kinematics(q)
    hip = np.array(robot.data.oMi[robot.model.getJointId(f"{leg}_Side_joint")].translation)
    foot = np.array(robot.data.oMf[robot.foot_frame_ids[leg]].translation)
    return (foot - hip)[[0, 2]]


def _ik_leg(robot, leg: str, target_xz: np.ndarray, seed: np.ndarray) -> np.ndarray:
    """Solve (thigh, calf) so the foot reaches ``target_xz`` (hip-relative).

    Damped least squares with a backtracking line search — robust through the
    leg's workspace-boundary singularities, where it returns the nearest
    reachable configuration.
    """
    q = np.array(seed, dtype=float)
    lam = 1e-4

    def err_norm(qq):
        return np.linalg.norm(_leg_foot_xz(robot, leg, qq[0], qq[1]) - target_xz)

    e = err_norm(q)
    for _ in range(200):
        if e < 1e-11:
            break
        cur = _leg_foot_xz(robot, leg, q[0], q[1])
        err = target_xz - cur
        J = np.zeros((2, 2))
        h = 1e-6
        for j in range(2):
            dq = q.copy()
            dq[j] += h
            J[:, j] = (_leg_foot_xz(robot, leg, dq[0], dq[1]) - cur) / h
        step = np.linalg.solve(J.T @ J + lam * np.eye(2), J.T @ err)
        t = 1.0
        while t > 1e-4 and err_norm(q + t * step) >= e:
            t *= 0.5
        q = q + t * step
        e = err_norm(q)
    return q


def _grid_seed(robot, leg: str, target_xz: np.ndarray) -> np.ndarray:
    """Coarse joint-grid search for an IK seed on the correct branch."""
    best = np.zeros(2)
    best_d = np.inf
    for a in np.radians(np.arange(-100, 101, 5)):
        for b in np.radians(np.arange(-60, 170, 5)):
            d = np.linalg.norm(_leg_foot_xz(robot, leg, a, b) - target_xz)
            if d < best_d:
                best_d = d
                best = np.array([a, b])
    return best


def _periodic_velocity(q: np.ndarray, dt: float) -> np.ndarray:
    """Central-difference velocity of a periodic trajectory (q[:, 0] == q[:, N])."""
    n_pt = q.shape[1]
    period = n_pt - 1
    v = np.zeros_like(q)
    for k in range(n_pt):
        v[:, k] = (q[:, (k + 1) % period] - q[:, (k - 1) % period]) / (2.0 * dt)
    return v


def build_initial_guess(
    dyn: SymbolicDynamics,
    gait: str,
    N: int,
    T_FIXED: float,
    TAU_MAX: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (X_guess, U_guess) for the gait OCP.

    Front-leg joints follow the paper's paddling trajectory directly; hind-leg
    joints are solved so the hind feet trace the same hip-relative path.
    Torques are computed via inverse dynamics on the kinematic trajectory.
    """
    if gait not in GAITS:
        raise ValueError(f"Unknown gait '{gait}'. Choose from {list(GAITS)}")

    robot = dyn.robot
    nq = robot.nq
    nx = nq + robot.nv
    n_act = robot.n_actuated
    dt_val = T_FIXED / N

    q_trim = dyn.find_trim_state()

    pp_ratio = GAITS[gait]
    phase_offsets = _TLPG_OFFSETS if gait == "TLPG50" else _LSPG_OFFSETS
    theta1, theta2 = paper_fourier_trajectory(pp_ratio)

    # ── Step 1: one-cycle front- and hind-leg joint trajectories ─────────
    t_cycle = np.arange(N) / N
    front_thigh = np.radians(theta1(t_cycle) + _THIGH_OFFSET_DEG)
    front_calf = np.radians(_CALF_OFFSET_DEG - theta2(t_cycle))

    front_xz = np.array(
        [_leg_foot_xz(robot, "Front_Left", a, b) for a, b in zip(front_thigh, front_calf)]
    )

    hind_xz = hind_target_path(front_xz, robot.spec)
    moved = not np.allclose(hind_xz, front_xz)
    print(f"  Solving hind-leg IK ({'transformed' if moved else 'matching'} "
          f"front foot path)...")
    hind_thigh = np.zeros(N)
    hind_calf = np.zeros(N)
    seed = _grid_seed(robot, "Hind_Left", hind_xz[0])
    for k in range(N):
        seed = _ik_leg(robot, "Hind_Left", hind_xz[k], seed)
        hind_thigh[k], hind_calf[k] = seed

    # ── Step 2: assemble per-leg joint trajectory with phase offsets ─────
    q_joints = np.zeros((n_act, N + 1))
    for k in range(N + 1):
        t_norm = k / N
        for i, phi_off in enumerate(phase_offsets):
            t_leg = (t_norm - phi_off) % 1.0
            b = i * 3
            is_hind = i >= 2
            thigh_cyc = hind_thigh if is_hind else front_thigh
            calf_cyc = hind_calf if is_hind else front_calf

            q_joints[b, k] = q_trim[7 + b]  # side joint held at trim
            q_joints[b + 1, k] = np.interp(t_leg, t_cycle, thigh_cyc, period=1.0)
            q_joints[b + 2, k] = np.interp(t_leg, t_cycle, calf_cyc, period=1.0)

    v_joints = _periodic_velocity(q_joints, dt_val)
    a_joints = (v_joints[:, 1:] - v_joints[:, :-1]) / dt_val

    # ── Step 3: simulate base DOF with prescribed joints ─────────────────
    print("  Simulating base DOF (prescribed joint kinematics)...")
    q_base_traj, v_base_traj = simulate_base_kinematics(
        dyn, q_joints, v_joints, a_joints, q_trim, dt_val
    )

    # ── Step 4: assemble full state trajectory ───────────────────────────
    X_guess = np.zeros((nx, N + 1))
    for k in range(N + 1):
        q_k = np.concatenate([q_base_traj[:, k], q_joints[:, k]])
        v_k = np.concatenate([v_base_traj[:, k], v_joints[:, k]])
        X_guess[:, k] = np.concatenate([q_k, v_k])

    # ── Step 5: torque guess via inverse dynamics ────────────────────────
    U_guess = np.zeros((n_act, N))
    for k in range(N):
        a_k = (X_guess[nq:, k + 1] - X_guess[nq:, k]) / dt_val
        tau_id = dyn.eval_inverse_dynamics(X_guess[:nq, k], X_guess[nq:, k], a_k)
        U_guess[:, k] = np.clip(tau_id[6:], -TAU_MAX, TAU_MAX)

    return X_guess, U_guess
