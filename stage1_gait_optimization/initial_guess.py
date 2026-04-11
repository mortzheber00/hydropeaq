"""
Initial guess for the gait OCP — paddling trajectory from Qu et al. 2025.

Qu, J. et al. "Amphibious robotic dog: design, paddling gait planning,
and experimental characterization." Bioinspir. Biomim. 20, 036012 (2025).

Public API:
  build_initial_guess(dyn, gait, N, T_FIXED, D_MIN, TAU_MAX)
      -> X_guess (nx, N+1), U_guess (n_act, N)
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
    D_MIN: float,
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
