"""
Initial guess for Task 3 OCP — paddling trajectory from Qu et al. 2025.

Qu, J. et al. "Amphibious robotic dog: design, paddling gait planning,
and experimental characterization." Bioinspir. Biomim. 20, 036012 (2025).

Public API:
  build_initial_guess(dyn, gait, N, T_FIXED, D_MIN, TAU_MAX)
      → X_guess (nx, N+1), U_guess (n_act, N)
"""

from __future__ import annotations

import numpy as np

from hydro_model import SymbolicDynamics

# Supported gaits and their power-phase ratios
GAITS = {
    "LSPG25": 0.25,   # lateral-sequence, 25 % power phase
    "LSPG33": 0.33,  # lateral-sequence, 33 % power phase (fastest in paper)
    "TLPG50": 0.50,   # trot-like,        50 % power phase (most stable)
}

# Phase offsets (cycle fraction) per leg [FL, FR, HL, HR].
# Paper notation: LF=FL, RF=FR, LH=HL, RH=HR.
# LSPG: LH=0%, LF=25%, RF=50%, RH=75%
# TLPG: diagonal pairs (LF,RH)=0%, (RF,LH)=50%
_LSPG_OFFSETS = np.array([0.25, 0.50, 0.00, 0.75])
_TLPG_OFFSETS = np.array([0.00, 0.50, 0.50, 0.00])

# Joint-angle mapping amplitudes
_THIGH_AMP = 0.35   # rad — thigh joint swing amplitude
_CALF_AMP  = 0.50   # rad — calf  joint swing amplitude


# ── Fourier trajectory ───────────────────────────────────────────────────────

def paper_fourier_trajectory(pp_ratio: float, n_harmonics: int = 3):
    """Fourier-series leg trajectory from Qu et al. 2025.

    The cycle is defined by 12 keypoints: the four canonical postures
    (PPIP, PPMP, PPEP, RPMP) plus two evenly-spaced intermediate points
    between each consecutive pair.  A 3-harmonic Fourier series is fit by
    least squares (matching the paper's supplementary material method).

    Paper angle convention:
      θ1 - thigh angle measured from the negative swimming axis (70° = vertical).
            Larger = leg leaning forward; smaller = leaning backward.
      θ2 - knee angle (0° = fully extended; larger = more bent).

    Parameters
    ----------
    pp_ratio : float
        Fraction of the cycle spent in the power phase (0.25 / 0.33 / 0.50).
    n_harmonics : int
        Number of Fourier harmonics (paper uses 3).

    Returns
    -------
    alpha1, alpha2 : callables  t → float|array, normalised angle ∈ [-1, 1]
    dalpha1, dalpha2 : callables  t → float|array, d(alpha)/d(t_norm)
    All callables accept normalised cycle time t ∈ [0, 1).
    """
    # 12 keypoints (degrees):
    #   PPIP, int1, int2, PPMP, int3, int4, PPEP=RPIP, int5, int6, RPMP, int7, int8
    theta1_kp = np.array([100, 90, 80, 70, 60, 50, 40, 50, 60, 70, 80, 90], dtype=float)
    theta2_kp = np.array([80, 60, 40, 20, 38.33, 56.67, 75, 86.67, 98.33, 110, 100, 90], dtype=float)

    theta1_mean, theta1_half = 70.0, 30.0   # degrees
    theta2_mean, theta2_half = 65.0, 45.0

    alpha1_kp = (theta1_kp - theta1_mean) / theta1_half  # normalised ∈ [-1, 1]
    alpha2_kp = (theta2_kp - theta2_mean) / theta2_half

    # Keypoint times: 6 evenly spaced in the power phase, 6 in recovery
    n_half = 6
    t_kp = np.concatenate([
        np.linspace(0,        pp_ratio, n_half + 1)[:-1],
        np.linspace(pp_ratio, 1.0,      n_half + 1)[:-1],
    ])

    # Fourier design matrix (constant + n_harmonics cosine/sine pairs)
    n = len(t_kp)
    A = np.ones((n, 1 + 2 * n_harmonics))
    for h in range(1, n_harmonics + 1):
        A[:, 2*h - 1] = np.cos(2 * np.pi * h * t_kp)
        A[:, 2*h]     = np.sin(2 * np.pi * h * t_kp)

    c1, _, _, _ = np.linalg.lstsq(A, alpha1_kp, rcond=None)
    c2, _, _, _ = np.linalg.lstsq(A, alpha2_kp, rcond=None)

    def _eval(t, c):
        t = np.asarray(t, dtype=float)
        out = np.full_like(t, c[0])
        for h in range(1, n_harmonics + 1):
            out += c[2*h - 1] * np.cos(2 * np.pi * h * t)
            out += c[2*h]     * np.sin(2 * np.pi * h * t)
        return out

    def _deval(t, c):
        t = np.asarray(t, dtype=float)
        out = np.zeros_like(t)
        for h in range(1, n_harmonics + 1):
            w = 2 * np.pi * h
            out += -c[2*h - 1] * w * np.sin(w * t)
            out +=  c[2*h]     * w * np.cos(w * t)
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
) -> tuple[np.ndarray, np.ndarray]:
    """Build (X_guess, U_guess) for the Task 3 OCP.

    Kinematics follow the paddling trajectory of Qu et al. 2025.
    Torques are computed via inverse dynamics on the kinematic trajectory.

    Parameters
    ----------
    dyn : SymbolicDynamics
        Fully initialised symbolic dynamics object.
    gait : str
        One of "LSPG25", "LSPG33", "TLPG50".
    N : int
        Number of shooting intervals.
    T_FIXED : float
        Cycle period [s].
    D_MIN : float
        Minimum forward displacement per cycle [m].
    TAU_MAX : float
        Joint torque clipping limit [Nm].

    Returns
    -------
    X_guess : (nx, N+1) ndarray
    U_guess : (n_act, N) ndarray
    """
    if gait not in GAITS:
        raise ValueError(f"Unknown gait '{gait}'. Choose from {list(GAITS)}")

    robot = dyn.robot
    nq, nv = robot.nq, robot.nv
    nx = nq + nv
    n_act = robot.n_actuated
    dt_val = T_FIXED / N

    q_trim = dyn.find_trim_state()
    v_trim = np.zeros(nv)

    pp_ratio = GAITS[gait]
    phase_offsets = _TLPG_OFFSETS if gait == "TLPG50" else _LSPG_OFFSETS
    alpha1, alpha2, dalpha1, dalpha2 = paper_fourier_trajectory(pp_ratio)

    # Joint-angle mapping:
    #   α1 = +1 (θ1=100°, leg forward) → front thigh at NEGATIVE angle
    #   α2 = +1 (θ2=110°, knee bent)   → front calf  at POSITIVE angle
    #   Hind legs have the sagittal plane flipped (side-joint rpy=0 0 π),
    #   handled by s = -1 (same convention throughout the codebase).
    X_guess = np.zeros((nx, N + 1))
    for k in range(N + 1):
        t_norm = k / N
        q_k, v_k = q_trim.copy(), v_trim.copy()
        q_k[0] = D_MIN * k / N       # uniform forward progress
        v_k[0] = D_MIN / T_FIXED     # body-frame forward speed

        for i, phi_off in enumerate(phase_offsets):
            t_leg = (t_norm + phi_off) % 1.0
            a1  = float(alpha1(t_leg))
            a2  = float(alpha2(t_leg))
            da1 = float(dalpha1(t_leg)) / T_FIXED   # normalised → rad/s
            da2 = float(dalpha2(t_leg)) / T_FIXED

            s = -1 if i >= 2 else 1   # hind-leg sagittal-frame sign flip
            b = i * 3

            q_k[7 + b] = q_trim[7 + b]                        # side: no abduction
            q_k[8 + b] = q_trim[8 + b] - s * a1 * _THIGH_AMP  # thigh
            q_k[9 + b] = q_trim[9 + b] + s * a2 * _CALF_AMP   # calf

            v_k[6 + b] = 0.0
            v_k[7 + b] = -s * da1 * _THIGH_AMP
            v_k[8 + b] =  s * da2 * _CALF_AMP

        X_guess[:, k] = np.concatenate([q_k, v_k])

    # Torque guess via inverse dynamics
    U_guess = np.zeros((n_act, N))
    for k in range(N):
        v_next = X_guess[nq:, min(k + 1, N)]
        a_k = (v_next - X_guess[nq:, k]) / dt_val
        tau_id = dyn.eval_inverse_dynamics(X_guess[:nq, k], X_guess[nq:, k], a_k)
        U_guess[:, k] = np.clip(tau_id[6:], -TAU_MAX, TAU_MAX)

    return X_guess, U_guess
