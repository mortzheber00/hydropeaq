"""Base-DOF simulator for kinematic initial guesses.

Forward-integrates the floating-base dynamics with prescribed joint
kinematics, using RK4 to match the OCP integrator.  Joint kinematics at RK4
sub-stages are obtained by linear interpolation between grid-point values;
the joint acceleration is taken as the forward-difference value on the
interval, matching the finite-difference scheme used to build ``a_joints``.
"""

from __future__ import annotations

import numpy as np
from hydro_model import SymbolicDynamics


def _base_state_dot(
    q_base: np.ndarray,
    v_base: np.ndarray,
    q_joints: np.ndarray,
    v_joints: np.ndarray,
    a_joints: np.ndarray,
    dyn: SymbolicDynamics,
) -> tuple[np.ndarray, np.ndarray]:
    """Time derivatives of the base state at the given configuration.

    Returns
    -------
    q_dot_base : (7,) ndarray
        Base position derivative + quaternion derivative.
    a_base : (6,) ndarray
        Base linear + angular acceleration in body frame, from the
        ``[:6, :]`` block of the full equations of motion.
    """
    q = np.concatenate([q_base, q_joints])
    v = np.concatenate([v_base, v_joints])

    M = np.array(dyn.f_M_rb(q)) + np.array(dyn.f_M_added(q))
    C = np.array(dyn.f_C_rb(q, v))
    g = np.array(dyn.f_g_rb(q)).flatten()
    tb = np.array(dyn.f_tau_buoyancy(q)).flatten()
    td = np.array(dyn.f_tau_drag(q, v)).flatten()

    rhs = tb[:6] + td[:6] - C[:6, :] @ v - g[:6] - M[:6, 6:] @ a_joints
    a_base = np.linalg.solve(M[:6, :6], rhs)

    qx, qy, qz, qw = q_base[3], q_base[4], q_base[5], q_base[6]
    vx, vy, vz = v_base[0], v_base[1], v_base[2]
    wx, wy, wz = v_base[3], v_base[4], v_base[5]

    q_dot_base = np.empty(7)
    q_dot_base[0] = (
        (1 - 2 * (qy**2 + qz**2)) * vx
        + 2 * (qx * qy - qw * qz) * vy
        + 2 * (qx * qz + qw * qy) * vz
    )
    q_dot_base[1] = (
        2 * (qx * qy + qw * qz) * vx
        + (1 - 2 * (qx**2 + qz**2)) * vy
        + 2 * (qy * qz - qw * qx) * vz
    )
    q_dot_base[2] = (
        2 * (qx * qz - qw * qy) * vx
        + 2 * (qy * qz + qw * qx) * vy
        + (1 - 2 * (qx**2 + qy**2)) * vz
    )
    q_dot_base[3] = 0.5 * (qw * wx + qy * wz - qz * wy)
    q_dot_base[4] = 0.5 * (qw * wy + qz * wx - qx * wz)
    q_dot_base[5] = 0.5 * (qw * wz + qx * wy - qy * wx)
    q_dot_base[6] = 0.5 * (-qx * wx - qy * wy - qz * wz)

    return q_dot_base, a_base


def _rk4_base_step(
    q_base: np.ndarray,
    v_base: np.ndarray,
    dt: float,
    q_j_k: np.ndarray,
    v_j_k: np.ndarray,
    q_j_kp1: np.ndarray,
    v_j_kp1: np.ndarray,
    a_j_k: np.ndarray,
    dyn: SymbolicDynamics,
) -> tuple[np.ndarray, np.ndarray]:
    """One RK4 step on the base state across the interval ``[t_k, t_{k+1}]``.

    Joint acceleration is held constant at ``a_j_k`` over the interval,
    matching the forward-difference scheme.  Joint position/velocity at the
    sub-stage midpoint use linear interpolation between the grid-point values.
    """
    q_j_m = 0.5 * (q_j_k + q_j_kp1)
    v_j_m = 0.5 * (v_j_k + v_j_kp1)

    # Substage 1 — at t_k
    k1_q, k1_v = _base_state_dot(q_base, v_base, q_j_k, v_j_k, a_j_k, dyn)

    # Substage 2 — at t_k + dt/2
    q_mid = q_base + 0.5 * dt * k1_q
    q_mid[3:7] /= np.linalg.norm(q_mid[3:7])
    v_mid = v_base + 0.5 * dt * k1_v
    k2_q, k2_v = _base_state_dot(q_mid, v_mid, q_j_m, v_j_m, a_j_k, dyn)

    # Substage 3 — at t_k + dt/2 with k2 state
    q_mid = q_base + 0.5 * dt * k2_q
    q_mid[3:7] /= np.linalg.norm(q_mid[3:7])
    v_mid = v_base + 0.5 * dt * k2_v
    k3_q, k3_v = _base_state_dot(q_mid, v_mid, q_j_m, v_j_m, a_j_k, dyn)

    # Substage 4 — at t_k + dt
    q_end = q_base + dt * k3_q
    q_end[3:7] /= np.linalg.norm(q_end[3:7])
    v_end = v_base + dt * k3_v
    k4_q, k4_v = _base_state_dot(q_end, v_end, q_j_kp1, v_j_kp1, a_j_k, dyn)

    q_new = q_base + (dt / 6.0) * (k1_q + 2 * k2_q + 2 * k3_q + k4_q)
    q_new[3:7] /= np.linalg.norm(q_new[3:7])
    v_new = v_base + (dt / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)

    return q_new, v_new


def simulate_base_kinematics(
    dyn: SymbolicDynamics,
    q_joints: np.ndarray,
    v_joints: np.ndarray,
    a_joints: np.ndarray,
    q_trim: np.ndarray,
    dt: float,
    n_cycles: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate base DOF with prescribed joint trajectory using RK4.

    Runs for up to ``n_cycles`` periods.  Forward velocity is carried over
    between cycles; orientation and lateral position are reset to trim at
    the start of each cycle.  Returns the last (or converged) cycle's base
    trajectory.
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
        q_cycle[:, 0] = q_base
        v_cycle[:, 0] = v_base

        for k in range(N):
            q_base, v_base = _rk4_base_step(
                q_base, v_base, dt,
                q_joints[:, k], v_joints[:, k],
                q_joints[:, k + 1], v_joints[:, k + 1],
                a_joints[:, k],
                dyn,
            )
            q_cycle[:, k + 1] = q_base
            v_cycle[:, k + 1] = v_base

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
