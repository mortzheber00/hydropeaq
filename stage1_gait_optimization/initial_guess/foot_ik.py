"""Foot-path inverse kinematics in a robot's actuated coordinates.

Works for closed-chain legs too: damped least squares on the sagittal foot
position ``(x, z)``, rejecting steps that violate the spec's pose constraints
(the same ones the OCP imposes).
"""

from __future__ import annotations

import numpy as np


def pose_margin(robot, theta: np.ndarray) -> float:
    """Smallest pose inequality at ``theta``; ``+inf`` if the spec has none."""
    if robot.spec.pose_constraints is None:
        return np.inf
    _, ineqs = robot.spec.pose_constraints(np.asarray(theta, dtype=float))
    if not ineqs:
        return np.inf
    return min(float(np.asarray(e).ravel()[0]) for e in ineqs)


def foot_xz(robot, theta: np.ndarray, leg: str, q_base: np.ndarray) -> np.ndarray:
    """World-frame ``(x, z)`` of ``leg``'s foot at actuated coordinates ``theta``."""
    q = np.concatenate([q_base, robot.coord_map.expand_numeric(theta)])
    robot.forward_kinematics(q)
    return np.asarray(robot.foot_positions()[leg])[[0, 2]]


def solve_leg_theta(
    robot,
    leg: str,
    theta: np.ndarray,
    leg_slice: slice,
    target_xz: np.ndarray,
    q_base: np.ndarray,
    *,
    margin_floor: float = 0.0,
    tol: float = 1e-9,
    max_iter: int = 100,
) -> tuple[np.ndarray, float]:
    """Solve ``theta[leg_slice]`` so the foot reaches ``target_xz``.

    ``theta`` is the warm start and supplies the other legs' coordinates.
    Returns ``(theta_leg, residual)``; unreachable targets are not an error.
    """
    th = np.array(theta, dtype=float)
    x = th[leg_slice].copy()
    n = len(x)
    step_h = 1e-6
    lam = 1e-6

    def err_at(xx: np.ndarray) -> float:
        t = th.copy()
        t[leg_slice] = xx
        return float(np.linalg.norm(foot_xz(robot, t, leg, q_base) - target_xz))

    def admissible(xx: np.ndarray) -> bool:
        t = th.copy()
        t[leg_slice] = xx
        return pose_margin(robot, t) > margin_floor

    e = err_at(x)
    for _ in range(max_iter):
        if e < tol:
            break
        t = th.copy()
        t[leg_slice] = x
        cur = foot_xz(robot, t, leg, q_base)

        J = np.zeros((2, n))
        for j in range(n):
            xp = x.copy()
            xp[j] += step_h
            t[leg_slice] = xp
            J[:, j] = (foot_xz(robot, t, leg, q_base) - cur) / step_h

        step = np.linalg.solve(J.T @ J + lam * np.eye(n), J.T @ (target_xz - cur))

        # Backtrack until the step reduces the error and stays assemblable.
        a = 1.0
        while a > 1e-4 and not (admissible(x + a * step) and err_at(x + a * step) < e):
            a *= 0.5
        if a <= 1e-4:
            break

        x = x + a * step
        e = err_at(x)

    return x, e
