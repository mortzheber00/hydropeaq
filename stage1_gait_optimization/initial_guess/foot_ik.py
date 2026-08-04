"""Foot-path inverse kinematics in a robot's actuated coordinates.

``firmware._solve_leg_ik`` and ``paper._ik_leg`` both solve for a leg's *tree*
joints.  A closed-chain robot does not have any to solve for: its tree joints
are dependent coordinates fixed by the loop-closure solution, and writing them
directly pulls the linkage apart.  This module solves for the independent
actuated coordinates instead, composing ``CoordinateMap.expand_numeric`` with
forward kinematics, so the same Cartesian foot path can drive either kind of
robot.

Two things follow from working in the actuated coordinates:

* the leg block may be any width -- three joints for a serial leg, two crank
  angles for one of BODY2's five-bar-plus-parallelogram legs -- so the solve is
  a damped least squares rather than a square inverse;
* the reachable set is bounded by the robot's own assemblability constraint
  rather than by joint limits, so the line search rejects any step that leaves
  it.  ``pose_margin`` reads that constraint straight off ``RobotSpec``, which
  is the same expression ``build_collocation_nlp`` imposes, so a guess kept
  positive here starts inside the OCP's feasible set.

The target is the sagittal ``(x, z)`` pair, not a full 3D position: a planar
leg cannot control its foot's ``y`` at all.  BODY2's is planar to machine
precision -- its foot ``y`` is constant across the whole assemblable band.
"""

from __future__ import annotations

import numpy as np


def pose_margin(robot, theta: np.ndarray) -> float:
    """Smallest of the spec's pose inequalities at ``theta``.

    Returns ``+inf`` for a robot that declares none, which is what makes the
    solver below reduce to an unconstrained damped least squares on a serial
    robot.
    """
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
    """Actuated coordinates for one leg placing its foot at ``target_xz``.

    ``theta`` supplies both the warm start and the other legs' coordinates,
    which matter because ``pose_margin`` is evaluated on the whole vector.
    Only ``theta[leg_slice]`` is solved for.

    Returns ``(theta_leg, residual)``.  The residual is reported rather than
    raised on: a foot path that leaves the reachable set should be visible as a
    tracking error the caller can threshold, not an exception thrown from
    inside a per-timestep loop.
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

        # Backtrack until the step both reduces the error and stays assemblable.
        # Halving rather than rejecting outright matters near the band edge,
        # where the full Gauss-Newton step routinely overshoots out of it while
        # a fraction of the same direction is fine.
        a = 1.0
        while a > 1e-4 and not (admissible(x + a * step) and err_at(x + a * step) < e):
            a *= 0.5
        if a <= 1e-4:
            break

        x = x + a * step
        e = err_at(x)

    return x, e
