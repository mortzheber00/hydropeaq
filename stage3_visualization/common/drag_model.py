#!/usr/bin/env python3
"""Numeric per-link version of the OCP drag model, for fast evaluation in figures.

Mirrors ``hydrodynamics.drag_wrench`` (force only): strip-integrated transverse
drag, linear plus quadratic terms, scaled by the submersion ratio. Coefficients,
strip count and epsilons are imported from the model so the two stay in sync.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pinocchio as pin

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "stage1_gait_optimization"))
from stage1_gait_optimization.hydro_model import hydro_params  # noqa: E402
from stage1_gait_optimization.hydro_model.hydrodynamics import (  # noqa: E402
    _DRAG_N_STRIPS,
    _EPS,
    RHO_WATER,
)
from stage1_gait_optimization.hydro_model.robot import QuadrupedRobot  # noqa: E402

from stage3_visualization.common.collocation import D_COLLOC, cycle_mean  # noqa: E402

CD_T = hydro_params.CD_T
CD_A = hydro_params.CD_A
CD_LIN_T = hydro_params.CD_LIN_T
CD_LIN_A = hydro_params.CD_LIN_A
V_LIN_THRESH = hydro_params.V_LINEAR_THRESHOLD
RHO = RHO_WATER

LEG_THRUST_SCALE = hydro_params.LEG_THRUST_SCALE  # applied to non-base links
Z_SURFACE = 0.0  # same as SymbolicDynamics' default

# The sagittal-plane figures assume amph's 3-joint serial legs.
SUPPORTED_ROBOTS = ("amph",)

# amph leg links included in the leg drag
LEG_SEGMENTS = ("Thigh", "Calf", "Foot")


def leg_links(robot: QuadrupedRobot, leg: str) -> tuple[str, ...]:
    """Links included in one leg's drag, in kinematic order.

    amph: thigh, calf and foot (the Side link is pinned and excluded). Other
    robots: all segment cylinders whose name contains the leg name.
    """
    if robot.spec.name == "amph":
        return tuple(f"{leg}_{s}_link" for s in LEG_SEGMENTS)
    return tuple(cs.link for cs in robot.spec.cylinders
                 if cs.kind == "segment" and leg in cs.link)


def require_supported(robot_name: str, figure: str) -> None:
    """Refuse a robot whose legs this model does not describe."""
    if robot_name not in SUPPORTED_ROBOTS:
        raise NotImplementedError(
            f"{figure} supports {SUPPORTED_ROBOTS}, not {robot_name!r}: its leg "
            f"model assumes a 3-joint serial leg. See the plan, stage3 section.")


def submersion_ratio(cyl, R: np.ndarray, p_origin: np.ndarray,
                     axis: np.ndarray) -> float:
    """Numeric ``SymbolicHydrodynamicModel.submersion_ratio``."""
    z_center = float((p_origin + R @ cyl.center_local)[2])
    axis_z_abs = np.sqrt(axis[2] ** 2 + _EPS)
    dz_half = (0.5 * cyl.length * axis_z_abs
               + cyl.radius * np.sqrt(max(1.0 - axis[2] ** 2, 0.0) + _EPS))
    z_top, z_bottom = z_center + dz_half, z_center - dz_half
    x = (Z_SURFACE - z_bottom) / (z_top - z_bottom + 1e-6)
    eps = 1e-4
    return float(0.5 * (np.sqrt(x ** 2 + eps)
                        - np.sqrt((x - 1.0) ** 2 + eps) + 1.0))


def drag_force(cyl, R: np.ndarray, alpha: float, axis: np.ndarray,
               v_origin: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """World-frame drag force (numeric ``drag_wrench``) for ``(n, 3)`` batches of link twists."""
    n_strips = _DRAG_N_STRIPS
    L, D = cyl.length, 2.0 * cyl.radius
    F = np.zeros_like(v_origin)

    # Transverse, per strip
    A_t = D * (L / n_strips)
    D1_t = 0.5 * RHO * CD_LIN_T * A_t * V_LIN_THRESH
    kq_t = 0.5 * RHO * CD_T * A_t
    for i in range(n_strips):
        s = -L / 2.0 + (i + 0.5) * (L / n_strips)
        r_s = R @ (cyl.center_local + s * cyl.axis_local)
        v_s = v_origin + np.cross(omega, r_s)
        v_ax = (v_s @ axis)[:, None] * axis[None, :]
        v_tr = v_s - v_ax
        mag = np.sqrt((v_tr ** 2).sum(axis=1) + _EPS)[:, None]
        F -= D1_t * v_tr + kq_t * mag * v_tr

    # Axial, at the midpoint
    A_a = cyl.cross_section_axial
    v_mid = v_origin + np.cross(omega, R @ cyl.center_local)
    v_ax_mag = v_mid @ axis
    v_ax = v_ax_mag[:, None] * axis[None, :]
    D1_a = 0.5 * RHO * CD_LIN_A * A_a * V_LIN_THRESH
    F -= D1_a * v_ax + 0.5 * RHO * CD_A * A_a * np.sqrt(
        v_ax_mag ** 2 + _EPS)[:, None] * v_ax

    return alpha * F


def link_drag_terms(robot: QuadrupedRobot, link_name: str, q: np.ndarray):
    """Pose-dependent drag inputs ``(cyl, R, alpha, axis, J)``, or None (call FK first)."""
    link = robot.links.get(link_name)
    if link is None or link.cylinder is None:
        return None
    cyl = link.cylinder
    oMf = robot.data.oMf[link.frame_id]
    R = np.array(oMf.rotation)
    axis = R @ cyl.axis_local
    axis = axis / (np.linalg.norm(axis) + 1e-15)
    alpha = submersion_ratio(cyl, R, np.array(oMf.translation), axis)
    J = pin.computeFrameJacobian(
        robot.model, robot.data, q, link.frame_id,
        pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
    )
    return cyl, R, alpha, axis, J


def link_drag_x(robot: QuadrupedRobot, link_name: str, q: np.ndarray,
                v: np.ndarray) -> float:
    """World-x drag force on one link (positive = forward thrust).

    The moment is not needed: it does not enter the base linear rows of
    ``f_tau_drag``. Matches the symbolic model to ~1e-14 N.
    """
    terms = link_drag_terms(robot, link_name, q)
    if terms is None:
        return 0.0
    cyl, R, alpha, axis, J = terms
    v_origin = (J[:3, :] @ v)[None, :]
    omega = (J[3:, :] @ v)[None, :]
    scale = 1.0 if link_name == robot.spec.base_link else LEG_THRUST_SCALE
    return float(scale * drag_force(cyl, R, alpha, axis, v_origin, omega)[0, 0])


def leg_drag_x(robot: QuadrupedRobot, leg: str, q: np.ndarray,
               v: np.ndarray) -> float:
    """World-x drag summed over one leg's links [N] — see ``leg_links``."""
    return sum(link_drag_x(robot, name, q, v) for name in leg_links(robot, leg))


def _leg_joint_indices(robot: QuadrupedRobot, leg: str) -> list[int]:
    names = list(robot.spec.actuated_joint_names)
    return [names.index(f"{leg}_{s}_joint") for s in ("Side", "Thigh", "Calf")]


def leg_cycle_drag(robot, Xc_leg, nq, N, leg, hold=None, d=D_COLLOC) -> float:
    """Cycle-mean world-x drag on one leg [N].

    With ``hold`` (three joint angles) the leg is frozen there while the rest of
    the state stays as solved.
    """
    idx = _leg_joint_indices(robot, leg)
    trace = np.zeros(N * d)
    for c in range(N * d):
        q = Xc_leg[:nq, c].copy()
        v = np.asarray(Xc_leg[nq:, c], dtype=float).flatten().copy()
        if hold is not None:
            for j, ang in zip(idx, hold):
                q[7 + j] = ang       # 7 base coords precede the joint block
                v[6 + j] = 0.0       # 6 base velocities precede it
        robot.forward_kinematics(q)
        trace[c] = leg_drag_x(robot, leg, q, v)
    return cycle_mean(trace, N)


def leg_mean_pose(robot, Xc_leg, leg) -> np.ndarray:
    """This leg's three joint angles, averaged over the cycle."""
    return np.array([Xc_leg[7 + j, :].mean()
                     for j in _leg_joint_indices(robot, leg)])


def stroke_benefit(robot, Xc_leg, nq, N) -> dict:
    """``{leg: (solved, frozen)}`` cycle-mean forward drag [N].

    ``frozen`` holds the leg at its mean pose, the baseline for the stroke's benefit.
    """
    return {leg: (leg_cycle_drag(robot, Xc_leg, nq, N, leg),
                  leg_cycle_drag(robot, Xc_leg, nq, N, leg,
                                 hold=leg_mean_pose(robot, Xc_leg, leg)))
            for leg in robot.spec.leg_names}
