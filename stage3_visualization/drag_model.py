#!/usr/bin/env python3
"""Numeric port of the drag model the OCP solves, evaluated link by link.

``hydro_model.hydrodynamics`` builds the drag as CasADi expressions over a whole
robot; a figure wants one link's force at one pose, as floats, thousands of
times.  This is that same model rewritten numerically, and the point is that it
is the *same* model: the transverse drag is integrated over ``_DRAG_N_STRIPS``
midpoint strips along the axis (imported, not restated, so the two cannot drift
apart), Fossen's linear damping term sits beside the quadratic one, the
submersion ratio scales both, and the coefficients are the fitted ones.  Only
the force is ported — ``drag_wrench`` also returns a moment, which no figure
here needs; see ``link_drag_x`` for why that is safe for a forward force.

Getting that agreement was not free.  A hand-rolled version here previously
sampled the transverse drag once at the cylinder midpoint (which underestimates
the rotational drag moment by 50%), left every link fully wetted at all times
(this robot swims at the surface: the hull averages 13% submerged and the front
legs leave the water each cycle), and carried its own ``CD_T = 1.0 / CD_A = 0.1``
from before the 2026-08-16 SPH fit set 3.25 / 0.8.  The front-left leg's cycle
impulse came out 4.6x low.  Every one of those numbers is now imported rather
than restated — the coefficients from ``hydro_params``, which calls itself the
single source of truth and says to change them nowhere else, and the strip count
and epsilons from ``hydrodynamics`` itself.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pinocchio as pin

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "stage1_gait_optimization"))
from stage1_gait_optimization.hydro_model import hydro_params  # noqa: E402
from stage1_gait_optimization.hydro_model.hydrodynamics import (  # noqa: E402
    _DRAG_N_STRIPS,
    _EPS,
    RHO_WATER,
)
from stage1_gait_optimization.hydro_model.robot import QuadrupedRobot  # noqa: E402

from collocation import D_COLLOC, cycle_mean  # noqa: E402

CD_T = hydro_params.CD_T
CD_A = hydro_params.CD_A
CD_LIN_T = hydro_params.CD_LIN_T
CD_LIN_A = hydro_params.CD_LIN_A
V_LIN_THRESH = hydro_params.V_LINEAR_THRESHOLD
RHO = RHO_WATER

# Wake-slip scale the model applies to every non-trunk link at assembly time
# (``hydrodynamics.build``).  It is 1.0 as fitted, so it changes nothing today —
# it is carried here so that setting it does not silently split this port from
# the model it claims to reproduce.
LEG_THRUST_SCALE = hydro_params.LEG_THRUST_SCALE

# The free surface the submersion ratio is measured against, matching
# SymbolicDynamics' own default.
Z_SURFACE = 0.0

# The sagittal-plane figures assume a 3-joint serial leg whose drag is the sum
# over thigh, calf and foot.  The BODY2 analogue -- sweep (theta1, theta2), mask
# by assemblability, sum over the leg's six links -- is a separate piece of work.
SUPPORTED_ROBOTS = ("amph",)

# The three links an amph leg's drag is summed over, in kinematic order.
LEG_SEGMENTS = ("Thigh", "Calf", "Foot")


def leg_links(robot: QuadrupedRobot, leg: str) -> tuple[str, ...]:
    """The links one leg's drag is summed over, in kinematic order.

    amph keeps the three it has always used.  Its fourth leg cylinder, the
    Side link, is a stub on a joint the OCP pins to zero, so it never sweeps
    and has never been counted as leg drag; folding it in now would move every
    published amph number.

    Any other robot takes every segment cylinder whose link carries the leg's
    name — ``QuadrupedRobot.leg_skeleton``'s own rule, so BODY2's six-link
    closed chain needs no list of its own here.
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
    """Fraction of one cylinder below the surface, with the model's smooth clamp.

    Numeric port of ``SymbolicHydrodynamicModel.submersion_ratio``; a link clear
    of the water scales to zero drag here exactly as it does there.
    """
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
    """World-frame drag force on one cylinder — numeric port of ``drag_wrench``.

    ``v_origin`` and ``omega`` are ``(n, 3)`` batches of the link frame's twist,
    so a whole velocity-direction sweep costs one call.  Returns ``(n, 3)``.
    """
    n_strips = _DRAG_N_STRIPS
    L, D = cyl.length, 2.0 * cyl.radius
    F = np.zeros_like(v_origin)

    # Transverse: strip-wise, each with its own midpoint velocity.
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

    # Axial: end-cap form drag, one sample at the midpoint (not distributed).
    A_a = cyl.cross_section_axial
    v_mid = v_origin + np.cross(omega, R @ cyl.center_local)
    v_ax_mag = v_mid @ axis
    v_ax = v_ax_mag[:, None] * axis[None, :]
    D1_a = 0.5 * RHO * CD_LIN_A * A_a * V_LIN_THRESH
    F -= D1_a * v_ax + 0.5 * RHO * CD_A * A_a * np.sqrt(
        v_ax_mag ** 2 + _EPS)[:, None] * v_ax

    return alpha * F


def link_drag_terms(robot: QuadrupedRobot, link_name: str, q: np.ndarray):
    """``(cyl, R, alpha, axis, J)`` for a link at the current pose, or None.

    Pose-only, so a caller sweeping velocities at a fixed pose evaluates it once.
    ``robot.forward_kinematics(q)`` must already have been called.
    """
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
    """World-x drag force on one link.

    The fluid force ON the link opposes its motion, so a backward-sweeping link
    (v_x < 0) gives F_drag[0] > 0 — forward thrust.

    This is the quantity that lands in the base linear rows of ``f_tau_drag``:
    the free-flyer's first three Jacobian columns are the base rotation, so
    ``tau[0:3] = R^T F`` and the drag *moment* — which this port does not carry —
    contributes only to the rotational and joint rows.  Checked against the
    symbolic model's own per-link assembly at every collocation point of the
    reference solution: worst disagreement 7.1e-15 N.
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
    """Cycle-mean world-x drag on one leg [N], optionally with the leg frozen.

    ``hold`` is a per-joint angle vector for this leg's three joints; when
    given, the leg is pinned there with zero joint velocity and everything else
    in the state — the base motion, the other three legs — is left as solved, so
    the difference from the free case is attributable to this leg's own motion.
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

    ``frozen`` holds the leg at its own cycle-mean pose, which is the baseline
    the raw cycle-mean force is missing: a submerged limb costs drag whether or
    not it moves, so zero is the wrong thing to compare a leg against.
    """
    return {leg: (leg_cycle_drag(robot, Xc_leg, nq, N, leg),
                  leg_cycle_drag(robot, Xc_leg, nq, N, leg,
                                 hold=leg_mean_pose(robot, Xc_leg, leg)))
            for leg in robot.spec.leg_names}
