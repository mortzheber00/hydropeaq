"""Closed-form coordinate map for BODY2's parallel legs.

Each leg is driven by two hip angles and closes two loops:

    P2, P5, P6  from rotating the cranks about the fixed hips
    P3          = circle(P2, |P2P3|)  x  circle(P6, |P6P3|)     -> links 1.2, 1.3
    P8          rigid on link 1.3
    P7          = circle(P5, |P5P7|)  x  circle(P8, |P8P7|)     -> links 2.2, 2.3

Joint rotations are returned as ``(cos, sin)`` pairs rather than angles, since
the workspace wraps past +-180 deg and ``atan2`` would put a branch cut inside
it. Both vectors have a known length r, so

    cos = (w0 . w) / r^2        sin = (w0 x w) / r^2

which is smooth and matches Pinocchio's representation of continuous joints.
Velocities follow from ``thetadot = c * sdot - s * cdot``.
"""

from __future__ import annotations

import casadi as ca
import numpy as np

from ..coordinate_map import CoordinateMap

# The six tree joints of a leg
JOINT_KEYS = ("1.1", "1.2", "1.3", "2.1", "2.2", "2.3")

# Floor on h^2 = (H_MIN/2)^2. Lies inside the infeasible region, so the feasible
# set is unchanged, but keeps the sqrt derivative bounded. Keep it below H_MIN^2
# so the kink does not land on the constraint boundary.
_H_FLOOR = 1e-6


def _rot(c, s, p):
    """Rotate the 2-vector ``p`` by the rotation whose pair is ``(c, s)``."""
    return ca.vertcat(c * p[0] - s * p[1], s * p[0] + c * p[1])


def _pair(v0, v, r_sq):
    """``(cos, sin)`` of the rotation from ``v0`` to ``v``; both have length ``sqrt(r_sq)``."""
    c = (v0[0] * v[0] + v0[1] * v[1]) / r_sq
    s = (v0[0] * v[1] - v0[1] * v[0]) / r_sq
    return c, s


def _circle_circle(c1, r1, c2, r2, branch, h_floor=_H_FLOOR):
    """Circle-circle intersection point and squared half-chord ``h_sq``.

    ``h_sq`` is the assemblability margin and is returned unclamped. The sqrt
    is floored so IPOPT iterates outside the feasible band do not produce NaNs.
    """
    d = c2 - c1
    L_sq = d[0] ** 2 + d[1] ** 2
    L = ca.sqrt(L_sq)
    a = (r1 ** 2 - r2 ** 2 + L_sq) / (2 * L)
    h_sq = r1 ** 2 - a ** 2
    u = d / L
    perp = ca.vertcat(-u[1], u[0])
    return c1 + a * u + branch * ca.sqrt(ca.fmax(h_sq, h_floor)) * perp, h_sq


def _solve_leg(q1, q2, leg_data, branch):
    """Rotation pairs of a leg's six tree joints and both loop margins.

    Same geometry as ``leg_linkage_sim.Leg.solve``.
    """
    P = {k: ca.DM(v) for k, v in leg_data["pins"].items()}
    sgn, Ls = leg_data["sgn"], leg_data["lengths"]
    r1, r2 = Ls["P2P3"], Ls["P6P3"]
    r3, r4 = Ls["P5P7"], Ls["P8P7"]

    # Crank body rotations are sgn*q; only sin picks up the sign.
    t1c, t1s = ca.cos(q1), sgn["1.1"] * ca.sin(q1)
    t2c, t2s = ca.cos(q2), sgn["2.1"] * ca.sin(q2)

    P2 = P["P1"] + _rot(t1c, t1s, P["P2"] - P["P1"])
    P5 = P["P4"] + _rot(t2c, t2s, P["P5"] - P["P4"])
    P6 = P["P4"] + _rot(t2c, t2s, P["P6"] - P["P4"])

    P3, h_sq_1 = _circle_circle(P2, r1, P6, r2, branch[0])
    c12, s12 = _pair(P["P3"] - P["P2"], P3 - P2, r1 ** 2)
    c13, s13 = _pair(P["P6"] - P["P3"], P6 - P3, r2 ** 2)

    P8 = P3 + _rot(c13, s13, P["P8"] - P["P3"])
    P7, h_sq_2 = _circle_circle(P5, r3, P8, r4, branch[1])
    c22, s22 = _pair(P["P7"] - P["P5"], P7 - P5, r3 ** 2)
    c23, s23 = _pair(P["P8"] - P["P7"], P8 - P7, r4 ** 2)

    def rel(cc, ss, cp, sp, key):
        """Joint value pair for a child rotation (cc,ss) under a parent (cp,sp)."""
        return (cc * cp + ss * sp, sgn[key] * (ss * cp - cc * sp))

    pairs = {
        "1.1": (ca.cos(q1), ca.sin(q1)),
        "1.2": rel(c12, s12, t1c, t1s, "1.2"),
        "1.3": rel(c13, s13, c12, s12, "1.3"),
        "2.1": (ca.cos(q2), ca.sin(q2)),
        "2.2": rel(c22, s22, t2c, t2s, "2.2"),
        "2.3": rel(c23, s23, c22, s22, "2.3"),
    }
    return pairs, (h_sq_1, h_sq_2)


def feasibility_expr(theta, leg_names, linkage):
    """Squared half-chords of all loops (must stay positive); ``theta`` is (q1, q2) per leg."""
    branch = linkage["branch"]
    out = []
    for i, leg in enumerate(leg_names):
        _, h_sq = _solve_leg(theta[2 * i], theta[2 * i + 1], linkage["legs"][leg], branch)
        out.extend(h_sq)
    return out


class Body2CoordinateMap(CoordinateMap):
    """Maps BODY2's 8 hip angles onto its 24-joint (48-configuration) tree."""

    def __init__(self, robot):
        from .body2 import LEG_NAMES, LINKAGE, urdf_joint

        model = robot.model
        self.n_theta = robot.n_actuated
        self.nq_j = model.nq - 7
        self.nv_j = model.nv - 6
        if self.n_theta != 2 * len(LEG_NAMES):
            raise ValueError(f"expected {2 * len(LEG_NAMES)} actuated joints, got {self.n_theta}")

        th = ca.SX.sym("theta", self.n_theta)
        q = ca.SX.zeros(self.nq_j)
        cos_of = [None] * self.nv_j
        sin_of = [None] * self.nv_j

        # Pinocchio's joint order differs from the spec's; index via the model.
        for i, leg in enumerate(LEG_NAMES):
            pairs, _ = _solve_leg(th[2 * i], th[2 * i + 1],
                                  LINKAGE["legs"][leg], LINKAGE["branch"])
            for key in JOINT_KEYS:
                name = urdf_joint(leg, key)
                joint = model.joints[model.getJointId(name)]
                if joint.nq != 2:
                    raise ValueError(
                        f"{name} has nq={joint.nq}; BODY2's joints must all "
                        f"be continuous (unbounded) for this map's representation"
                    )
                iq, iv = joint.idx_q - 7, joint.idx_v - 6
                c, s = pairs[key]
                q[iq], q[iq + 1] = c, s
                cos_of[iv], sin_of[iv] = c, s

        # thetadot_i = c_i * sdot_i - s_i * cdot_i
        S = ca.vertcat(*[
            cos_of[i] * ca.jacobian(sin_of[i], th) - sin_of[i] * ca.jacobian(cos_of[i], th)
            for i in range(self.nv_j)
        ])

        thd = ca.SX.sym("thd", self.n_theta)
        thdd = ca.SX.sym("thdd", self.n_theta)
        tau = ca.SX.sym("tau", self.nv_j)
        feas = ca.vertcat(*feasibility_expr(th, LEG_NAMES, LINKAGE))

        self._f_q = ca.Function("body2_q", [th], [q])
        self._f_S = ca.Function("body2_S", [th], [S])
        self._f_v = ca.Function("body2_v", [th, thd], [S @ thd])
        # jtimes gives Sdot @ thd without forming dS/dtheta
        self._f_a = ca.Function("body2_a", [th, thd, thdd],
                                [S @ thdd + ca.jtimes(S @ thd, th, thd)])
        self._f_tau = ca.Function("body2_tau", [th, tau], [S.T @ tau])
        self._f_feas = ca.Function("body2_feas", [th], [feas])

    def q_joints(self, theta):
        return self._f_q(theta)

    def v_joints(self, theta, thd):
        return self._f_v(theta, thd)

    def a_joints(self, theta, thd, thdd):
        return self._f_a(theta, thd, thdd)

    def tau_joints(self, theta, tau_tree_j):
        return self._f_tau(theta, tau_tree_j)

    def S(self, theta):
        return self._f_S(theta)

    def feasibility(self, theta):
        return self._f_feas(theta)

    def expand_numeric(self, theta: np.ndarray) -> np.ndarray:
        return np.asarray(self._f_q(theta)).ravel()

    def v_numeric(self, theta: np.ndarray, thd: np.ndarray) -> np.ndarray:
        return np.asarray(self._f_v(theta, thd)).ravel()
