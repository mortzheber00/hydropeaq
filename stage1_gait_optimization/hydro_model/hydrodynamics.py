"""
CasADi-symbolic hydrodynamic model for an underwater quadruped robot.

Implements four force contributions for each rigid-body link:
  1. **Buoyancy** — Archimedes' principle applied to the submerged cylinder volume.
  2. **Viscous drag** — Hybrid linear+quadratic drag (Fossen 2011) with separate
     axial and transverse drag coefficients, scaled by the submersion ratio.
  3. **Pressure-gradient (Froude-Krylov)** — Force from the ambient
     pressure field on the displaced volume.
  4. **Added (virtual) mass** — Inertia of entrained water surrounding
     each moving link, using strip-theory coefficients for a circular
     cylinder.

Partial submersion is handled via a per-link submersion ratio alpha in [0,1]
that scales all hydrodynamic forces.

Smooth approximations (sqrt(x^2 + eps) instead of abs(x)) are used for
CasADi differentiability.

References
----------
- Fossen, T. I. (2011). *Handbook of Marine Craft Hydrodynamics and
  Motion Control*. Wiley.
- Newman, J. N. (1977). *Marine Hydrodynamics*. MIT Press.
"""

from __future__ import annotations

import casadi as ca
import pinocchio as pin
import pinocchio.casadi as cpin

from .robot import CylinderPrimitive, QuadrupedRobot

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

RHO_WATER = 997.0  # freshwater density [kg/m^3]
GRAVITY = 9.81  # gravitational acceleration [m/s^2]

# Smoothing epsilon for differentiability (abs, norm).
_EPS = 1e-8


def _skew(v: ca.SX) -> ca.SX:
    """3×3 skew-symmetric matrix such that _skew(a) @ b == a × b."""
    return ca.vertcat(
        ca.horzcat(     0, -v[2],  v[1]),
        ca.horzcat(  v[2],     0, -v[0]),
        ca.horzcat( -v[1],  v[0],     0),
    )


# ---------------------------------------------------------------------------
# Symbolic hydrodynamic model
# ---------------------------------------------------------------------------


class SymbolicHydrodynamicModel:
    """CasADi-symbolic hydrodynamic model for the full robot.

    Each public method computes a single hydrodynamic contribution for one
    link, mirroring the structure of a numeric model but returning CasADi
    symbolic expressions.  The ``build`` method loops over all links,
    accumulates joint-space contributions, and wraps the results as
    ``ca.Function`` objects.

    **Important:** call ``cpin.forwardKinematics`` and
    ``cpin.updateFramePlacements`` on (cmodel, cdata, q_sym) before
    calling ``build``.

    Parameters
    ----------
    robot : QuadrupedRobot
        Pinocchio-backed robot with cylinder-approximated links.
    cmodel : pinocchio.casadi.Model
    cdata : pinocchio.casadi.Data (FK already computed)
    q_sym : ca.SX (nq,) configuration symbol
    v_sym : ca.SX (nv,) velocity symbol
    nv : int
    rho : float
        Fluid density [kg/m^3].
    Cd_transverse, Cd_axial : float
        Drag coefficients for crossflow / axial flow.
    Ca_transverse, Ca_axial : float
        Added-mass coefficients.
    z_surface : float
        Water surface height [m].
    v_linear_threshold : float
        Velocity scale for the linear damping term [m/s].
    """

    def __init__(
        self,
        robot: QuadrupedRobot,
        cmodel,
        cdata,
        q_sym: ca.SX,
        v_sym: ca.SX,
        nv: int,
        rho: float = RHO_WATER,
        Cd_transverse: float = 1.0,
        Cd_axial: float = 0.8,
        Ca_transverse: float = 1.0,
        Ca_axial: float = 0.1,
        z_surface: float = 0.0,
        v_linear_threshold: float = 0.005,
    ):
        self.robot = robot
        self.cmodel = cmodel
        self.cdata = cdata
        self.q = q_sym
        self.v = v_sym
        self.nv = nv

        self.rho = rho
        self.Cd_transverse = Cd_transverse
        self.Cd_axial = Cd_axial
        self.Ca_transverse = Ca_transverse
        self.Ca_axial = Ca_axial
        self.z_surface = float(z_surface)
        self.v_linear_threshold = float(v_linear_threshold)

        self.f_tau_buoyancy: ca.Function | None = None
        self.f_tau_drag: ca.Function | None = None
        self.f_M_added: ca.Function | None = None

        self.build()

    # ------------------------------------------------------------------
    # Per-link symbolic expressions
    # ------------------------------------------------------------------

    def submersion_ratio(
        self,
        cyl: CylinderPrimitive,
        axis_sym: ca.SX,
        oMf,
        R_sym: ca.SX,
    ) -> ca.SX:
        """Symbolic submersion ratio alpha(q) in [0, 1] for one cylinder.

        Reconstructs the cylinder midpoint from FK and computes the fraction
        of the cylinder below the water surface using a smooth clamp.

        The clamp uses sqrt(x^2 + eps) approximations of fmax(0, x) so the
        ratio is C^1 everywhere, avoiding the gradient kinks that fmin/fmax
        introduce at alpha = 0 and alpha = 1.
        """
        p_center = oMf.translation + R_sym @ ca.SX(cyl.center_local)
        z_center = p_center[2]

        # Smooth |axis_z| and sqrt(1 - axis_z^2) so dz_half stays C^1.
        axis_z_abs = ca.sqrt(axis_sym[2] ** 2 + _EPS)
        dz_axial = 0.5 * cyl.length * axis_z_abs
        dz_radial = cyl.radius * ca.sqrt(ca.fmax(1.0 - axis_sym[2] ** 2, 0.0) + _EPS)
        dz_half = dz_axial + dz_radial

        z_top = z_center + dz_half
        z_bottom = z_center - dz_half

        x = (self.z_surface - z_bottom) / (z_top - z_bottom + 1e-6)
        # Smooth clamp to [0,1]: clamp01(x) = max(0,x) - max(0,x-1),
        # with max(0,x) ~ 0.5*(sqrt(x^2+e) + x).  Linear on [0,1], saturates
        # outside; eps controls the corner radius.
        eps = 1e-4
        return 0.5 * (
            ca.sqrt(x ** 2 + eps) - ca.sqrt((x - 1.0) ** 2 + eps) + 1.0
        )

    def buoyancy_wrench(
        self,
        cyl: CylinderPrimitive,
        alpha: ca.SX,
        axis_sym: ca.SX,
        R_sym: ca.SX,
    ) -> ca.SX:
        """Symbolic 6D buoyancy wrench [F; r x F] at the frame origin.

        The force acts at the center of buoyancy (COB), which for partial
        submersion lies at the centroid of the submerged volume — shifted from
        the cylinder midpoint toward the submerged end by (alpha-1)/2 * L.

        The "which end is down" factor uses tanh(k*axis_z) instead of a
        smooth-sign, so the shift goes smoothly through zero as the cylinder
        crosses horizontal (no gradient discontinuity at axis_z = 0).
        """
        F_buoy = ca.vertcat(0.0, 0.0, alpha * self.rho * GRAVITY * cyl.volume_displaced)

        # Smooth orientation factor in (-1, 1): -1 if axis points down,
        # +1 if up, 0 (no axial shift) when the cylinder is horizontal.
        down_factor = ca.tanh(10.0 * axis_sym[2])
        cob_local = ca.SX(cyl.center_local) + (
            (alpha - 1.0) / 2.0 * cyl.length * down_factor
        ) * ca.SX(cyl.axis_local)
        r_cob = R_sym @ cob_local
        return ca.vertcat(F_buoy, ca.cross(r_cob, F_buoy))

    def drag_force(
        self,
        cyl: CylinderPrimitive,
        alpha: ca.SX,
        axis_sym: ca.SX,
        v_link: ca.SX,
    ) -> ca.SX:
        """Symbolic hybrid viscous drag force (3D, world frame) for one link.

        Hybrid drag: F = -(D1*v + 0.5*rho*Cd*A*|v|*v) with smooth abs/norm.
        Decomposed into axial and transverse components.
        """
        rho = self.rho
        A_t = cyl.cross_section_transverse
        A_a = cyl.cross_section_axial
        v_thresh = self.v_linear_threshold

        # Decompose velocity
        v_ax_mag = ca.dot(v_link, axis_sym)
        v_ax = v_ax_mag * axis_sym
        v_tr = v_link - v_ax

        # Axial drag (smooth abs)
        D1_a = 0.5 * rho * self.Cd_axial * A_a * v_thresh
        v_ax_abs = ca.sqrt(v_ax_mag**2 + _EPS)
        F_drag_ax = -(
            D1_a * v_ax + 0.5 * rho * self.Cd_axial * A_a * v_ax_abs * v_ax
        )

        # Transverse drag (smooth norm)
        D1_t = 0.5 * rho * self.Cd_transverse * A_t * v_thresh
        v_tr_mag = ca.sqrt(ca.dot(v_tr, v_tr) + _EPS)
        F_drag_tr = -(
            D1_t * v_tr + 0.5 * rho * self.Cd_transverse * A_t * v_tr_mag * v_tr
        )

        return alpha * (F_drag_ax + F_drag_tr)

    def added_mass_matrix(
        self,
        cyl: CylinderPrimitive,
        alpha: ca.SX,
        axis_sym: ca.SX,
        Jv: ca.SX,
    ) -> ca.SX:
        """Symbolic joint-space added-mass contribution (nv x nv) for one link.

        M_added_joint = Jv^T @ (alpha * M_A_cartesian) @ Jv
        where M_A_cartesian = ma_t*I + (ma_a - ma_t)*(a x a^T).
        """
        rho = self.rho
        V = cyl.volume_displaced
        ma_t = self.Ca_transverse * rho * V
        ma_a = self.Ca_axial * rho * V

        M_A_cart = alpha * (
            ma_t * ca.SX.eye(3) + (ma_a - ma_t) * (axis_sym @ axis_sym.T)
        )
        return Jv.T @ M_A_cart @ Jv

    # ------------------------------------------------------------------
    # Build: accumulate over all links and wrap as ca.Function
    # ------------------------------------------------------------------

    def build(self):
        """Loop over all links and build ca.Function objects.

        Sets:
          - ``self.f_tau_buoyancy`` : (q) -> tau  [nv x 1]
          - ``self.f_tau_drag``     : (q, v) -> tau  [nv x 1]
          - ``self.f_M_added``      : (q) -> M  [nv x nv]
        """
        q, v, nv = self.q, self.v, self.nv

        tau_buoyancy = ca.SX.zeros(nv, 1)
        tau_drag = ca.SX.zeros(nv, 1)
        M_added = ca.SX.zeros(nv, nv)

        for link in self.robot.links.values():
            cyl = link.cylinder
            if cyl is None:
                continue

            fid = link.frame_id
            oMf = self.cdata.oMf[fid]
            R_sym = oMf.rotation

            J_full = cpin.computeFrameJacobian(
                self.cmodel,
                self.cdata,
                q,
                fid,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )

            # Velocity at the cylinder midpoint, not the frame origin.
            # The frame origin is at the parent joint (the rotation pivot), so
            # J_full[:3,:] @ v == 0 there for any rotation about that joint.
            # The correct velocity is:  v_mid = v_origin + omega × r_offset
            #   = (J_lin - skew(r_offset) @ J_ang) @ v
            # where r_offset = R @ center_local is the joint-to-midpoint vector.
            r_offset = R_sym @ ca.SX(cyl.center_local)
            Jv = J_full[:3, :] - _skew(r_offset) @ J_full[3:, :]
            v_link = Jv @ v

            # Cylinder axis in world frame (symbolic)
            axis_sym = R_sym @ ca.SX(cyl.axis_local)
            axis_sym = axis_sym / (ca.norm_2(axis_sym) + 1e-15)

            # Submersion
            alpha = self.submersion_ratio(cyl, axis_sym, oMf, R_sym)

            # Buoyancy
            wrench_buoy = self.buoyancy_wrench(cyl, alpha, axis_sym, R_sym)
            tau_buoyancy += J_full.T @ wrench_buoy

            # Drag
            F_drag = self.drag_force(cyl, alpha, axis_sym, v_link)
            wrench_drag = ca.vertcat(F_drag, ca.cross(r_offset, F_drag))
            tau_drag += J_full.T @ wrench_drag

            # Added mass
            M_added += self.added_mass_matrix(cyl, alpha, axis_sym, Jv)

        # Wrap as CasADi Functions
        self.f_tau_buoyancy = ca.Function(
            "tau_buoyancy", [q], [tau_buoyancy], ["q"], ["tau"]
        )
        self.f_tau_drag = ca.Function(
            "tau_drag", [q, v], [tau_drag], ["q", "v"], ["tau"]
        )
        self.f_M_added = ca.Function(
            "M_added", [q], [M_added], ["q"], ["M"]
        )
