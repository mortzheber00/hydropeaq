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

# Number of midpoint strips per cylinder for transverse drag integration.
# N=1 (single midpoint) underestimates the rotational drag moment by 50%.
# N=5 closes the gap to ~2%. Cost: 5× more symbolic terms in drag.
_DRAG_N_STRIPS = 2


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
        Quadratic form-drag coefficients for crossflow / axial flow.
    Cd_lin_transverse, Cd_lin_axial : float, optional
        Independent *linear* (skin-friction / potential) damping coefficients
        — Fossen's D_S term — decoupled from the quadratic Cd.  Default None
        falls back to the corresponding Cd, preserving prior behaviour.
        Linear damping force = 0.5·rho·Cd_lin·A·v_linear_threshold · v.
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
        Cd_transverse: float = 0.7,
        Cd_axial: float = 0.275,
        Cd_lin_transverse: float = 0.1,
        Cd_lin_axial: float = 0.5,
        Ca_transverse: float = 1.3,
        Ca_axial: float = 0.3,
        z_surface: float = 0.0,
        v_linear_threshold: float = 0.7,
        leg_thrust_scale: float = 1.0,
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
        # Linear damping coeffs default to the form-drag Cd (prior behaviour).
        self.Cd_lin_transverse = (
            Cd_transverse if Cd_lin_transverse is None else Cd_lin_transverse
        )
        self.Cd_lin_axial = Cd_axial if Cd_lin_axial is None else Cd_lin_axial
        self.Ca_transverse = Ca_transverse
        self.Ca_axial = Ca_axial
        self.z_surface = float(z_surface)
        self.v_linear_threshold = float(v_linear_threshold)
        self.leg_thrust_scale = float(leg_thrust_scale)

        self.f_tau_buoyancy: ca.Function | None = None
        self.f_tau_drag: ca.Function | None = None
        self.f_M_added: ca.Function | None = None
        self.M_added_expr: ca.SX | None = None  # set in build(), used by SymbolicDynamics

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
        Decomposed into axial and transverse components. Single-point sample
        — for distributed integration along the cylinder length see
        ``drag_wrench`` (used in ``build``).
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
        D1_a = 0.5 * rho * self.Cd_lin_axial * A_a * v_thresh
        v_ax_abs = ca.sqrt(v_ax_mag**2 + _EPS)
        F_drag_ax = -(
            D1_a * v_ax + 0.5 * rho * self.Cd_axial * A_a * v_ax_abs * v_ax
        )

        # Transverse drag (smooth norm)
        D1_t = 0.5 * rho * self.Cd_lin_transverse * A_t * v_thresh
        v_tr_mag = ca.sqrt(ca.dot(v_tr, v_tr) + _EPS)
        F_drag_tr = -(
            D1_t * v_tr + 0.5 * rho * self.Cd_transverse * A_t * v_tr_mag * v_tr
        )

        return alpha * (F_drag_ax + F_drag_tr)

    def drag_wrench(
        self,
        cyl: CylinderPrimitive,
        alpha: ca.SX,
        axis_sym: ca.SX,
        J_full: ca.SX,
        R_sym: ca.SX,
        n_strips: int = _DRAG_N_STRIPS,
    ) -> ca.SX:
        """6D drag wrench (force, moment about frame origin) in world frame.

        Splits the cylinder into ``n_strips`` equal slices along its axis.
        For each strip, the transverse drag is evaluated with the velocity at
        the strip's midpoint and an area of ``D · L/n_strips``.  The total
        wrench is summed.

        This captures the linear velocity gradient from joint rotation that a
        single-midpoint sample misses (a single sample underestimates the
        transverse rotational moment by 50%; n_strips=5 closes that to ~2%).

        Axial drag uses end-cap area and is applied once at the cylinder
        midpoint (no along-length integration — end-cap form drag isn't
        distributed).
        """
        rho = self.rho
        D = 2.0 * cyl.radius
        L = cyl.length
        A_a = cyl.cross_section_axial
        v_thresh = self.v_linear_threshold

        # Transverse drag — N midpoint strips along the cylinder length.
        # Per-strip transverse area = D · (L / n_strips).
        A_t_strip = D * (L / n_strips)
        D1_t = 0.5 * rho * self.Cd_lin_transverse * A_t_strip * v_thresh
        coef_q_t = 0.5 * rho * self.Cd_transverse * A_t_strip

        F_tr = ca.SX.zeros(3, 1)
        M_tr = ca.SX.zeros(3, 1)

        for i in range(n_strips):
            # Strip midpoint along cylinder axis, in [-L/2, +L/2].
            s = -L / 2.0 + (i + 0.5) * (L / n_strips)

            # World-frame offset from frame origin to this strip midpoint.
            r_s = R_sym @ (
                ca.SX(cyl.center_local) + s * ca.SX(cyl.axis_local)
            )

            # Velocity at strip midpoint via shifted Jacobian.
            Jv_s = J_full[:3, :] - _skew(r_s) @ J_full[3:, :]
            v_s = Jv_s @ self.v

            # Transverse component of strip velocity.
            v_ax_mag_s = ca.dot(v_s, axis_sym)
            v_tr_s = v_s - v_ax_mag_s * axis_sym
            v_tr_mag_s = ca.sqrt(ca.dot(v_tr_s, v_tr_s) + _EPS)

            F_strip = -(D1_t * v_tr_s + coef_q_t * v_tr_mag_s * v_tr_s)
            M_strip = ca.cross(r_s, F_strip)

            F_tr += F_strip
            M_tr += M_strip

        # Axial drag — single midpoint sample (end-cap form drag).
        r_mid = R_sym @ ca.SX(cyl.center_local)
        Jv_mid = J_full[:3, :] - _skew(r_mid) @ J_full[3:, :]
        v_mid = Jv_mid @ self.v
        v_ax_mag = ca.dot(v_mid, axis_sym)
        v_ax = v_ax_mag * axis_sym
        v_ax_abs = ca.sqrt(v_ax_mag ** 2 + _EPS)

        D1_a = 0.5 * rho * self.Cd_lin_axial * A_a * v_thresh
        F_ax = -(
            D1_a * v_ax + 0.5 * rho * self.Cd_axial * A_a * v_ax_abs * v_ax
        )
        M_ax = ca.cross(r_mid, F_ax)

        F = alpha * (F_tr + F_ax)
        M = alpha * (M_tr + M_ax)
        return ca.vertcat(F, M)

    def added_mass_matrix(
        self,
        cyl: CylinderPrimitive,
        alpha: ca.SX,
        axis_sym: ca.SX,
        Jv: ca.SX,
        Jw: ca.SX,
    ) -> ca.SX:
        """Symbolic joint-space added-mass contribution (nv x nv) for one link.

        Two contributions, both anisotropic in the cylinder's body frame:
          translational  M_A_lin  = ma_t·I + (ma_a − ma_t)·(a a^T)
          rotational     M_A_rot  = I_a_perp·I − I_a_perp·(a a^T)
                                  = I_a_perp · (I − a a^T)
        with ma_a = Ca_axial · ρ · V, ma_t = Ca_transverse · ρ · V, and
        I_a_perp = (1/12) · Ca_transverse · ρ · V · L^2 (Schjølberg & Fossen
        1994, eq. for M_A55/M_A66 of a circular cylinder).  Axial rotational
        added mass is zero for a body of revolution about its own axis.

        Joint-space projection:
          M_added_joint = Jv^T (α·M_A_lin) Jv  +  Jw^T (α·M_A_rot) Jw
        """
        rho = self.rho
        V = cyl.volume_displaced
        L = cyl.length

        # Translational
        ma_t = self.Ca_transverse * rho * V
        ma_a = self.Ca_axial * rho * V
        M_A_lin = alpha * (
            ma_t * ca.SX.eye(3) + (ma_a - ma_t) * (axis_sym @ axis_sym.T)
        )

        # Rotational (about axes perpendicular to cylinder axis only)
        I_a_perp = (1.0 / 12.0) * self.Ca_transverse * rho * V * L ** 2
        M_A_rot = alpha * I_a_perp * (
            ca.SX.eye(3) - (axis_sym @ axis_sym.T)
        )

        return Jv.T @ M_A_lin @ Jv + Jw.T @ M_A_rot @ Jw

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

            # Jacobian at the cylinder midpoint (used by added_mass_matrix).
            # The frame origin is at the parent joint (the rotation pivot), so
            # J_full[:3,:] @ v == 0 there for any rotation about that joint.
            # The correct velocity is:  v_mid = v_origin + omega × r_offset
            #   = (J_lin - skew(r_offset) @ J_ang) @ v
            # where r_offset = R @ center_local is the joint-to-midpoint vector.
            r_offset = R_sym @ ca.SX(cyl.center_local)
            Jv = J_full[:3, :] - _skew(r_offset) @ J_full[3:, :]
            Jw = J_full[3:, :]   # angular Jacobian (same for any point on link)

            # Cylinder axis in world frame (symbolic)
            axis_sym = R_sym @ ca.SX(cyl.axis_local)
            axis_sym = axis_sym / (ca.norm_2(axis_sym) + 1e-15)

            # Submersion
            alpha = self.submersion_ratio(cyl, axis_sym, oMf, R_sym)

            # Buoyancy
            wrench_buoy = self.buoyancy_wrench(cyl, alpha, axis_sym, R_sym)
            tau_buoyancy += J_full.T @ wrench_buoy

            # Drag — distributed along the cylinder via N-strip midpoint
            # integration. Scaled on non-trunk links to model wake-induced
            # thrust slip.
            scale = 1.0 if link.name == "base_link" else self.leg_thrust_scale
            wrench_drag = scale * self.drag_wrench(
                cyl, alpha, axis_sym, J_full, R_sym
            )
            tau_drag += J_full.T @ wrench_drag

            # Added mass (translational + rotational)
            M_added += self.added_mass_matrix(cyl, alpha, axis_sym, Jv, Jw)

        # Expose the symbolic M_added expression so SymbolicDynamics can build
        # an added-mass Coriolis term C_A·v via Christoffel symbols on M_A(q).
        self.M_added_expr = M_added

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
