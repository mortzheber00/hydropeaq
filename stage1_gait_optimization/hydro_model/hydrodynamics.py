"""CasADi hydrodynamic model with each link approximated as a cylinder.

Per link: buoyancy, linear + quadratic drag (Fossen 2011) and added mass,
all scaled by a smooth submersion ratio alpha in [0, 1]. Non-smooth functions
(abs, norm, clamp) are smoothed for differentiability.

References
----------
- Fossen, T. I. (2011). *Handbook of Marine Craft Hydrodynamics and
  Motion Control*. Wiley.
- Newman, J. N. (1977). *Marine Hydrodynamics*. MIT Press.
"""

from __future__ import annotations

import casadi as ca
import numpy as np
import pinocchio as pin
import pinocchio.casadi as cpin

from . import hydro_params
from .robot import CylinderPrimitive, QuadrupedRobot

RHO_WATER = 997.0  # freshwater density [kg/m^3]
GRAVITY = 9.81  # gravitational acceleration [m/s^2]

# Smoothing epsilon for abs and norm.
_EPS = 1e-8

# Strips per cylinder for the transverse drag integral. One strip underestimates
# the rotational drag moment by 50 %, five get within ~2 %; cost grows linearly.
_DRAG_N_STRIPS = 2


def _skew(v: ca.SX) -> ca.SX:
    """3×3 skew-symmetric matrix such that _skew(a) @ b == a × b."""
    return ca.vertcat(
        ca.horzcat(     0, -v[2],  v[1]),
        ca.horzcat(  v[2],     0, -v[0]),
        ca.horzcat( -v[1],  v[0],     0),
    )


class SymbolicHydrodynamicModel:
    """Symbolic hydrodynamic forces of the whole robot.

    The per-link methods return CasADi expressions; ``build`` (called on
    construction) sums them in joint space and exposes ``f_tau_buoyancy``,
    ``f_tau_drag``, ``f_M_added`` and ``f_tau_added``.

    Parameters
    ----------
    robot : QuadrupedRobot
        Pinocchio-backed robot with cylinder-approximated links.
    cmodel : pinocchio.casadi.Model
    cdata : pinocchio.casadi.Data
    q_sym : ca.SX (nq,) configuration symbol
    v_sym : ca.SX (nv,) velocity symbol
    nv : int
    a_sym : ca.SX (nv,) acceleration symbol, optional
        Used by ``f_tau_added``; created if omitted.
    rho : float
        Fluid density [kg/m^3].
    Cd_transverse, Cd_axial : float
        Quadratic form-drag coefficients for crossflow / axial flow.
    Cd_lin_transverse, Cd_lin_axial : float or None
        Linear damping coefficients (Fossen's D_S); None uses the quadratic Cd.
        Force = 0.5·rho·Cd_lin·A·v_linear_threshold · v.
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
        a_sym: ca.SX | None = None,
        rho: float = RHO_WATER,
        Cd_transverse: float = hydro_params.CD_T,
        Cd_axial: float = hydro_params.CD_A,
        Cd_lin_transverse: float = hydro_params.CD_LIN_T,
        Cd_lin_axial: float = hydro_params.CD_LIN_A,
        Ca_transverse: float = hydro_params.CA_T,
        Ca_axial: float = hydro_params.CA_A,
        z_surface: float = 0.0,
        v_linear_threshold: float = hydro_params.V_LINEAR_THRESHOLD,
        leg_thrust_scale: float = hydro_params.LEG_THRUST_SCALE,
    ):
        self.robot = robot
        self.cmodel = cmodel
        self.cdata = cdata
        self.q = q_sym
        self.v = v_sym
        self.a = ca.SX.sym("a", nv) if a_sym is None else a_sym
        self.nv = nv

        self.rho = rho
        self.Cd_transverse = Cd_transverse
        self.Cd_axial = Cd_axial
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
        self.f_tau_added: ca.Function | None = None

        self.build()

    # --- Per-link terms ---

    def submersion_ratio(
        self,
        cyl: CylinderPrimitive,
        axis_sym: ca.SX,
        oMf,
        R_sym: ca.SX,
    ) -> ca.SX:
        """Submerged fraction alpha(q) in [0, 1] of one cylinder (C^1-smooth)."""
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
        # Smooth clamp01(x) = max(0, x) - max(0, x - 1) with
        # max(0, x) ~ 0.5 * (sqrt(x^2 + eps) + x); eps sets the corner radius.
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
        """Buoyancy wrench [F; r x F] about the frame origin.

        For partial submersion the centre of buoyancy moves from the midpoint
        toward the submerged end by (alpha - 1)/2 * L.
        """
        F_buoy = ca.vertcat(0.0, 0.0, alpha * self.rho * GRAVITY * cyl.volume_displaced)

        # Smooth sign of the axis z-component: which end is down (0 when horizontal).
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
        """Drag force F = -(D1*v + 0.5*rho*Cd*A*|v|*v) at a single point (world frame).

        ``build`` uses the strip-integrated ``drag_wrench`` instead.
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
        """Drag wrench (force, moment about the frame origin) in world frame.

        Transverse drag is summed over ``n_strips`` slices, each using its own
        midpoint velocity, which captures the velocity gradient along a rotating
        link. Axial (end-cap) drag is evaluated once at the midpoint.
        """
        rho = self.rho
        D = 2.0 * cyl.radius
        L = cyl.length
        A_a = cyl.cross_section_axial
        v_thresh = self.v_linear_threshold

        # --- Transverse drag, strip-wise ---
        A_t_strip = D * (L / n_strips)
        D1_t = 0.5 * rho * self.Cd_lin_transverse * A_t_strip * v_thresh
        coef_q_t = 0.5 * rho * self.Cd_transverse * A_t_strip

        F_tr = ca.SX.zeros(3, 1)
        M_tr = ca.SX.zeros(3, 1)

        for i in range(n_strips):
            s = -L / 2.0 + (i + 0.5) * (L / n_strips)  # position along the axis

            # Offset from the frame origin to the strip midpoint (world frame)
            r_s = R_sym @ (
                ca.SX(cyl.center_local) + s * ca.SX(cyl.axis_local)
            )

            Jv_s = J_full[:3, :] - _skew(r_s) @ J_full[3:, :]
            v_s = Jv_s @ self.v

            v_ax_mag_s = ca.dot(v_s, axis_sym)
            v_tr_s = v_s - v_ax_mag_s * axis_sym
            v_tr_mag_s = ca.sqrt(ca.dot(v_tr_s, v_tr_s) + _EPS)

            F_strip = -(D1_t * v_tr_s + coef_q_t * v_tr_mag_s * v_tr_s)
            M_strip = ca.cross(r_s, F_strip)

            F_tr += F_strip
            M_tr += M_strip

        # --- Axial drag at the midpoint ---
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
        """Joint-space added-mass matrix (nv x nv) of one link.

        Translational and rotational parts, anisotropic in the cylinder frame:
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
        V = cyl.volume_entrained
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

    def added_mass_force(
        self,
        cyl: CylinderPrimitive,
        alpha: ca.SX,
        fid: int,
        J_local: ca.SX,
    ) -> ca.SX:
        """Joint-space added-mass force τ_A = J^T (M_A·ν̇ + C_A(ν)·ν) of one link.

        Kirchhoff's equations in the link frame at the cylinder midpoint, where
        the added mass is constant:

            F = M_lin·v̇ + ω × (M_lin·v)
            M = M_rot·ω̇ + ω × (M_rot·ω) + v × (M_lin·v)

        with M_lin and M_rot as in ``added_mass_matrix``. This keeps the graph
        much smaller than differentiating M_A(q). ∂α/∂q terms are neglected,
        which is exact for fully submerged links.
        """
        rho = self.rho
        V = cyl.volume_entrained
        L = cyl.length

        ma_t = self.Ca_transverse * rho * V
        ma_a = self.Ca_axial * rho * V
        I_a_perp = (1.0 / 12.0) * self.Ca_transverse * rho * V * L ** 2

        ax = cyl.axis_local / np.linalg.norm(cyl.axis_local)
        ax = ca.SX(ax)
        c_loc = ca.SX(cyl.center_local)

        # Local-frame velocity and acceleration, shifted to the cylinder midpoint.
        v_f = cpin.getFrameVelocity(
            self.cmodel, self.cdata, fid, pin.ReferenceFrame.LOCAL
        )
        a_f = cpin.getFrameAcceleration(
            self.cmodel, self.cdata, fid, pin.ReferenceFrame.LOCAL
        )
        omega = v_f.angular
        domega = a_f.angular
        v_mid = v_f.linear + ca.cross(omega, c_loc)
        a_mid = a_f.linear + ca.cross(domega, c_loc)

        # M_lin·x = ma_t·x + (ma_a − ma_t)·(â·x)·â ;  M_rot·x = I_perp·(x − (â·x)·â)
        Mlin_v = ma_t * v_mid + (ma_a - ma_t) * ca.dot(ax, v_mid) * ax
        Mlin_a = ma_t * a_mid + (ma_a - ma_t) * ca.dot(ax, a_mid) * ax
        Mrot_w = I_a_perp * (omega - ca.dot(ax, omega) * ax)
        Mrot_dw = I_a_perp * (domega - ca.dot(ax, domega) * ax)

        F_A = Mlin_a + ca.cross(omega, Mlin_v)
        M_A = Mrot_dw + ca.cross(omega, Mrot_w) + ca.cross(v_mid, Mlin_v)

        Jv_mid = J_local[:3, :] - _skew(c_loc) @ J_local[3:, :]
        return alpha * (Jv_mid.T @ F_A + J_local[3:, :].T @ M_A)

    # --- Assembly ---

    def build(self):
        """Sum all links and create the CasADi functions.

        Sets:
          - ``self.f_tau_buoyancy`` : (q) -> tau  [nv x 1]
          - ``self.f_tau_drag``     : (q, v) -> tau  [nv x 1]
          - ``self.f_M_added``      : (q) -> M  [nv x nv]
          - ``self.f_tau_added``    : (q, v, a) -> tau  [nv x 1]
        """
        q, v, a, nv = self.q, self.v, self.a, self.nv

        # Frame velocities and accelerations are needed by added_mass_force.
        cpin.forwardKinematics(self.cmodel, self.cdata, q, v, a)
        cpin.updateFramePlacements(self.cmodel, self.cdata)

        tau_buoyancy = ca.SX.zeros(nv, 1)
        tau_drag = ca.SX.zeros(nv, 1)
        M_added = ca.SX.zeros(nv, nv)
        tau_added = ca.SX.zeros(nv, 1)

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

            # Linear Jacobian at the cylinder midpoint; the frame origin sits on
            # the joint axis: v_mid = (J_lin - skew(r_offset) @ J_ang) @ v.
            r_offset = R_sym @ ca.SX(cyl.center_local)
            Jv = J_full[:3, :] - _skew(r_offset) @ J_full[3:, :]
            Jw = J_full[3:, :]

            axis_sym = R_sym @ ca.SX(cyl.axis_local)
            axis_sym = axis_sym / (ca.norm_2(axis_sym) + 1e-15)

            alpha = self.submersion_ratio(cyl, axis_sym, oMf, R_sym)

            wrench_buoy = self.buoyancy_wrench(cyl, alpha, axis_sym, R_sym)
            tau_buoyancy += J_full.T @ wrench_buoy

            # Leg drag is scaled to model wake-induced thrust slip.
            scale = 1.0 if link.name == self.robot.spec.base_link else self.leg_thrust_scale
            wrench_drag = scale * self.drag_wrench(
                cyl, alpha, axis_sym, J_full, R_sym
            )
            tau_drag += J_full.T @ wrench_drag

            M_added += self.added_mass_matrix(cyl, alpha, axis_sym, Jv, Jw)

            J_local = cpin.computeFrameJacobian(
                self.cmodel, self.cdata, q, fid, pin.ReferenceFrame.LOCAL
            )
            tau_added += self.added_mass_force(cyl, alpha, fid, J_local)

        self.f_tau_buoyancy = ca.Function(
            "tau_buoyancy", [q], [tau_buoyancy], ["q"], ["tau"]
        )
        self.f_tau_drag = ca.Function(
            "tau_drag", [q, v], [tau_drag], ["q", "v"], ["tau"]
        )
        self.f_M_added = ca.Function(
            "M_added", [q], [M_added], ["q"], ["M"]
        )
        self.f_tau_added = ca.Function(
            "tau_added", [q, v, a], [tau_added], ["q", "v", "a"], ["tau"]
        )
