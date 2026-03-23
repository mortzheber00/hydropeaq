"""
CasADi-symbolic rigid-body + hydrodynamic dynamics for the AMPH quadruped.

Uses ``pinocchio.casadi`` to build symbolic expressions for the standard
rigid-body terms (mass matrix, Coriolis, gravity, Jacobians, FK) and adds
the hydrodynamic forces (buoyancy, drag, added mass) on top.

The final equations of motion are:

    [M_rb(q) + M_A(q)] * q̈ + [C_rb(q, q̇) + C_A(q, q̇)] * q̇
        + g_rb(q) = τ + τ_hydro(q, q̇)

where τ_hydro collects buoyancy, drag, and pressure-gradient forces
projected into joint space via link Jacobians.

All public functions return CasADi ``ca.Function`` objects that can be
evaluated numerically or embedded in an NLP.
"""

from __future__ import annotations

import casadi as ca
import numpy as np
import pinocchio as pin
import pinocchio.casadi as cpin

from .robot import QuadrupedRobot
from .hydrodynamics import RHO_WATER, GRAVITY


class SymbolicDynamics:
    """CasADi-symbolic dynamics for the AMPH quadruped.

    Parameters
    ----------
    robot : QuadrupedRobot
        Pinocchio-backed robot (must have cylinders built already).
    rho : float
        Fluid density [kg/m^3].
    Cd_t, Cd_a : float
        Transverse / axial drag coefficients.
    Ca_t, Ca_a : float
        Transverse / axial added-mass coefficients.
    """

    def __init__(
        self,
        robot: QuadrupedRobot,
        rho: float = RHO_WATER,
        Cd_t: float = 1.0,
        Cd_a: float = 0.8,
        Ca_t: float = 1.0,
        Ca_a: float = 0.1,
    ):
        self.robot = robot
        self.nq = robot.nq # number of configuration variables
        self.nv = robot.nv # number of velocity variables

        # Cast the Pinocchio model to CasADi
        self.cmodel = cpin.Model(robot.model)
        self.cdata = self.cmodel.createData()

        # Hydro parameters
        self.rho = rho
        self.Cd_t = Cd_t
        self.Cd_a = Cd_a
        self.Ca_t = Ca_t
        self.Ca_a = Ca_a

        # Symbolic state variables
        self.q = ca.SX.sym("q", self.nq)
        self.v = ca.SX.sym("v", self.nv)   # q̇
        self.a = ca.SX.sym("a", self.nv)   # q̈
        self.tau = ca.SX.sym("tau", self.nv)

        # Pre-compute all symbolic expressions and wrap as CasADi Functions
        self._build_rigid_body_functions()
        self._build_fk_functions()
        self._build_hydro_functions()
        # compute full equations of motion and wrap forward/inverse dynamics
        self._build_eom()

    # ==================================================================
    # Rigid-body dynamics
    # ==================================================================

    def _build_rigid_body_functions(self):
        """Build M_rb(q), C_rb(q, v), g_rb(q) as CasADi Functions."""
        q, v = self.q, self.v

        # Mass matrix M_rb(q) [nv x nv] computed via Composite Rigid Body Algorithm
        # ca.Function("name", [inputs], [outputs], ["input_names"], ["output_names"])
        M_expr = cpin.crba(self.cmodel, self.cdata, q)
        self.f_M_rb = ca.Function("M_rb", [q], [M_expr], ["q"], ["M"])

        # Coriolis matrix  C_rb(q, v)  [nv x nv]
        # computeCoriolisMatrix with symbolic expressions
        cpin.computeCoriolisMatrix(self.cmodel, self.cdata, q, v)
        C_expr = self.cdata.C
        self.f_C_rb = ca.Function("C_rb", [q, v], [C_expr], ["q", "v"], ["C"])

        # Gravity vector  g_rb(q)  [nv x 1]
        g_expr = cpin.computeGeneralizedGravity(self.cmodel, self.cdata, q)
        self.f_g_rb = ca.Function("g_rb", [q], [g_expr], ["q"], ["g"])

    # ==================================================================
    # Forward kinematics & Jacobians
    # ==================================================================

    def _build_fk_functions(self):
        """Build FK positions, velocities, and Jacobians for every link.

        Creates:
          - f_fk[link_name] : (q) → (pos_3x1, R_3x3)
          - f_Jv[link_name] : (q) → J_3xnv  (translational Jacobian)
          - f_J[link_name]  : (q) → J_6xnv  (full spatial Jacobian)
          - f_foot_pos[leg] : (q) → pos_3x1  (foot position)
        """
        q, v = self.q, self.v

        # Run symbolic FK (needed before frame queries)
        cpin.forwardKinematics(self.cmodel, self.cdata, q, v, ca.SX.zeros(self.nv))
        cpin.updateFramePlacements(self.cmodel, self.cdata)

        self.f_fk: dict[str, ca.Function] = {}
        self.f_Jv: dict[str, ca.Function] = {}
        self.f_J: dict[str, ca.Function] = {}

        for name, link in self.robot.links.items():
            fid = link.frame_id
            oMf = self.cdata.oMf[fid]

            pos = oMf.translation     # (3, 1)
            R = oMf.rotation          # (3, 3)

            self.f_fk[name] = ca.Function(
                f"fk_{name}", [q], [pos, R], ["q"], ["pos", "R"],
            )

            J_full = cpin.computeFrameJacobian(
                self.cmodel, self.cdata, q, fid, pin.ReferenceFrame.WORLD,
            )
            # Translational part: rows 0-2
            J_trans = J_full[:3, :]

            self.f_J[name] = ca.Function(
                f"J_{name}", [q], [J_full], ["q"], ["J"],
            )
            self.f_Jv[name] = ca.Function(
                f"Jv_{name}", [q], [J_trans], ["q"], ["Jv"],
            )

        # Foot positions (convenience)
        self.f_foot_pos: dict[str, ca.Function] = {}
        for leg, fid in self.robot.foot_frame_ids.items():
            pos = self.cdata.oMf[fid].translation
            self.f_foot_pos[leg] = ca.Function(
                f"foot_{leg}", [q], [pos], ["q"], ["pos"],
            )

    # ==================================================================
    # Hydrodynamic forces (symbolic)
    # ==================================================================

    def _build_hydro_functions(self):
        """Build CasADi symbolic expressions for hydrodynamic forces.

        For each link with a cylinder:
          - Buoyancy: constant upward force, projected into joint space via J^T
          - Drag: quadratic drag decomposed into axial/transverse components
          - Added mass: configuration-dependent added inertia in joint space

        The cylinder geometry (radius, length) is frozen at the values
        computed by ``robot.build_cylinders()``.  The cylinder *axis* is
        recomputed symbolically from FK so that it tracks the current q.
        """
        q, v = self.q, self.v
        rho, g = self.rho, GRAVITY

        # Accumulate joint-space contributions
        tau_buoyancy = ca.SX.zeros(self.nv, 1)
        tau_drag = ca.SX.zeros(self.nv, 1)
        M_added = ca.SX.zeros(self.nv, self.nv)

        for link in self.robot.links.values():
            cyl = link.cylinder
            if cyl is None:
                continue

            fid = link.frame_id
            oMf = self.cdata.oMf[fid]
            R_sym = oMf.rotation   # 3x3 symbolic rotation

            # Jacobian (translational part) for this link
            J_full = cpin.computeFrameJacobian(
                self.cmodel, self.cdata, q, fid, pin.ReferenceFrame.WORLD,
            )
            Jv = J_full[:3, :]  # (3, nv)

            # Link CoM velocity in world frame
            v_link = Jv @ v     # (3, 1)

            # ── Cylinder axis in world frame (symbolic) ──
            # The local axis was computed at build time; rotate it
            # by the current frame rotation.
            # Cylinder axis in world frame: rotate the body-frame axis by
            # the current (symbolic) link rotation.
            axis_sym = R_sym @ ca.SX(cyl.axis_local)
            axis_sym = axis_sym / (ca.norm_2(axis_sym) + 1e-15)

            r = cyl.radius
            L = cyl.length
            V = cyl.volume
            A_t = cyl.cross_section_transverse
            A_a = cyl.cross_section_axial

            # ── Buoyancy ──
            F_buoy = ca.SX([0.0, 0.0, rho * g * V])
            tau_buoyancy += Jv.T @ F_buoy

            # ── Viscous drag ──
            # Decompose velocity into axial and transverse
            v_ax_mag = ca.dot(v_link, axis_sym)
            v_ax = v_ax_mag * axis_sym
            v_tr = v_link - v_ax

            F_drag_ax = -0.5 * rho * self.Cd_a * A_a * ca.fabs(v_ax_mag) * v_ax
            v_tr_mag = ca.norm_2(v_tr)
            F_drag_tr = -0.5 * rho * self.Cd_t * A_t * v_tr_mag * v_tr
            F_drag = F_drag_ax + F_drag_tr
            tau_drag += Jv.T @ F_drag

            # ── Added mass (joint-space) ──
            # M_A = J^T * M_A_cartesian * J
            # M_A_cartesian = ma_t * I + (ma_a - ma_t) * (a ⊗ a)
            ma_t = self.Ca_t * rho * V
            ma_a = self.Ca_a * rho * V
            a_col = axis_sym  # already (3,1)
            M_A_cart = ma_t * ca.SX.eye(3) + (ma_a - ma_t) * (a_col @ a_col.T)
            M_added += Jv.T @ M_A_cart @ Jv

        # ── Wrap as CasADi Functions ──
        self.f_tau_buoyancy = ca.Function(
            "tau_buoyancy", [q], [tau_buoyancy], ["q"], ["tau"],
        )
        self.f_tau_drag = ca.Function(
            "tau_drag", [q, v], [tau_drag], ["q", "v"], ["tau"],
        )
        self.f_M_added = ca.Function(
            "M_added", [q], [M_added], ["q"], ["M"],
        )

    # ==================================================================
    # Full equations of motion
    # ==================================================================

    def _build_eom(self):
        """Assemble the complete EoM as a CasADi Function.

        [M_rb(q) + M_A(q)] * q̈ + C_rb(q,q̇) * q̇ + g_rb(q)
            = τ + τ_buoyancy(q) + τ_drag(q, q̇)

        The added-mass Coriolis term C_A is neglected (small for slow motion).

        Rearranged to give q̈:
            q̈ = M_total^{-1} * (τ + τ_buoyancy + τ_drag - C_rb*q̇ - g)
        """
        q, v, tau = self.q, self.v, self.tau

        M_rb = self.f_M_rb(q)
        C_rb = self.f_C_rb(q, v)
        g_rb = self.f_g_rb(q)
        M_A = self.f_M_added(q)
        tau_b = self.f_tau_buoyancy(q)
        tau_d = self.f_tau_drag(q, v)

        M_total = M_rb + M_A
        rhs = tau + tau_b + tau_d - C_rb @ v - g_rb

        # Forward dynamics: q̈ = M_total \ rhs
        a_expr = ca.solve(M_total, rhs)

        self.f_forward_dynamics = ca.Function(
            "forward_dynamics",
            [q, v, tau], [a_expr],
            ["q", "v", "tau"], ["a"],
        )

        # Inverse dynamics: τ = M_total * q̈ + C*q̇ + g - τ_hydro
        a = self.a
        M_rb2 = self.f_M_rb(q)
        C_rb2 = self.f_C_rb(q, v)
        g_rb2 = self.f_g_rb(q)
        M_A2 = self.f_M_added(q)
        tau_b2 = self.f_tau_buoyancy(q)
        tau_d2 = self.f_tau_drag(q, v)

        tau_id = (M_rb2 + M_A2) @ a + C_rb2 @ v + g_rb2 - tau_b2 - tau_d2

        self.f_inverse_dynamics = ca.Function(
            "inverse_dynamics",
            [q, v, a], [tau_id],
            ["q", "v", "a"], ["tau"],
        )

        # State-space form for ODE integration.
        # The EoM is second-order (M·q̈ = ...), but integrators expect first-order.
        #
        #   ẋ = [ dq/dt  ]  =  [           q̇              ]
        #       [ dq̇/dt  ]     [ M_total⁻¹·(τ+τ_hydro-C·q̇-g) ]
        x = ca.vertcat(q, v)
        u = tau
        xdot = ca.vertcat(v, a_expr)

        # given x = [q, v] and u = tau, return xdot = [q̇, q̈]
        self.f_xdot = ca.Function(
            "xdot",
            [x, u], [xdot],
            ["x", "u"], ["xdot"],
        )

    # ==================================================================
    # Public convenience methods
    # ==================================================================

    def eval_forward_dynamics(
        self, q: np.ndarray, v: np.ndarray, tau: np.ndarray,
    ) -> np.ndarray:
        """Evaluate forward dynamics numerically."""
        return np.array(self.f_forward_dynamics(q, v, tau)).flatten()

    def eval_inverse_dynamics(
        self, q: np.ndarray, v: np.ndarray, a: np.ndarray,
    ) -> np.ndarray:
        """Evaluate inverse dynamics numerically."""
        return np.array(self.f_inverse_dynamics(q, v, a)).flatten()

    def print_summary(self):
        """Print a summary of the symbolic dynamics."""
        print("=== Symbolic Dynamics Summary ===")
        print(f"  State dimension:   nq={self.nq}, nv={self.nv}")
        print(f"  Links with hydro:  {sum(1 for l in self.robot.links.values() if l.cylinder)}")
        print(f"  Fluid density:     {self.rho} kg/m^3")
        print(f"  Drag coeffs:       Cd_t={self.Cd_t}, Cd_a={self.Cd_a}")
        print(f"  Added-mass coeffs: Ca_t={self.Ca_t}, Ca_a={self.Ca_a}")
        print()
        print("CasADi Functions:")
        for attr_name in sorted(dir(self)):
            if attr_name.startswith("f_") and isinstance(getattr(self, attr_name), ca.Function):
                fn = getattr(self, attr_name)
                print(f"  {fn.name():25s}  {fn.size_in(0)} → {fn.size_out(0)}")
