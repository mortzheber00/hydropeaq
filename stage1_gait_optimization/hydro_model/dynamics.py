"""Symbolic rigid-body plus hydrodynamic dynamics (pinocchio.casadi).

    [M_rb(q) + M_A(q)] * qdd + [C_rb(q, qd) + C_A(q, qd)] * qd + g_rb(q)
        = tau + tau_buoyancy(q) + tau_drag(q, qd)

All ``f_*`` attributes are ``ca.Function`` objects.
"""

from __future__ import annotations

import casadi as ca
import numpy as np
import pinocchio as pin
import pinocchio.casadi as cpin

from . import hydro_params
from .coordinate_map import IdentityMap
from .hydrodynamics import RHO_WATER, SymbolicHydrodynamicModel
from .robot import QuadrupedRobot


def fn_name(*parts: str) -> str:
    """Join ``parts`` into a valid CasADi function name (URDF names may contain dots)."""
    raw = "_".join(parts)
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in raw)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")


class SymbolicDynamics:
    """Symbolic dynamics of a swimming robot.

    Parameters
    ----------
    robot : QuadrupedRobot
        Robot with cylinders already built.
    rho : float
        Fluid density [kg/m^3].
    Cd_t, Cd_a : float
        Transverse / axial quadratic drag coefficients.
    Cd_lin_t, Cd_lin_a : float or None
        Linear damping coefficients (Fossen's D_S); None uses Cd_t / Cd_a.
        Fitted together with ``v_linear_threshold``.
    Ca_t, Ca_a : float
        Transverse / axial added-mass coefficients.

    Coefficients default to ``hydro_params``.
    """

    def __init__(
        self,
        robot: QuadrupedRobot,
        rho: float = RHO_WATER,
        Cd_t: float = hydro_params.CD_T,
        Cd_a: float = hydro_params.CD_A,
        Cd_lin_t: float = hydro_params.CD_LIN_T,
        Cd_lin_a: float = hydro_params.CD_LIN_A,
        Ca_t: float = hydro_params.CA_T,
        Ca_a: float = hydro_params.CA_A,
        z_surface: float = 0.0,
        v_linear_threshold: float = hydro_params.V_LINEAR_THRESHOLD,
        leg_thrust_scale: float = hydro_params.LEG_THRUST_SCALE,
    ):
        self.robot = robot
        self.nq = robot.nq
        self.nv = robot.nv

        self.cmodel = cpin.Model(robot.model)
        self.cdata = self.cmodel.createData()

        self.rho = rho
        self.Cd_t = Cd_t
        self.Cd_a = Cd_a
        self.Cd_lin_t = Cd_lin_t
        self.Cd_lin_a = Cd_lin_a
        self.Ca_t = Ca_t
        self.Ca_a = Ca_a
        self.z_surface = z_surface
        self.v_linear_threshold = v_linear_threshold
        self.leg_thrust_scale = leg_thrust_scale

        self.q = ca.SX.sym("q", self.nq)
        self.v = ca.SX.sym("v", self.nv)  # qd
        self.a = ca.SX.sym("a", self.nv)  # qdd
        self.tau = ca.SX.sym("tau", self.nv)

        self._f_reduced_Mb = None  # built lazily
        self._build_rigid_body_functions()
        self._build_fk_functions()
        self._build_hydro_functions()
        self._build_eom()

    # --- Rigid-body dynamics ---

    def _build_rigid_body_functions(self):
        """Build M_rb(q), C_rb(q, v), g_rb(q) as CasADi Functions."""
        q, v = self.q, self.v

        M_expr = cpin.crba(self.cmodel, self.cdata, q)
        self.f_M_rb = ca.Function("M_rb", [q], [M_expr], ["q"], ["M"])

        cpin.computeCoriolisMatrix(self.cmodel, self.cdata, q, v)
        C_expr = self.cdata.C
        self.f_C_rb = ca.Function("C_rb", [q, v], [C_expr], ["q", "v"], ["C"])

        g_expr = cpin.computeGeneralizedGravity(self.cmodel, self.cdata, q)
        self.f_g_rb = ca.Function("g_rb", [q], [g_expr], ["q"], ["g"])

    # --- Forward kinematics and Jacobians ---

    def _build_fk_functions(self):
        """Build per-link FK and Jacobian functions.

        Creates:
          - f_fk[link_name] : (q) -> (pos_3x1, R_3x3)
          - f_Jv[link_name] : (q) -> J_3xnv  (translational Jacobian)
          - f_J[link_name]  : (q) -> J_6xnv  (full spatial Jacobian)
          - f_foot_pos[leg] : (q) -> pos_3x1  (foot position)
        """
        q, v = self.q, self.v

        cpin.forwardKinematics(self.cmodel, self.cdata, q, v, ca.SX.zeros(self.nv))
        cpin.updateFramePlacements(self.cmodel, self.cdata)

        self.f_fk: dict[str, ca.Function] = {}
        self.f_Jv: dict[str, ca.Function] = {}
        self.f_J: dict[str, ca.Function] = {}

        for name, link in self.robot.links.items():
            fid = link.frame_id
            oMf = self.cdata.oMf[fid]

            pos = oMf.translation
            R = oMf.rotation

            self.f_fk[name] = ca.Function(
                fn_name("fk", name), [q], [pos, R], ["q"], ["pos", "R"]
            )

            J_full = cpin.computeFrameJacobian(
                self.cmodel, self.cdata, q, fid, pin.ReferenceFrame.WORLD
            )
            J_trans = J_full[:3, :]

            self.f_J[name] = ca.Function(
                fn_name("J", name), [q], [J_full], ["q"], ["J"]
            )
            self.f_Jv[name] = ca.Function(
                fn_name("Jv", name), [q], [J_trans], ["q"], ["Jv"]
            )

        # Foot = fixed local offset from a frame; must match QuadrupedRobot.foot_positions().
        self.f_foot_pos: dict[str, ca.Function] = {}
        for leg, fid in self.robot.foot_frame_ids.items():
            oMf = self.cdata.oMf[fid]
            pos = oMf.translation
            offset = self.robot.foot_offsets[leg]
            if offset.any():
                pos = pos + oMf.rotation @ ca.DM(offset)
            self.f_foot_pos[leg] = ca.Function(
                fn_name("foot", leg), [q], [pos], ["q"], ["pos"]
            )

    # --- Hydrodynamic forces ---

    def _build_hydro_functions(self):
        """Build the hydrodynamic functions via SymbolicHydrodynamicModel."""
        hydro = SymbolicHydrodynamicModel(
            robot=self.robot,
            cmodel=self.cmodel,
            cdata=self.cdata,
            q_sym=self.q,
            v_sym=self.v,
            a_sym=self.a,
            nv=self.nv,
            rho=self.rho,
            Cd_transverse=self.Cd_t,
            Cd_axial=self.Cd_a,
            Cd_lin_transverse=self.Cd_lin_t,
            Cd_lin_axial=self.Cd_lin_a,
            Ca_transverse=self.Ca_t,
            Ca_axial=self.Ca_a,
            z_surface=self.z_surface,
            v_linear_threshold=self.v_linear_threshold,
            leg_thrust_scale=self.leg_thrust_scale,
        )
        self.f_tau_buoyancy = hydro.f_tau_buoyancy
        self.f_tau_drag = hydro.f_tau_drag
        self.f_M_added = hydro.f_M_added
        self.f_tau_added = hydro.f_tau_added

        # tau_added = M_A·a + C_A·v, so evaluating it at a = 0 gives C_A·v.
        C_A_v = hydro.f_tau_added(self.q, self.v, ca.SX.zeros(self.nv))
        self.f_C_A_v = ca.Function(
            "C_A_v", [self.q, self.v], [C_A_v], ["q", "v"], ["C_A_v"]
        )

    # --- Configuration derivative ---

    def _dq_dt(self, q: ca.SX, v: ca.SX) -> ca.SX:
        """Configuration derivative for a free-flyer base with revolute joints.

        The base twist ``v[0:6]`` is in the body frame:

          dp/dt    = R(q_base) * v_lin
          dquat/dt = 0.5 * q_base x [w_body; 0]
          dq_j/dt  = v_j

        Layout: q = [pos (3); quat xyzw (4); joints], v = [v_lin; w; joint rates].
        """
        qx, qy, qz, qw = q[3], q[4], q[5], q[6]
        vx, vy, vz = v[0], v[1], v[2]
        wx, wy, wz = v[3], v[4], v[5]

        # dp/dt = R_world_body * v_body
        dp = ca.vertcat(
            (1 - 2 * (qy**2 + qz**2)) * vx
            + 2 * (qx * qy - qw * qz) * vy
            + 2 * (qx * qz + qw * qy) * vz,
            2 * (qx * qy + qw * qz) * vx
            + (1 - 2 * (qx**2 + qz**2)) * vy
            + 2 * (qy * qz - qw * qx) * vz,
            2 * (qx * qz - qw * qy) * vx
            + 2 * (qy * qz + qw * qx) * vy
            + (1 - 2 * (qx**2 + qy**2)) * vz,
        )

        # Quaternion kinematics:  dq/dt = 0.5 * q x [w; 0]
        dqx = 0.5 * (qw * wx + qy * wz - qz * wy)
        dqy = 0.5 * (qw * wy + qz * wx - qx * wz)
        dqz = 0.5 * (qw * wz + qx * wy - qy * wx)
        dqw = 0.5 * (-qx * wx - qy * wy - qz * wz)

        return ca.vertcat(dp, dqx, dqy, dqz, dqw, v[6:])

    # --- Equations of motion ---

    def _build_eom(self):
        """Build forward dynamics, inverse dynamics and the ODE ``xdot = f(x, tau)``.

        State x = [q; v], xdot = [dq/dt; vdot] (dq/dt != v for the free-flyer).
        """
        q, v, tau = self.q, self.v, self.tau

        M_rb = self.f_M_rb(q)
        C_rb = self.f_C_rb(q, v)
        g_rb = self.f_g_rb(q)
        M_A = self.f_M_added(q)
        tau_b = self.f_tau_buoyancy(q)
        tau_d = self.f_tau_drag(q, v)
        C_A_v = self.f_C_A_v(q, v)

        M_total = M_rb + M_A
        rhs = tau + tau_b + tau_d - C_rb @ v - C_A_v - g_rb
        #rhs = tau + tau_b + tau_d - C_rb @ v - g_rb

        a_expr = ca.solve(M_total, rhs)

        self.f_forward_dynamics = ca.Function(
            "forward_dynamics",
            [q, v, tau],
            [a_expr],
            ["q", "v", "tau"],
            ["a"],
        )

        # Inverse dynamics; RNEA keeps the graph much smaller than assembling M_rb and C_rb.
        a = self.a
        tau_id = (
            cpin.rnea(self.cmodel, self.cdata, q, v, a)
            + self.f_tau_added(q, v, a)
            - self.f_tau_buoyancy(q)
            - self.f_tau_drag(q, v)
        )

        self.f_inverse_dynamics = ca.Function(
            "inverse_dynamics",
            [q, v, a],
            [tau_id],
            ["q", "v", "a"],
            ["tau"],
        )

        dq_dt = self._dq_dt(q, v)
        x = ca.vertcat(q, v)
        xdot = ca.vertcat(dq_dt, a_expr)

        self.f_xdot = ca.Function(
            "xdot", [x, tau], [xdot], ["x", "tau"], ["xdot"]
        )

    # --- Tangent-space dynamics for collocation ---

    def build_tangent_dynamics(
        self, q_ref_quat: np.ndarray
    ) -> tuple[ca.Function, ca.Function]:
        """Return ``(f_kin, f_inv_dyn)`` on the tangent state.

        State layout (2 * nv_reduced):
            xt[0:3]           base position (world)
            xt[3:6]           base rotation vector phi around q_ref
            xt[6 : 6+n_act]   actuated coordinates theta
            xt[6+n_act:]      [v_base (6); thetadot]

        Uses the small-angle approximation dphi/dt ≈ ω_body. In collocation:
            f_kin(x_j) · dt          == xp[:6+n_act]
            f_inv_dyn(x_j, v̇_poly_j) == τ_j_full
        """
        cmap = self.robot.coord_map
        n_act = cmap.n_theta
        nv_r = 6 + n_act

        xt = ca.SX.sym("xt", 2 * nv_r)
        a_r = ca.SX.sym("a", nv_r)
        pos, phi, theta = xt[0:3], xt[3:6], xt[6 : 6 + n_act]
        v_r = xt[6 + n_act :]
        vb, thd = v_r[0:6], v_r[6:]
        ab, thdd = a_r[0:6], a_r[6:]

        # q_base = q_ref ⊗ exp(φ). pin.neutral keeps continuous joints valid.
        q_ref_full = ca.SX(pin.neutral(self.robot.model))
        q_ref_full[3:7] = ca.SX(q_ref_quat)
        dv = ca.SX.zeros(self.nv)
        dv[3:6] = phi
        q_base = cpin.integrate(self.cmodel, q_ref_full, dv)[3:7]
        q_pin = ca.vertcat(pos, q_base, cmap.q_joints(theta))

        # Kinematics: ṗ = R·v_lin, φ̇ ≈ ω_body, θ̇ = thd
        v_tree = ca.vertcat(vb, cmap.v_joints(theta, thd))       # nv rows
        a_tree = ca.vertcat(ab, cmap.a_joints(theta, thd, thdd))  # nv rows

        dp = self._dq_dt(q_pin, v_tree)[0:3]
        xt_kin = ca.vertcat(dp, vb[3:6], thd)               # 6 + n_act rows

        tau_tree = self.f_inverse_dynamics(q_pin, v_tree, a_tree)   # nv rows
        # Project onto the reduced coordinates: tau_r = S^T tau
        tau_r = ca.vertcat(tau_tree[0:6], cmap.tau_joints(theta, tau_tree[6:]))

        f_kin = ca.Function(
            "kin_tangent", [xt], [xt_kin], ["xt"], ["xtkin"]
        )
        f_inv_dyn = ca.Function(
            "inv_dyn_tangent", [xt, a_r], [tau_r], ["xt", "a"], ["tau"]
        )
        return f_kin, f_inv_dyn

    # --- Numeric evaluation ---

    def eval_forward_dynamics(
        self,
        q: np.ndarray,
        v: np.ndarray,
        tau: np.ndarray,
    ) -> np.ndarray:
        """Evaluate forward dynamics numerically."""
        return np.array(self.f_forward_dynamics(q, v, tau)).flatten()

    def eval_inverse_dynamics(
        self,
        q: np.ndarray,
        v: np.ndarray,
        a: np.ndarray,
    ) -> np.ndarray:
        """Evaluate inverse dynamics numerically."""
        return np.array(self.f_inverse_dynamics(q, v, a)).flatten()

    # --- Reduced-coordinate dynamics (closed chains) ---

    def build_reduced_dynamics(self) -> ca.Function:
        """``(q_base, theta, v_r) -> (M_r, b_r)`` with ``M_r a_r + b_r = tau_r``.

        Needed to simulate closed-chain robots, whose URDF tree would come apart.
        Inverse dynamics is affine in the acceleration, so
        ``M_r = d resid / d a_r`` and ``b_r = resid(a_r = 0)``. Built lazily
        and cached.
        """
        if getattr(self, "_f_reduced_Mb", None) is not None:
            return self._f_reduced_Mb

        cmap = self.robot.coord_map
        n_theta = cmap.n_theta
        nv_r = 6 + n_theta

        q_base = ca.SX.sym("q_base", 7)
        theta = ca.SX.sym("theta", n_theta)
        v_r = ca.SX.sym("v_r", nv_r)
        a_r = ca.SX.sym("a_r", nv_r)

        q = ca.vertcat(q_base, cmap.q_joints(theta))
        v = ca.vertcat(v_r[:6], cmap.v_joints(theta, v_r[6:]))
        a = ca.vertcat(a_r[:6], cmap.a_joints(theta, v_r[6:], a_r[6:]))

        tau_tree = self.f_inverse_dynamics(q, v, a)
        resid = ca.vertcat(tau_tree[:6], cmap.tau_joints(theta, tau_tree[6:]))

        self._f_reduced_Mb = ca.Function(
            "reduced_Mb", [q_base, theta, v_r],
            [ca.jacobian(resid, a_r), ca.substitute(resid, a_r, ca.SX.zeros(nv_r))],
            ["q_base", "theta", "v_r"], ["M_r", "b_r"],
        )
        return self._f_reduced_Mb

    def eval_reduced_forward_dynamics(
        self,
        q_base: np.ndarray,
        theta: np.ndarray,
        v_r: np.ndarray,
        tau_r: np.ndarray,
    ) -> np.ndarray:
        """Reduced acceleration ``a_r`` for the reduced force ``tau_r``.

        Serial robots use the tree forward dynamics directly.
        """
        if isinstance(self.robot.coord_map, IdentityMap):
            q = np.concatenate([np.asarray(q_base), np.asarray(theta)])
            return self.eval_forward_dynamics(q, v_r, tau_r)

        M_r, b_r = self.build_reduced_dynamics()(q_base, theta, v_r)
        return np.linalg.solve(np.asarray(M_r), np.asarray(tau_r) - np.asarray(b_r).ravel())

    def eval_reduced_base_acceleration(
        self,
        q_base: np.ndarray,
        theta: np.ndarray,
        v_r: np.ndarray,
        a_joints: np.ndarray,
    ) -> np.ndarray:
        """Base acceleration with the joint motion prescribed.

        Solves the unactuated rows ``M_bb a_b + M_bj a_j + b_b = 0``, as for
        a position-controlled replay (Gazebo) and as imposed by the OCP.
        """
        nv_base = 6
        if isinstance(self.robot.coord_map, IdentityMap):
            q = np.concatenate([np.asarray(q_base), np.asarray(theta)])
            v = np.asarray(v_r)
            M = np.array(self.f_M_rb(q)) + np.array(self.f_M_added(q))
            b = (np.array(self.f_C_rb(q, v)) @ v
                 + np.array(self.f_C_A_v(q, v)).flatten()
                 + np.array(self.f_g_rb(q)).flatten()
                 - np.array(self.f_tau_buoyancy(q)).flatten()
                 - np.array(self.f_tau_drag(q, v)).flatten())
        else:
            M_r, b_r = self.build_reduced_dynamics()(q_base, theta, v_r)
            M = np.asarray(M_r)
            b = np.asarray(b_r).ravel()

        return np.linalg.solve(
            M[:nv_base, :nv_base],
            -b[:nv_base] - M[:nv_base, nv_base:] @ np.asarray(a_joints),
        )

    def find_trim_state(
        self,
        q_joints: np.ndarray | None = None,
        z_guess: float = 0.05,
    ) -> np.ndarray:
        """Hydrostatic equilibrium configuration at rest.

        Solves for base height and pitch (roll stays zero) such that heave
        force and pitch moment vanish at v = 0, tau = 0.

        Parameters
        ----------
        q_joints : (n_actuated,) array, optional
            Fixed actuated coordinates; defaults to the spec's home pose.
        z_guess : float
            Initial guess for base height [m].

        Returns
        -------
        q_trim : (model.nq,) ndarray
            Full trim configuration in *tree* coordinates.
        """
        from scipy.optimize import minimize

        if q_joints is None:
            home = self.robot.spec.theta_home
            q_joints = (np.zeros(self.robot.n_actuated) if home is None
                        else np.asarray(home, dtype=float))
        q_joints = self.robot.coord_map.expand_numeric(q_joints)

        n_base = self.robot.n_base_q

        def make_q(z: float, roll: float, pitch: float) -> np.ndarray:
            q = pin.neutral(self.robot.model)
            q[2] = z
            cr, sr = np.cos(roll / 2), np.sin(roll / 2)
            cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
            q[3] = sr * cp  # qx
            q[4] = cr * sp  # qy
            q[5] = -sr * sp  # qz
            q[6] = cr * cp  # qw
            q[n_base:] = q_joints
            return q

        def residual(params):
            z, pitch = params
            q = make_q(z, 0.0, pitch)
            tau_b = np.array(self.f_tau_buoyancy(q)).flatten()
            g_rb = np.array(self.f_g_rb(q)).flatten()
            rhs = tau_b - g_rb
            return rhs[2] ** 2 + rhs[4] ** 2

        res = minimize(
            residual,
            [z_guess, 0.0],
            method="Nelder-Mead",
            options={"xatol": 1e-8, "fatol": 1e-12, "maxiter": 5000},
        )

        z_eq, pitch_eq = res.x
        roll_eq = 0.0
        q_trim = make_q(z_eq, roll_eq, pitch_eq)
        tau_b_trim = np.array(self.f_tau_buoyancy(q_trim)).flatten()
        g_trim = np.array(self.f_g_rb(q_trim)).flatten()
        rhs_trim = tau_b_trim - g_trim

        print("Trim state (hydrostatic base equilibrium):")
        print(f"  Base z:          {z_eq:.4f} m")
        print(f"  Roll:            {np.degrees(roll_eq):.2f} deg")
        print(f"  Pitch:           {np.degrees(pitch_eq):.2f} deg")
        print(f"  Total buoyancy:  {tau_b_trim[2]:.4f} N  (weight = {g_trim[2]:.4f} N)")
        print(f"  rhs[2] (heave):  {rhs_trim[2]:.2e} N   (-> 0 = force balanced)")
        print(f"  rhs[4] (pitch):  {rhs_trim[4]:.2e} N*m (-> 0 = moment balanced)")
        print("Base force residuals at trim:")
        for i, name in enumerate(["surge", "sway", "heave", "roll", "pitch", "yaw"]):
            print(f"  rhs[{i}] ({name:5s}) = {rhs_trim[i]:.4e}")
        return q_trim

    def print_summary(self):
        """Print a summary of the symbolic dynamics."""
        print("=== Symbolic Dynamics Summary ===")
        print(
            f"  State dimension:   nq={self.nq} (base_q=7 + joints=12), "
            f"nv={self.nv} (base_v=6 + joints=12)"
        )
        print(
            f"  State vector x:    dim={self.nq + self.nv}  [q({self.nq}); v({self.nv})]"
        )
        print(
            f"  Links with hydro:  "
            f"{sum(1 for link in self.robot.links.values() if link.cylinder)}"
        )
        print(f"  Fluid density:     {self.rho} kg/m^3")
        print(f"  Drag coeffs:       Cd_t={self.Cd_t}, Cd_a={self.Cd_a}")
        print(f"  Added-mass coeffs: Ca_t={self.Ca_t}, Ca_a={self.Ca_a}")
        print(f"  Water surface:     z = {self.z_surface:.3f} m")
        print(f"  Drag threshold:    v = {self.v_linear_threshold:.2f} m/s")
        print()
        print("CasADi Functions:")
        for attr_name in sorted(dir(self)):
            if attr_name.startswith("f_") and isinstance(
                getattr(self, attr_name), ca.Function
            ):
                fn = getattr(self, attr_name)
                ins = " x ".join(str(fn.size_in(i)) for i in range(fn.n_in()))
                outs = " x ".join(str(fn.size_out(i)) for i in range(fn.n_out()))
                print(f"  {fn.name():25s}  ({ins}) -> ({outs})")
