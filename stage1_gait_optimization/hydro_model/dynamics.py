"""
CasADi-symbolic rigid-body + hydrodynamic dynamics for the AMPH quadruped.

Uses ``pinocchio.casadi`` to build symbolic expressions for the standard
rigid-body terms (mass matrix, Coriolis, gravity, Jacobians, FK) and adds
the hydrodynamic forces via ``SymbolicHydrodynamicModel``.

The final equations of motion are:

    [M_rb(q) + M_A(q)] * qdd + [C_rb(q, qd) + C_A(q, qd)] * qd
        + g_rb(q) = tau + tau_hydro(q, qd)

where tau_hydro collects buoyancy, drag, and pressure-gradient forces
projected into joint space via link Jacobians.

All public functions return CasADi ``ca.Function`` objects that can be
evaluated numerically or embedded in an NLP.
"""

from __future__ import annotations

import casadi as ca
import numpy as np
import pinocchio as pin
import pinocchio.casadi as cpin

from .hydrodynamics import RHO_WATER, SymbolicHydrodynamicModel
from .robot import QuadrupedRobot


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
        z_surface: float = 0.0,
        v_linear_threshold: float = 0.005,
    ):
        self.robot = robot
        self.nq = robot.nq
        self.nv = robot.nv

        # Cast the Pinocchio model to CasADi
        self.cmodel = cpin.Model(robot.model)
        self.cdata = self.cmodel.createData()

        # Hydro parameters (stored for print_summary)
        self.rho = rho
        self.Cd_t = Cd_t
        self.Cd_a = Cd_a
        self.Ca_t = Ca_t
        self.Ca_a = Ca_a
        self.z_surface = z_surface
        self.v_linear_threshold = v_linear_threshold

        # Symbolic state variables
        self.q = ca.SX.sym("q", self.nq)
        self.v = ca.SX.sym("v", self.nv)  # qd
        self.a = ca.SX.sym("a", self.nv)  # qdd
        self.tau = ca.SX.sym("tau", self.nv)

        # Pre-compute all symbolic expressions and wrap as CasADi Functions
        self._build_rigid_body_functions()
        self._build_fk_functions()
        self._build_hydro_functions()
        self._build_eom()

    # ==================================================================
    # Rigid-body dynamics
    # ==================================================================

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

    # ==================================================================
    # Forward kinematics & Jacobians
    # ==================================================================

    def _build_fk_functions(self):
        """Build FK positions, velocities, and Jacobians for every link.

        Creates:
          - f_fk[link_name] : (q) -> (pos_3x1, R_3x3)
          - f_Jv[link_name] : (q) -> J_3xnv  (translational Jacobian)
          - f_J[link_name]  : (q) -> J_6xnv  (full spatial Jacobian)
          - f_foot_pos[leg] : (q) -> pos_3x1  (foot position)
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

            pos = oMf.translation
            R = oMf.rotation

            self.f_fk[name] = ca.Function(
                f"fk_{name}", [q], [pos, R], ["q"], ["pos", "R"]
            )

            J_full = cpin.computeFrameJacobian(
                self.cmodel, self.cdata, q, fid, pin.ReferenceFrame.WORLD
            )
            J_trans = J_full[:3, :]

            self.f_J[name] = ca.Function(
                f"J_{name}", [q], [J_full], ["q"], ["J"]
            )
            self.f_Jv[name] = ca.Function(
                f"Jv_{name}", [q], [J_trans], ["q"], ["Jv"]
            )

        # Foot positions (convenience)
        self.f_foot_pos: dict[str, ca.Function] = {}
        for leg, fid in self.robot.foot_frame_ids.items():
            pos = self.cdata.oMf[fid].translation
            self.f_foot_pos[leg] = ca.Function(
                f"foot_{leg}", [q], [pos], ["q"], ["pos"]
            )

    # ==================================================================
    # Hydrodynamic forces (delegated to SymbolicHydrodynamicModel)
    # ==================================================================

    def _build_hydro_functions(self):
        """Build hydrodynamic CasADi Functions via SymbolicHydrodynamicModel."""
        hydro = SymbolicHydrodynamicModel(
            robot=self.robot,
            cmodel=self.cmodel,
            cdata=self.cdata,
            q_sym=self.q,
            v_sym=self.v,
            nv=self.nv,
            rho=self.rho,
            Cd_transverse=self.Cd_t,
            Cd_axial=self.Cd_a,
            Ca_transverse=self.Ca_t,
            Ca_axial=self.Ca_a,
            z_surface=self.z_surface,
            v_linear_threshold=self.v_linear_threshold,
        )
        self.f_tau_buoyancy = hydro.f_tau_buoyancy
        self.f_tau_drag = hydro.f_tau_drag
        self.f_M_added = hydro.f_M_added

    # ==================================================================
    # SE(3) configuration time derivative
    # ==================================================================

    def _dq_dt(self, q: ca.SX, v: ca.SX) -> ca.SX:
        """Configuration time derivative for a free-flyer + revolute joints.

        For the free-floating base the velocity ``v[0:6]`` is expressed in the
        LOCAL (body) frame, so the configuration derivative is NOT simply ``v``:

          dp/dt   = R(q_base) * v_lin        (rotate body-frame velocity to world)
          dquat/dt = 0.5 * q_base x [w_body; 0]  (quaternion kinematics)

        For the revolute joints:
          dq_j/dt = v_j   (trivial, Euclidean)

        Returns ``dq_dt`` of shape (nq=19, 1).

        Configuration layout (Pinocchio free-flyer convention):
          q[0:3]  = world position  [x, y, z]
          q[3:7]  = unit quaternion [qx, qy, qz, qw]  (scalar last)
          q[7:19] = joint angles
        Velocity layout:
          v[0:3]  = linear velocity  in body frame
          v[3:6]  = angular velocity in body frame
          v[6:18] = joint velocities
        """
        # Quaternion components (scalar last: [qx, qy, qz, qw])
        qx, qy, qz, qw = q[3], q[4], q[5], q[6]
        vx, vy, vz = v[0], v[1], v[2]
        wx, wy, wz = v[3], v[4], v[5]

        # World-frame position derivative:  dp/dt = R_world_body * v_body
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

    # ==================================================================
    # Full equations of motion
    # ==================================================================

    def _build_eom(self):
        """Assemble the complete EoM as CasADi Functions.

        [M_rb(q) + M_A(q)] * vdot  +  C_rb(q,v) * v  +  g_rb(q)
            = tau  +  tau_buoyancy(q)  +  tau_drag(q, v)

        The added-mass Coriolis term C_A is neglected (small for slow motion).

        State-space ODE:
          x  = [q (19); v (18)]           dim = 37
          xdot  = [dq/dt (19); vdot (18)]       dim = 37

        Note: dq/dt != v for the free-flyer because the quaternion
        derivative and position derivative involve the current orientation.
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

        # Forward dynamics: vdot = M_total \ rhs   (nv = 18)
        a_expr = ca.solve(M_total, rhs)

        self.f_forward_dynamics = ca.Function(
            "forward_dynamics",
            [q, v, tau],
            [a_expr],
            ["q", "v", "tau"],
            ["a"],
        )

        # Inverse dynamics: tau = M_total * vdot + C*v + g - tau_hydro
        a = self.a
        tau_id = (
            (self.f_M_rb(q) + self.f_M_added(q)) @ a
            + self.f_C_rb(q, v) @ v
            + self.f_g_rb(q)
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

        # Continuous-time state-space ODE  xdot = f(x, u)
        dq_dt = self._dq_dt(q, v)
        x = ca.vertcat(q, v)
        xdot = ca.vertcat(dq_dt, a_expr)

        self.f_xdot = ca.Function(
            "xdot", [x, tau], [xdot], ["x", "tau"], ["xdot"]
        )

    # ==================================================================
    # Tangent-space (reduced) state-space ODE
    # ==================================================================

    def build_tangent_dynamics(
        self, q_ref_quat: np.ndarray
    ) -> tuple[ca.Function, ca.Function]:
        """Return ``(f_kin, f_inv_dyn)`` for the tangent state representation.

        Reduced state layout (dim = 2*nv = 36 here):
            xt[0:3]                base position (world)
            xt[3:6]                base rotation tangent phi  (around q_ref)
            xt[6 : 6+n_act]        joint positions
            xt[6+n_act:]           Pinocchio velocity v (18-dim, body frame)

        Small-angle approx dphi/dt ≈ ω_body.

        Use in collocation with v̇_poly_j = (1/dt)·Σ_i C[i,j]·v_all[i]:
            f_kin(x_j) · dt          == xp[:6+n_act]      (kinematic rows)
            f_inv_dyn(x_j, v̇_poly_j) == τ_j_full           (dynamic rows)

        The dynamic constraint is the inverse-dynamics equality
        ``M(q)·a + C·v + g − τ_hydro = τ``
        """
        n_act = self.nv - 6
        xt = ca.SX.sym("xt", 2 * self.nv)
        a = ca.SX.sym("a", self.nv)
        pos, phi, joints = xt[0:3], xt[3:6], xt[6 : 6 + n_act]
        v = xt[6 + n_act :]

        # Recover quaternion from tangent vector:
        #   q_base = q_ref ⊗ exp_SO3(φ),  exp_SO3(φ) = [sin(‖φ‖/2)·φ/‖φ‖, cos(‖φ‖/2)]
        # cpin.integrate implements q_ref ⊞ dv on the Lie group; setting dv[3:6]=φ
        # selects only the rotational DOF so position and joints stay at zero.
        q_ref_full = ca.SX.zeros(self.nq)
        q_ref_full[3:7] = ca.SX(q_ref_quat)
        dv = ca.SX.zeros(self.nv)
        dv[3:6] = phi
        q_base = cpin.integrate(self.cmodel, q_ref_full, dv)[3:7]
        q_pin = ca.vertcat(pos, q_base, joints)

        # Kinematic time derivatives of the position block:
        #   ṗ       = R(q_base) · v_lin          (body→world rotation of linear velocity)
        #   φ̇       ≈ ω_body = v[3:6]            (small-angle: tangent rate ≈ body angular vel.)
        #   q̇_joints = v_joints = v[6:]           (revolute joints: trivial)
        dp = self._dq_dt(q_pin, v)[0:3]
        xt_kin = ca.vertcat(dp, v[3:6], v[6:])              # 6 + n_act rows

        # Inverse dynamics:  τ = (M_rb + M_a)·a + C_rb·v + g − τ_buoy − τ_drag
        tau = self.f_inverse_dynamics(q_pin, v, a)          # nv rows

        f_kin = ca.Function(
            "kin_tangent", [xt], [xt_kin], ["xt"], ["xtkin"]
        )
        f_inv_dyn = ca.Function(
            "inv_dyn_tangent", [xt, a], [tau], ["xt", "a"], ["tau"]
        )
        return f_kin, f_inv_dyn

    # ==================================================================
    # Public convenience methods
    # ==================================================================

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

    def find_trim_state(
        self,
        q_joints: np.ndarray | None = None,
        z_guess: float = 0.05,
    ) -> np.ndarray:
        """Find the full floating-equilibrium configuration at rest.

        Solves for base z, pitch, and roll such that all base
        accelerations are zero at v=0, tau=0.

        Parameters
        ----------
        q_joints : (n_actuated,) array, optional
            Joint angles held fixed during the solve.  Defaults to
            neutral (all zeros).
        z_guess : float
            Initial guess for base height [m].

        Returns
        -------
        q_trim : (nq=19,) ndarray
            Full trim configuration.
        """
        from scipy.optimize import minimize

        if q_joints is None:
            q_joints = np.zeros(self.robot.n_actuated)

        n_base = self.robot.n_base_q

        def make_q(z: float, roll: float, pitch: float) -> np.ndarray:
            q = np.zeros(self.nq)
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
