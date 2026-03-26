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
        z_surface: float = 0.0,
        v_linear_threshold: float = 0.2,
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
        self.z_surface = z_surface
        self.v_linear_threshold = v_linear_threshold

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
          - Buoyancy: upward force scaled by symbolic submersion ratio α(q)
          - Drag: hybrid linear+quadratic drag scaled by α(q)
          - Added mass: configuration-dependent added inertia scaled by α(q)

        The cylinder geometry (radius, length) is frozen at the values
        computed by ``robot.build_cylinders()``.  The cylinder *axis* and
        *center z-position* are recomputed symbolically from FK so that they
        track the current q.

        Submersion ratio α ∈ [0,1] is computed per link from the symbolic
        FK z-position and the constant water surface height z_surface.

        Hybrid drag: F = -(D₁·v + ½ρCdA|v|v), D₁ = ½ρCdA·v_threshold.
        This is smooth and differentiable everywhere, unlike a hard switch.
        """
        q, v = self.q, self.v
        rho, g = self.rho, GRAVITY
        z_surf = float(self.z_surface)
        v_thresh = float(self.v_linear_threshold)

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

            # Jacobian at the frame origin, world-aligned.
            # LOCAL_WORLD_ALIGNED gives the spatial velocity at the *frame
            # origin* in world-frame coordinates.  This matches the wrench
            # convention [F; r_offset × F] used below (wrench at frame origin).
            # Using WORLD would give velocity/wrench at the *world* origin,
            # making the base columns identical for all frames on the same
            # body and breaking left/right symmetry of torque projections.
            J_full = cpin.computeFrameJacobian(
                self.cmodel, self.cdata, q, fid,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )
            Jv = J_full[:3, :]  # (3, nv)

            # Link velocity at frame origin in world frame
            v_link = Jv @ v     # (3, 1)

            # ── Cylinder axis in world frame (symbolic) ──
            axis_sym = R_sym @ ca.SX(cyl.axis_local)
            axis_sym = axis_sym / (ca.norm_2(axis_sym) + 1e-15)

            r = cyl.radius
            L = cyl.length
            V = cyl.volume
            A_t = cyl.cross_section_transverse
            A_a = cyl.cross_section_axial

            # ── Symbolic submersion ratio α(q) ──
            # Reconstruct the cylinder midpoint in world frame from FK:
            #   p_center = R_link @ center_local + t_link
            # center_local is the fixed offset (in link LOCAL frame) from the
            # link body-frame origin to the cylinder midpoint, stored at
            # build time by QuadrupedRobot.build_cylinders().
            p_center = oMf.translation + R_sym @ ca.SX(cyl.center_local)
            z_center = p_center[2]
            # Vertical half-span: axial projection + radial projection
            axis_z_abs = ca.fabs(axis_sym[2])
            dz_axial = 0.5 * L * axis_z_abs
            dz_radial = r * ca.sqrt(ca.fmax(1.0 - axis_sym[2] ** 2, 0.0))
            dz_half = dz_axial + dz_radial
            z_top = z_center + dz_half
            z_bottom = z_center - dz_half
            # α = clamp((z_surf - z_bottom) / (z_top - z_bottom), 0, 1)
            alpha = ca.fmin(1.0, ca.fmax(0.0,
                (z_surf - z_bottom) / (z_top - z_bottom + 1e-6)
            ))

            # ── Buoyancy (scaled by α) ──
            # The buoyancy force acts at the cylinder center, offset from
            # the frame origin by r_offset = R @ center_local.  A force F
            # at offset r from the frame origin produces the spatial wrench
            # [F; r × F] at the frame origin, matching the
            # LOCAL_WORLD_ALIGNED Jacobian convention.
            F_buoy = ca.vertcat(0.0, 0.0, alpha * rho * g * V)
            r_offset = R_sym @ ca.SX(cyl.center_local)   # frame origin → CoB
            wrench_buoy = ca.vertcat(F_buoy, ca.cross(r_offset, F_buoy))
            tau_buoyancy += J_full.T @ wrench_buoy

            # ── Hybrid drag (scaled by α) ──
            # F = -(D₁·v + ½ρCdA|v|v), D₁ = ½ρCdA·v_threshold
            v_ax_mag = ca.dot(v_link, axis_sym)
            v_ax = v_ax_mag * axis_sym
            v_tr = v_link - v_ax
            # Smooth norm/abs to avoid NaN gradients at v=0
            _eps2 = 1e-8
            v_tr_mag = ca.sqrt(ca.dot(v_tr, v_tr) + _eps2)

            D1_a = 0.5 * rho * self.Cd_a * A_a * v_thresh
            v_ax_abs = ca.sqrt(v_ax_mag ** 2 + _eps2)
            F_drag_ax = -(D1_a * v_ax + 0.5 * rho * self.Cd_a * A_a * v_ax_abs * v_ax)

            D1_t = 0.5 * rho * self.Cd_t * A_t * v_thresh
            F_drag_tr = -(D1_t * v_tr + 0.5 * rho * self.Cd_t * A_t * v_tr_mag * v_tr)

            # Drag acts at the cylinder center — use full wrench like buoyancy
            F_drag = alpha * (F_drag_ax + F_drag_tr)
            wrench_drag = ca.vertcat(F_drag, ca.cross(r_offset, F_drag))
            tau_drag += J_full.T @ wrench_drag

            # ── Added mass (joint-space, scaled by α) ──
            # M_A = J^T * (α · M_A_cartesian) * J
            # M_A_cartesian = ma_t * I + (ma_a - ma_t) * (a ⊗ a)
            ma_t = self.Ca_t * rho * V
            ma_a = self.Ca_a * rho * V
            a_col = axis_sym  # already (3,1)
            M_A_cart = alpha * (ma_t * ca.SX.eye(3) + (ma_a - ma_t) * (a_col @ a_col.T))
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
    # SE(3) configuration time derivative
    # ==================================================================

    def _dq_dt(self, q: ca.SX, v: ca.SX) -> ca.SX:
        """Configuration time derivative for a free-flyer + revolute joints.

        For the free-floating base the velocity ``v[0:6]`` is expressed in the
        LOCAL (body) frame, so the configuration derivative is NOT simply ``v``:

          dp/dt   = R(q_base) · v_lin        (rotate body-frame velocity to world)
          dquat/dt = ½ · q_base ⊗ [ω_body; 0]  (quaternion kinematics)

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
        qx = q[3];  qy = q[4];  qz = q[5];  qw = q[6]

        # Linear and angular velocity in body frame
        vx = v[0];  vy = v[1];  vz = v[2]
        wx = v[3];  wy = v[4];  wz = v[5]

        # World-frame position derivative:  dp/dt = R_world_body · v_body
        # R constructed from quaternion (scalar last convention)
        dp = ca.vertcat(
            (1 - 2*(qy**2 + qz**2))*vx + 2*(qx*qy - qw*qz)*vy + 2*(qx*qz + qw*qy)*vz,
            2*(qx*qy + qw*qz)*vx + (1 - 2*(qx**2 + qz**2))*vy + 2*(qy*qz - qw*qx)*vz,
            2*(qx*qz - qw*qy)*vx + 2*(qy*qz + qw*qx)*vy + (1 - 2*(qx**2 + qy**2))*vz,
        )

        # Quaternion kinematics:  dq/dt = ½ · q ⊗ [ω; 0]
        dqx = 0.5 * ( qw*wx + qy*wz - qz*wy)
        dqy = 0.5 * ( qw*wy + qz*wx - qx*wz)
        dqz = 0.5 * ( qw*wz + qx*wy - qy*wx)
        dqw = 0.5 * (-qx*wx - qy*wy - qz*wz)

        # Joint angle derivatives (trivial)
        return ca.vertcat(dp, dqx, dqy, dqz, dqw, v[6:])   # (19, 1)

    # ==================================================================
    # Full equations of motion
    # ==================================================================

    def _build_eom(self):
        """Assemble the complete EoM as CasADi Functions.

        [M_rb(q) + M_A(q)] · v̇  +  C_rb(q,v) · v  +  g_rb(q)
            = τ  +  τ_buoyancy(q)  +  τ_drag(q, v)

        The added-mass Coriolis term C_A is neglected (small for slow motion).

        For the free-floating base the first 6 rows of τ carry the external
        wrench on the base (zero for a purely swimming robot with no
        direct base actuation).

        State-space ODE:
          x  = [q (19); v (18)]           dim = 37
          ẋ  = [dq/dt (19); v̇ (18)]       dim = 37

        Note: dq/dt ≠ v for the free-flyer because the quaternion
        derivative and position derivative involve the current orientation.
        Use ``f_xdot`` for ODE integration and ``f_integrate`` for
        single-step configuration updates on the SE(3) manifold.
        """
        q, v, tau = self.q, self.v, self.tau

        M_rb  = self.f_M_rb(q)
        C_rb  = self.f_C_rb(q, v)
        g_rb  = self.f_g_rb(q)
        M_A   = self.f_M_added(q)
        tau_b = self.f_tau_buoyancy(q)
        tau_d = self.f_tau_drag(q, v)

        M_total = M_rb + M_A
        rhs     = tau + tau_b + tau_d - C_rb @ v - g_rb

        # Forward dynamics: v̇ = M_total \ rhs   (nv = 18)
        a_expr = ca.solve(M_total, rhs)

        self.f_forward_dynamics = ca.Function(
            "forward_dynamics",
            [q, v, tau], [a_expr],
            ["q", "v", "tau"], ["a"],
        )

        # Inverse dynamics: τ = M_total · v̇ + C·v + g − τ_hydro
        a     = self.a
        tau_id = (self.f_M_rb(q) + self.f_M_added(q)) @ a \
                 + self.f_C_rb(q, v) @ v \
                 + self.f_g_rb(q) \
                 - self.f_tau_buoyancy(q) \
                 - self.f_tau_drag(q, v)

        self.f_inverse_dynamics = ca.Function(
            "inverse_dynamics",
            [q, v, a], [tau_id],
            ["q", "v", "a"], ["tau"],
        )

        # Continuous-time state-space ODE  ẋ = f(x, u)
        # x = [q (19); v (18)],  u = tau (18)
        # ẋ = [dq/dt (19); v̇ (18)]
        #
        # dq/dt is computed via the SE(3) tangent map (see _dq_dt), which
        # accounts for the quaternion kinematics of the floating base.
        dq_dt = self._dq_dt(q, v)   # (19, 1)
        x     = ca.vertcat(q, v)    # (37, 1)
        xdot  = ca.vertcat(dq_dt, a_expr)  # (37, 1)

        self.f_xdot = ca.Function(
            "xdot",
            [x, tau], [xdot],
            ["x", "tau"], ["xdot"],
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

    def find_trim_state(
        self,
        q_joints: np.ndarray | None = None,
        z_guess: float = 0.05,
    ) -> np.ndarray:
        """Find the full floating-equilibrium configuration at rest.

        Solves for base z, pitch, and roll such that all base
        accelerations are zero at v=0, τ=0.  Uses scipy.optimize.minimize
        to drive the base acceleration residual to zero.

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

        n_base = self.robot.n_base_q  # 7
        v_zero  = np.zeros(self.nv)
        tau_zero = np.zeros(self.nv)

        def make_q(z: float, roll: float, pitch: float) -> np.ndarray:
            """Build q from base z, roll, pitch (yaw=0)."""
            q = np.zeros(self.nq)
            q[2] = z
            # Quaternion from roll-pitch-yaw (scalar last: qx, qy, qz, qw)
            cr, sr = np.cos(roll / 2), np.sin(roll / 2)
            cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
            # yaw = 0
            q[3] = sr * cp           # qx
            q[4] = cr * sp           # qy
            q[5] = -sr * sp          # qz
            q[6] = cr * cp           # qw
            q[n_base:] = q_joints
            return q

        def residual(params):
            """Sum of squared heave + pitch generalized force residuals.

            Uses direct force balance (tau_buoyancy - g_rb) rather than
            forward dynamics.  The forward-dynamics formulation (a[2]**2 +
            a[4]**2) is wrong here because the joint–base coupling in
            M^{-1} can zero those two acceleration components while leaving
            a large net force unbalanced (e.g. 14 N net upward when only
            the two base DOFs happen to cancel through off-diagonal terms).
            """
            z, pitch = params
            q = make_q(z, 0.0, pitch)  # roll = 0 (left-right symmetric)
            tau_b = np.array(self.f_tau_buoyancy(q)).flatten()
            g_rb  = np.array(self.f_g_rb(q)).flatten()
            rhs   = tau_b - g_rb
            return rhs[2]**2 + rhs[4]**2

        res = minimize(residual, [z_guess, 0.0], method="Nelder-Mead",
                       options={"xatol": 1e-8, "fatol": 1e-12, "maxiter": 5000})

        z_eq, pitch_eq = res.x
        roll_eq = 0.0
        q_trim = make_q(z_eq, roll_eq, pitch_eq)
        tau_b_trim = np.array(self.f_tau_buoyancy(q_trim)).flatten()
        g_trim     = np.array(self.f_g_rb(q_trim)).flatten()
        rhs_trim   = tau_b_trim - g_trim

        print("Trim state (hydrostatic base equilibrium):")
        print(f"  Base z:          {z_eq:.4f} m")
        print(f"  Roll:            {np.degrees(roll_eq):.2f}°")
        print(f"  Pitch:           {np.degrees(pitch_eq):.2f}°")
        print(f"  Total buoyancy:  {tau_b_trim[2]:.4f} N  (weight = {g_trim[2]:.4f} N)")
        print(f"  rhs[2] (heave):  {rhs_trim[2]:.2e} N   (→ 0 = force balanced)")
        print(f"  rhs[4] (pitch):  {rhs_trim[4]:.2e} N·m (→ 0 = moment balanced)")
        print('Base force residuals at trim:')
        for i, name in enumerate(['surge','sway','heave','roll','pitch','yaw']):
            print(f'  rhs[{i}] ({name:5s}) = {rhs_trim[i]:.4e}')
        return q_trim

    def print_summary(self):
        """Print a summary of the symbolic dynamics."""
        print("=== Symbolic Dynamics Summary ===")
        print(f"  State dimension:   nq={self.nq} (base_q=7 + joints=12), "
              f"nv={self.nv} (base_v=6 + joints=12)")
        print(f"  State vector x:    dim={self.nq + self.nv}  [q({self.nq}); v({self.nv})]")
        print(f"  Links with hydro:  {sum(1 for l in self.robot.links.values() if l.cylinder)}")
        print(f"  Fluid density:     {self.rho} kg/m^3")
        print(f"  Drag coeffs:       Cd_t={self.Cd_t}, Cd_a={self.Cd_a}")
        print(f"  Added-mass coeffs: Ca_t={self.Ca_t}, Ca_a={self.Ca_a}")
        print(f"  Water surface:     z = {self.z_surface:.3f} m")
        print(f"  Drag threshold:    v = {self.v_linear_threshold:.2f} m/s")
        print()
        print("CasADi Functions:")
        for attr_name in sorted(dir(self)):
            if attr_name.startswith("f_") and isinstance(getattr(self, attr_name), ca.Function):
                fn = getattr(self, attr_name)
                ins  = " × ".join(str(fn.size_in(i))  for i in range(fn.n_in()))
                outs = " × ".join(str(fn.size_out(i)) for i in range(fn.n_out()))
                print(f"  {fn.name():25s}  ({ins}) → ({outs})")
