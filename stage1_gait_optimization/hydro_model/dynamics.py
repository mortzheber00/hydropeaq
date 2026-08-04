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

from .coordinate_map import IdentityMap
from .hydrodynamics import RHO_WATER, SymbolicHydrodynamicModel
from .robot import QuadrupedRobot


def fn_name(*parts: str) -> str:
    """CasADi function names allow only letters, digits and single underscores.

    Link and leg names come from the URDF, where dots are legal (BODY2 has
    ``Link_BL1.1``), so they must be sanitised before use as a name.
    """
    raw = "_".join(parts)
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in raw)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")


class SymbolicDynamics:
    """CasADi-symbolic dynamics for the AMPH quadruped.

    Parameters
    ----------
    robot : QuadrupedRobot
        Pinocchio-backed robot (must have cylinders built already).
    rho : float
        Fluid density [kg/m^3].
    Cd_t, Cd_a : float
        Transverse / axial quadratic form-drag coefficients.
    Cd_lin_t, Cd_lin_a : float, optional
        Independent linear (skin-friction) damping coefficients — Fossen's D_S.
        Default None falls back to Cd_t / Cd_a.  Tune together with
        v_linear_threshold.
    Ca_t, Ca_a : float
        Transverse / axial added-mass coefficients.
    """

    def __init__(
        self,
        robot: QuadrupedRobot,
        rho: float = RHO_WATER,
        Cd_t: float = 0.7,
        Cd_a: float = 0.275,
        Cd_lin_t: float = 0.1,
        Cd_lin_a: float = 0.5,
        Ca_t: float = 1.3,
        Ca_a: float = 0.3,
        z_surface: float = 0.0,
        v_linear_threshold: float = 0.7,
        leg_thrust_scale: float = 1.0,
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
        self.Cd_lin_t = Cd_lin_t
        self.Cd_lin_a = Cd_lin_a
        self.Ca_t = Ca_t
        self.Ca_a = Ca_a
        self.z_surface = z_surface
        self.v_linear_threshold = v_linear_threshold
        self.leg_thrust_scale = leg_thrust_scale

        # Symbolic state variables
        self.q = ca.SX.sym("q", self.nq)
        self.v = ca.SX.sym("v", self.nv)  # qd
        self.a = ca.SX.sym("a", self.nv)  # qdd
        self.tau = ca.SX.sym("tau", self.nv)

        # Pre-compute all symbolic expressions and wrap as CasADi Functions
        self._f_reduced_Mb = None
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

        # Foot positions (convenience)
        # Must agree with QuadrupedRobot.foot_positions(): the foot is a fixed
        # offset in some frame's local coordinates.  Robots with a dedicated
        # *_Foot_link use a zero offset, and the term is skipped entirely so
        # their expression graph is unchanged.
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

        # Added-mass Coriolis  C_A·v  from the per-link Kirchhoff force at
        # zero acceleration: tau_added is linear in a
        # (tau_added = M_A(q)·a + C_A(q,v)·v), so a = 0 isolates C_A·v.
        # Replaces the Christoffel-symbol construction on the joint-space
        # M_A(q), whose symbolic derivatives were prohibitively large in the
        # NLP; the only neglected physics is the ∂α/∂q (submersion-ratio)
        # Coriolis contribution.
        C_A_v = hydro.f_tau_added(self.q, self.v, ca.SX.zeros(self.nv))
        self.f_C_A_v = ca.Function(
            "C_A_v", [self.q, self.v], [C_A_v], ["q", "v"], ["C_A_v"]
        )

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

        [M_rb(q) + M_A(q)] * vdot  +  [C_rb(q,v) + C_A(q,v)] * v  +  g_rb(q)
            = tau  +  tau_buoyancy(q)  +  tau_drag(q, v)

        The added-mass terms M_A·vdot + C_A·v come from the per-link
        Kirchhoff force tau_added(q, v, a) built in the hydro model.

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
        C_A_v = self.f_C_A_v(q, v)

        M_total = M_rb + M_A
        rhs = tau + tau_b + tau_d - C_rb @ v - C_A_v - g_rb
        #rhs = tau + tau_b + tau_d - C_rb @ v - g_rb

        # Forward dynamics: vdot = M_total \ rhs   (nv = 18)
        a_expr = ca.solve(M_total, rhs)

        self.f_forward_dynamics = ca.Function(
            "forward_dynamics",
            [q, v, tau],
            [a_expr],
            ["q", "v", "tau"],
            ["a"],
        )

        # Inverse dynamics: tau = M_total * vdot + (C_rb + C_A)·v + g − tau_hydro
        # Rigid-body part via RNEA — a single O(n) recursion for
        # M_rb·a + C_rb·v + g_rb with a much smaller symbolic graph than
        # assembling M_rb and C_rb explicitly.  Added-mass part via the
        # per-link Kirchhoff force tau_added = M_A·a + C_A·v.
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

        Reduced state layout (dim = 2*nv_reduced; 36 for amph):
            xt[0:3]                base position (world)
            xt[3:6]                base rotation tangent phi  (around q_ref)
            xt[6 : 6+n_act]        theta — the independent actuated coordinates
            xt[6+n_act:]           reduced velocity [v_base (6); thetadot]

        For a serial robot theta is the joint vector and this is the tree
        velocity.  For a closed-chain robot the robot's ``coord_map`` expands
        both onto the tree and projects the resulting forces back.

        Small-angle approx dphi/dt ≈ ω_body.

        Use in collocation with v̇_poly_j = (1/dt)·Σ_i C[i,j]·v_all[i]:
            f_kin(x_j) · dt          == xp[:6+n_act]      (kinematic rows)
            f_inv_dyn(x_j, v̇_poly_j) == τ_j_full           (dynamic rows)

        The dynamic constraint is the inverse-dynamics equality
        ``M(q)·a + C·v + g − τ_hydro = τ``
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

        # Recover quaternion from tangent vector:
        #   q_base = q_ref ⊗ exp_SO3(φ),  exp_SO3(φ) = [sin(‖φ‖/2)·φ/‖φ‖, cos(‖φ‖/2)]
        # cpin.integrate implements q_ref ⊞ dv on the Lie group; setting dv[3:6]=φ
        # selects only the rotational DOF so position and joints stay at zero.
        # pin.neutral rather than zeros: a tree with continuous joints stores
        # (cos, sin) pairs, whose neutral element is (1, 0), not (0, 0).  For a
        # purely revolute tree neutral *is* zeros, so this is a no-op there.
        q_ref_full = ca.SX(pin.neutral(self.robot.model))
        q_ref_full[3:7] = ca.SX(q_ref_quat)
        dv = ca.SX.zeros(self.nv)
        dv[3:6] = phi
        q_base = cpin.integrate(self.cmodel, q_ref_full, dv)[3:7]
        q_pin = ca.vertcat(pos, q_base, cmap.q_joints(theta))

        # Kinematic time derivatives of the position block:
        #   ṗ       = R(q_base) · v_lin          (body→world rotation of linear velocity)
        #   φ̇       ≈ ω_body = v[3:6]            (small-angle: tangent rate ≈ body angular vel.)
        #   q̇_joints = v_joints = v[6:]           (revolute joints: trivial)
        v_tree = ca.vertcat(vb, cmap.v_joints(theta, thd))       # nv rows
        a_tree = ca.vertcat(ab, cmap.a_joints(theta, thd, thdd))  # nv rows

        dp = self._dq_dt(q_pin, v_tree)[0:3]
        xt_kin = ca.vertcat(dp, vb[3:6], thd)               # 6 + n_act rows

        # Inverse dynamics:  τ = (M_rb + M_a)·a + C_rb·v + g − τ_buoy − τ_drag
        tau_tree = self.f_inverse_dynamics(q_pin, v_tree, a_tree)   # nv rows
        # Project onto the reduced coordinates (virtual work: tau_r = S^T tau).
        # With n_theta actuators this keeps the reduced system fully actuated,
        # which is what lets the collocation transcription stay unchanged.
        tau_r = ca.vertcat(tau_tree[0:6], cmap.tau_joints(theta, tau_tree[6:]))

        f_kin = ca.Function(
            "kin_tangent", [xt], [xt_kin], ["xt"], ["xtkin"]
        )
        f_inv_dyn = ca.Function(
            "inv_dyn_tangent", [xt, a_r], [tau_r], ["xt", "a"], ["tau"]
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

    # ==================================================================
    # Constrained (reduced-coordinate) dynamics
    # ==================================================================

    def build_reduced_dynamics(self) -> ca.Function:
        """``(q_base, theta, v_r) -> (M_r, b_r)`` with ``M_r a_r + b_r = tau_r``.

        A closed-chain robot cannot be simulated in tree coordinates: the URDF
        is a spanning tree with the loop-closure pins missing, so integrating it
        lets the linkage come apart.  Its OCP controls are no help either --
        ``tau_r = S^T tau_tree`` with ``S`` of shape (n_tree, n_theta), and the
        16-dimensional null space of ``S^T`` is exactly the space of pin
        reaction forces, which the reduced formulation eliminates by design.

        The fix is to project the equations of motion onto the constraint
        manifold, which ``CoordinateMap`` already spans.  Since inverse dynamics
        is *affine* in acceleration, both blocks come straight out of it:

            resid(a_r) = M_r a_r + b_r      (verified to ~1e-16)
            M_r = d resid / d a_r           b_r = resid at a_r = 0

        Built lazily and cached; the Jacobian makes it a few times more
        expensive to construct than the tree functions.
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
        """Reduced acceleration ``a_r`` given the reduced generalised force.

        A robot whose actuated coordinates *are* its tree joints is dispatched
        to the ordinary tree forward dynamics, so its results are unchanged.
        """
        if isinstance(self.robot.coord_map, IdentityMap):
            q = np.concatenate([np.asarray(q_base), np.asarray(theta)])
            return self.eval_forward_dynamics(q, v_r, tau_r)

        M_r, b_r = self.build_reduced_dynamics()(q_base, theta, v_r)
        return np.linalg.solve(np.asarray(M_r), np.asarray(tau_r) - np.asarray(b_r).ravel())

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
            Reduced actuated coordinates theta, held fixed during the solve.
            Defaults to the robot's home pose (zeros unless the spec says
            otherwise — a closed-chain robot may not be assemblable at zero).
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
