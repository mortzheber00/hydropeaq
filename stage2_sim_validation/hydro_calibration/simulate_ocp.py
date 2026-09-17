#!/usr/bin/env python3
"""Roll out an OCP solution through the hydrodynamic model with prescribed joints.

The joints follow the solution, as a position controller in Gazebo would; only
the base is integrated (RK4). Coefficients default to hydro_params; the flags
override them for this run only. Writes ``<stem>_rollout.npz``.

Usage:
  python stage2_sim_validation/hydro_calibration/simulate_ocp.py --ocp task3_solution.npz
  python stage2_sim_validation/hydro_calibration/simulate_ocp.py --Cd_t 2.5 --Ca_t 1.0
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2]))
from stage1_gait_optimization.hydro_model import (
    SymbolicDynamics,
    hydro_params,
    load_robot,
)
from stage1_gait_optimization.hydro_model.trajectory import (
    coords_of,
    load_solution,
    save_solution,
)

# --- Helpers ---

def load_ocp(path: str, robot=None):
    """``(X, U, T, N, nq, robot)`` in the robot's reduced coordinates."""
    d = load_solution(path)
    if robot is None:
        robot = load_robot(d["robot"])
    return d["X"], d["U"], d["T"], d["N"], d["nq"], robot


def build_dynamics(args, robot) -> SymbolicDynamics:

    print("Building symbolic dynamics (CasADi compilation) …")
    t0 = time.time()
    dyn = SymbolicDynamics(
        robot,
        Cd_t=args.Cd_t,
        Cd_a=args.Cd_a,
        Cd_lin_t=args.Cd_lin_t,
        Cd_lin_a=args.Cd_lin_a,
        Ca_t=args.Ca_t,
        Ca_a=args.Ca_a,
        v_linear_threshold=args.v_lin,
        leg_thrust_scale=args.leg_thrust_scale,
    )
    print(f"  Done in {time.time() - t0:.1f}s")
    return dyn


def dq_base_dt(q_base: np.ndarray, v_base: np.ndarray) -> np.ndarray:
    """Numeric version of ``SymbolicDynamics._dq_dt`` for the base (body-frame twist)."""
    qx, qy, qz, qw = q_base[3], q_base[4], q_base[5], q_base[6]
    vx, vy, vz = v_base[0], v_base[1], v_base[2]
    wx, wy, wz = v_base[3], v_base[4], v_base[5]

    dp = np.array([
        (1 - 2*(qy**2 + qz**2))*vx + 2*(qx*qy - qw*qz)*vy + 2*(qx*qz + qw*qy)*vz,
        2*(qx*qy + qw*qz)*vx + (1 - 2*(qx**2 + qz**2))*vy + 2*(qy*qz - qw*qx)*vz,
        2*(qx*qz - qw*qy)*vx + 2*(qy*qz + qw*qx)*vy + (1 - 2*(qx**2 + qy**2))*vz,
    ])
    dquat = np.array([
        0.5 * ( qw*wx + qy*wz - qz*wy),
        0.5 * ( qw*wy + qz*wx - qx*wz),
        0.5 * ( qw*wz + qx*wy - qy*wx),
        0.5 * (-qx*wx - qy*wy - qz*wz),
    ])
    return np.concatenate([dp, dquat])



def rk4_base_step(
    q_base: np.ndarray,
    v_base: np.ndarray,
    theta_k: np.ndarray,
    theta_k1: np.ndarray,
    theta_dot_k: np.ndarray,
    theta_dot_k1: np.ndarray,
    a_joints: np.ndarray,
    dyn: SymbolicDynamics,
    dt: float,
):
    """One RK4 step of the base with prescribed joints (reduced coordinates).

    Joint position and velocity are interpolated at each stage's time; holding
    them at the left node lags the motion and causes spurious roll. The joint
    acceleration is constant per interval. The OCP torques are not used.
    """
    def f(qb, vb, s):
        theta = (1 - s) * theta_k + s * theta_k1
        theta_dot = (1 - s) * theta_dot_k + s * theta_dot_k1
        v_r = np.concatenate([vb, theta_dot])
        a_base = dyn.eval_reduced_base_acceleration(qb, theta, v_r, a_joints)
        return dq_base_dt(qb, vb), a_base

    dq1, dv1 = f(q_base, v_base, 0.0)
    dq2, dv2 = f(q_base + dt/2*dq1, v_base + dt/2*dv1, 0.5)
    dq3, dv3 = f(q_base + dt/2*dq2, v_base + dt/2*dv2, 0.5)
    dq4, dv4 = f(q_base + dt*dq3,   v_base + dt*dv3,   1.0)

    q_next = q_base + (dt/6) * (dq1 + 2*dq2 + 2*dq3 + dq4)
    v_next = v_base + (dt/6) * (dv1 + 2*dv2 + 2*dv3 + dv4)

    q_next[3:7] /= np.linalg.norm(q_next[3:7])
    return q_next, v_next


# --- Rollout ---

def rollout(
    X_ocp: np.ndarray,
    dyn: SymbolicDynamics,
    T: float,
    N: int,
):
    """Integrate the base with the joints following ``X_ocp``; returns X_sim (nq+nv, N+1)."""
    nq, nv = dyn.robot.nq_reduced, dyn.robot.nv_reduced
    dt = T / N

    # q = [pos (3), quat (4), theta], v = [base twist (6), thetadot]
    i_vbase_end = nq + 6

    X_sim = np.zeros((nq + nv, N + 1))
    X_sim[:, 0] = X_ocp[:, 0]

    print(f"Rolling out {N} steps (dt={dt:.4f}s, T={T:.3f}s, mode=prescribed-joint) …")
    t0 = time.time()

    q_base = X_ocp[:7, 0].copy()
    v_base = X_ocp[nq:i_vbase_end, 0].copy()
    for k in range(N):
        a_joints = (X_ocp[i_vbase_end:, k+1] - X_ocp[i_vbase_end:, k]) / dt

        q_base, v_base = rk4_base_step(
            q_base, v_base,
            X_ocp[7:nq, k], X_ocp[7:nq, k+1],
            X_ocp[i_vbase_end:, k], X_ocp[i_vbase_end:, k+1],
            a_joints, dyn, dt,
        )

        X_sim[:7, k+1]             = q_base
        X_sim[7:nq, k+1]           = X_ocp[7:nq, k+1]
        X_sim[nq:i_vbase_end, k+1] = v_base
        X_sim[i_vbase_end:, k+1]   = X_ocp[i_vbase_end:, k+1]

        if (k + 1) % 10 == 0 or k == N - 1:
            print(f"  step {k+1:3d}/{N}  x={q_base[0]:.4f}  z={q_base[2]:.4f}")

    print(f"Rollout done in {time.time() - t0:.2f}s")
    return X_sim


def main():
    parser = argparse.ArgumentParser(
        description="Prescribed-joint simulation through the hydrodynamic model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ocp", default="task3_solution.npz")
    parser.add_argument("--out", default=None,
                        help="Output path (default: <stem>_rollout.npz)")
    parser.add_argument("--Cd_t", type=float, default=hydro_params.CD_T)
    parser.add_argument("--Cd_a", type=float, default=hydro_params.CD_A)
    parser.add_argument("--Ca_t", type=float, default=hydro_params.CA_T)
    parser.add_argument("--Ca_a", type=float, default=hydro_params.CA_A)
    parser.add_argument("--Cd_lin_t", type=float, default=hydro_params.CD_LIN_T,
                        help="Linear transverse damping coeff (Fossen D_S)")
    parser.add_argument("--Cd_lin_a", type=float, default=hydro_params.CD_LIN_A,
                        help="Linear axial damping coeff (Fossen D_S)")
    parser.add_argument("--v_lin", type=float,
                        default=hydro_params.V_LINEAR_THRESHOLD,
                        help="Linear-damping velocity scale [m/s]: speed about which "
                             "the quadratic drag is linearized (D_S = 0.5*rho*Cd_lin*A*v_lin)")
    parser.add_argument("--leg_thrust_scale", type=float,
                        default=hydro_params.LEG_THRUST_SCALE,
                        help="Scale factor on drag for non-trunk links (wake slip)")
    args = parser.parse_args()

    ocp_path = Path(args.ocp)
    out_path = Path(args.out) if args.out else ocp_path.with_name(
        ocp_path.stem + "_rollout.npz"
    )

    print(f"Loading OCP: {ocp_path}")
    # The robot is read from the solution file.
    X_ocp, U, T, N, nq, robot = load_ocp(str(ocp_path))
    print(f"  Robot: {robot.spec.name}")
    print(f"  X: {X_ocp.shape}, U: {U.shape}, T={T:.3f}s, N={N}, nq={nq}")

    print("\nHydrodynamic parameters:")
    print(f"  Cd_t={args.Cd_t}, Cd_a={args.Cd_a}, Ca_t={args.Ca_t}, Ca_a={args.Ca_a}, "
          f"leg_thrust_scale={args.leg_thrust_scale}")
    print(f"  Cd_lin_t={args.Cd_lin_t}, Cd_lin_a={args.Cd_lin_a}, v_lin={args.v_lin} "
          f"(linear damping)")

    dyn = build_dynamics(args, robot)
    X_sim = rollout(X_ocp, dyn, T, N)

    save_solution(out_path, T=T, X=X_sim, U=U, N=N, nq=nq,
                  robot=robot.spec.name, coords=coords_of(robot))
    print(f"\nSaved: {out_path}")

    dx_ocp = X_ocp[0, -1] - X_ocp[0, 0]
    dx_sim = X_sim[0, -1] - X_sim[0, 0]
    dy_ocp = X_ocp[1, -1] - X_ocp[1, 0]
    dy_sim = X_sim[1, -1] - X_sim[1, 0]
    dz_ocp = X_ocp[2, -1] - X_ocp[2, 0]
    dz_sim = X_sim[2, -1] - X_sim[2, 0]
    print(f"\n  {'':20s}  {'OCP':>10}  {'Rollout':>10}")
    print(f"  {'Δx (forward) [m]':20s}  {dx_ocp:10.4f}  {dx_sim:10.4f}")
    print(f"  {'Δy (lateral) [m]':20s}  {dy_ocp:10.4f}  {dy_sim:10.4f}")
    print(f"  {'Δz (vertical) [m]':20s}  {dz_ocp:10.4f}  {dz_sim:10.4f}")
    print(f"  {'vx peak [m/s]':20s}  {np.max(X_ocp[nq, :]):10.4f}  {np.max(X_sim[nq, :]):10.4f}")


if __name__ == "__main__":
    main()
