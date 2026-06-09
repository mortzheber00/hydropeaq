#!/usr/bin/env python3
"""
Prescribed-joint simulation of an OCP solution through the symbolic hydrodynamics.

Joint angles and velocities are taken directly from the OCP reference at each
step (matching what a position controller does in Gazebo). Only the base state
(position + orientation + base velocity) is integrated forward under the
hydrodynamic forces.

This isolates the effect of hydrodynamic parameter changes on the base
trajectory, making it the right tool for tuning Cd_t / Cd_a against a bag.

Usage:
    python3 simulate_ocp.py [options]

Options:
    --ocp PATH          Input OCP .npz                    (default: task3_solution.npz)
    --out PATH          Output .npz path                  (default: <stem>_rollout.npz)
    --Cd_t FLOAT        Transverse drag coefficient        (default: 1.0)
    --Cd_a FLOAT        Axial drag coefficient             (default: 0.8)
    --Ca_t FLOAT        Transverse added-mass coefficient  (default: 1.0)
    --Ca_a FLOAT        Axial added-mass coefficient       (default: 0.1)
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

URDF_PATH = Path(__file__).parent.parent / "src" / "amph" / "urdf" / "amph.urdf"

sys.path.insert(0, str(Path(__file__).parent.parent))
from stage1_gait_optimization.hydro_model import QuadrupedRobot, SymbolicDynamics


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_ocp(path: str):
    d = np.load(path)
    return d["X"], d["U"], float(d["T"]), int(d["N"]), int(d["nq"])


def build_dynamics(args) -> SymbolicDynamics:
    print("Building robot model …")
    robot = QuadrupedRobot(URDF_PATH)
    robot.forward_kinematics(np.zeros(robot.nq))
    robot.build_cylinders()

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
    """Time derivative of the base configuration [x,y,z, qx,qy,qz,qw].

    Mirrors SymbolicDynamics._dq_dt but evaluated numerically.
    v_base = [vx, vy, vz, wx, wy, wz] in body frame.
    """
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
    q_joints: np.ndarray,
    v_joints: np.ndarray,
    tau_full: np.ndarray,
    dyn: SymbolicDynamics,
    dt: float,
):
    """RK4 step for the base DOF only, with joints prescribed.

    Assembles the full (q, v) from the base rollout state and the prescribed
    joint state, calls forward dynamics, and uses only the base acceleration
    qdd[0:6] for integration.

    Joint torques are included because they couple into base acceleration
    through the off-diagonal blocks of the mass matrix.
    """
    def f(qb, vb):
        q_full = np.concatenate([qb, q_joints])
        v_full = np.concatenate([vb, v_joints])
        qdd = dyn.eval_forward_dynamics(q_full, v_full, tau_full)
        return dq_base_dt(qb, vb), qdd[:6]

    dq1, dv1 = f(q_base, v_base)
    dq2, dv2 = f(q_base + dt/2*dq1, v_base + dt/2*dv1)
    dq3, dv3 = f(q_base + dt/2*dq2, v_base + dt/2*dv2)
    dq4, dv4 = f(q_base + dt*dq3,   v_base + dt*dv3)

    q_next = q_base + (dt/6) * (dq1 + 2*dq2 + 2*dq3 + dq4)
    v_next = v_base + (dt/6) * (dv1 + 2*dv2 + 2*dv3 + dv4)

    q_next[3:7] /= np.linalg.norm(q_next[3:7])
    return q_next, v_next


# ── Rollout ───────────────────────────────────────────────────────────────────

def rollout(
    X_ocp: np.ndarray,
    U: np.ndarray,
    dyn: SymbolicDynamics,
    T: float,
    N: int,
):
    """Prescribed-joint rollout: joints follow X_ocp; only the base is integrated.

    Returns X_sim : (nq+nv, N+1).
    """
    nq, nv = dyn.nq, dyn.nv
    dt = T / N

    # State index boundaries
    # q:  [0:3]=pos, [3:7]=quat, [7:nq]=joints
    # v:  [nq:nq+6]=base vel,   [nq+6:]=joint vel
    i_vbase_end = nq + 6

    X_sim = np.zeros((nq + nv, N + 1))
    X_sim[:, 0] = X_ocp[:, 0]

    print(f"Rolling out {N} steps (dt={dt:.4f}s, T={T:.3f}s, mode=prescribed-joint) …")
    t0 = time.time()

    q_base = X_ocp[:7, 0].copy()
    v_base = X_ocp[nq:i_vbase_end, 0].copy()
    for k in range(N):
        q_joints = X_ocp[7:nq, k]
        v_joints = X_ocp[i_vbase_end:, k]
        tau_full = np.concatenate([np.zeros(6), U[:, k]])

        q_base, v_base = rk4_base_step(
            q_base, v_base, q_joints, v_joints, tau_full, dyn, dt
        )

        X_sim[:7, k+1]             = q_base
        X_sim[7:nq, k+1]           = X_ocp[7:nq, k+1]
        X_sim[nq:i_vbase_end, k+1] = v_base
        X_sim[i_vbase_end:, k+1]   = X_ocp[i_vbase_end:, k+1]

        if (k + 1) % 10 == 0 or k == N - 1:
            print(f"  step {k+1:3d}/{N}  x={q_base[0]:.4f}  z={q_base[2]:.4f}")

    print(f"Rollout done in {time.time() - t0:.2f}s")
    return X_sim


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Prescribed-joint simulation through the hydrodynamic model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ocp", default="task3_solution.npz")
    parser.add_argument("--out", default=None,
                        help="Output path (default: <stem>_rollout.npz)")
    parser.add_argument("--Cd_t", type=float, default=1.0)
    parser.add_argument("--Cd_a", type=float, default=0.8)
    parser.add_argument("--Ca_t", type=float, default=1.0)
    parser.add_argument("--Ca_a", type=float, default=0.1)
    parser.add_argument("--Cd_lin_t", type=float, default=None,
                        help="Linear transverse damping coeff (Fossen D_S; default: = Cd_t)")
    parser.add_argument("--Cd_lin_a", type=float, default=None,
                        help="Linear axial damping coeff (Fossen D_S; default: = Cd_a)")
    parser.add_argument("--v_lin", type=float, default=0.7,
                        help="Linear-damping velocity scale [m/s]: speed about which "
                             "the quadratic drag is linearized (D_S = 0.5*rho*Cd_lin*A*v_lin)")
    parser.add_argument("--leg_thrust_scale", type=float, default=1.0,
                        help="Scale factor on drag for non-trunk links (wake slip)")
    args = parser.parse_args()

    ocp_path = Path(args.ocp)
    out_path = Path(args.out) if args.out else ocp_path.with_name(
        ocp_path.stem + "_rollout.npz"
    )

    print(f"Loading OCP: {ocp_path}")
    X_ocp, U, T, N, nq = load_ocp(str(ocp_path))
    print(f"  X: {X_ocp.shape}, U: {U.shape}, T={T:.3f}s, N={N}, nq={nq}")

    print("\nHydrodynamic parameters:")
    print(f"  Cd_t={args.Cd_t}, Cd_a={args.Cd_a}, Ca_t={args.Ca_t}, Ca_a={args.Ca_a}, "
          f"leg_thrust_scale={args.leg_thrust_scale}")
    print(f"  Cd_lin_t={args.Cd_lin_t}, Cd_lin_a={args.Cd_lin_a}, v_lin={args.v_lin} "
          f"(linear damping)")

    dyn = build_dynamics(args)
    X_sim = rollout(X_ocp, U, dyn, T, N)

    np.savez(out_path, X=X_sim, U=U, T=T, N=N, nq=nq)
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
