#!/usr/bin/env python3
"""
Hydrodynamic parameter sweep to match SPH simulation.

Builds six unit-coefficient CasADi functions once (quadratic drag Cd_t/Cd_a,
linear drag Cd_lin_t/Cd_lin_a, added mass Ca_t/Ca_a, each set to 1 in
isolation), then evaluates any parameter combination cheaply at rollout time
by scaling and summing the pre-compiled components.

This works because every force is linear in its coefficient. The quadratic
and linear drag terms are independently linear (the linear-drag unit
components keep the model default v_linear_threshold):
    tau_drag  = Cd_t * tau_drag_t(q,v)     + Cd_a * tau_drag_a(q,v)
              + Cd_lin_t * tau_drag_lin_t(q,v) + Cd_lin_a * tau_drag_lin_a(q,v)
    M_added   = Ca_t * M_added_t(q)        + Ca_a * M_added_a(q)

Usage:
    python3 sweep_hydro_params.py [--ocp PATH] [--bag PATH] [--start T]
                                  [--method optimize|grid] [--grid-n N]

Options:
    --ocp PATH      OCP solution .npz          (default: /home/ws/task3_solution.npz)
    --bag PATH      SPH rosbag                 (default: sim_log.bag in mlruns)
    --start FLOAT   Bag time [s] at OCP t=0    (default: 0.0)
    --method        optimize (scipy diff-evol) or grid sweep
    --grid-n INT    Grid points per param      (default: 5, grid mode only)
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import differential_evolution

URDF_PATH = Path(__file__).parent.parent / "src" / "amph" / "urdf" / "amph.urdf"
sys.path.insert(0, str(Path(__file__).parent.parent))

from stage1_gait_optimization.hydro_model import QuadrupedRobot, SymbolicDynamics
from stage1_gait_optimization.hydro_model.hydrodynamics import (
    RHO_WATER,
    SymbolicHydrodynamicModel,
)

# ── Realistic parameter bounds [lower, upper] ─────────────────────────────────
# Based on cylinder theory:
#   Cd_t:     cross-flow cylinder drag         0.7 – 1.5
#   Cd_a:     axial blunt-body drag            0.1 – 0.8
#   Ca_t:     transverse added mass (~1 for ∞ cylinder)  0.5 – 1.3
#   Ca_a:     axial added mass (tiny for elongated body) 0.02 – 0.30
#   Cd_lin_t: transverse linear (skin-friction) damping  0.0 – 2.0
#   Cd_lin_a: axial linear damping                       0.0 – 2.0
BOUNDS = [(0.7, 1.5), (0.1, 0.8), (0.5, 1.3), (0.02, 0.30), (0.0, 2.0), (0.0, 2.0)]
PARAM_NAMES = ["Cd_t", "Cd_a", "Ca_t", "Ca_a", "Cd_lin_t", "Cd_lin_a"]

# Objective weights for (x, y, z) RMSE
W_XYZ = np.array([1.0, 0.5, 0.5])


# ── Build ─────────────────────────────────────────────────────────────────────

def build_components(robot):
    """Build the base rigid-body dynamics and four unit hydro components.

    Returns
    -------
    base_dyn : SymbolicDynamics
        Built with all hydro coefficients = 0.
        Provides f_M_rb, f_C_rb, f_g_rb, f_tau_buoyancy.
    components : dict
        "drag_t"     -> SymbolicHydrodynamicModel  (Cd_t=1, others=0)
        "drag_a"     -> SymbolicHydrodynamicModel  (Cd_a=1, others=0)
        "drag_lin_t" -> SymbolicHydrodynamicModel  (Cd_lin_t=1, others=0)
        "drag_lin_a" -> SymbolicHydrodynamicModel  (Cd_lin_a=1, others=0)
        "added_t"    -> SymbolicHydrodynamicModel  (Ca_t=1, others=0)
        "added_a"    -> SymbolicHydrodynamicModel  (Ca_a=1, others=0)
    """
    print("Building base dynamics (no hydro) …")
    t0 = time.time()
    base_dyn = SymbolicDynamics(robot, Cd_t=0, Cd_a=0, Ca_t=0, Ca_a=0)
    print(f"  Done in {time.time() - t0:.1f}s\n")

    # FK was already set up in base_dyn; reuse cmodel/cdata/q/v for unit models.
    # Quadratic-drag units set Cd_lin_*=0 explicitly so the linear damping is
    # carried only by the dedicated drag_lin_* units (otherwise Cd_lin_*
    # defaults to the quadratic Cd and would double-count).
    unit_specs = [
        ("drag_t",     dict(Cd_transverse=1, Cd_axial=0, Cd_lin_transverse=0,
                            Cd_lin_axial=0, Ca_transverse=0, Ca_axial=0)),
        ("drag_a",     dict(Cd_transverse=0, Cd_axial=1, Cd_lin_transverse=0,
                            Cd_lin_axial=0, Ca_transverse=0, Ca_axial=0)),
        ("drag_lin_t", dict(Cd_transverse=0, Cd_axial=0, Cd_lin_transverse=1,
                            Cd_lin_axial=0, Ca_transverse=0, Ca_axial=0)),
        ("drag_lin_a", dict(Cd_transverse=0, Cd_axial=0, Cd_lin_transverse=0,
                            Cd_lin_axial=1, Ca_transverse=0, Ca_axial=0)),
        ("added_t",    dict(Cd_transverse=0, Cd_axial=0, Cd_lin_transverse=0,
                            Cd_lin_axial=0, Ca_transverse=1, Ca_axial=0)),
        ("added_a",    dict(Cd_transverse=0, Cd_axial=0, Cd_lin_transverse=0,
                            Cd_lin_axial=0, Ca_transverse=0, Ca_axial=1)),
    ]

    components = {}
    for name, kwargs in unit_specs:
        print(f"Building unit component '{name}' …")
        t0 = time.time()
        components[name] = SymbolicHydrodynamicModel(
            robot=robot,
            cmodel=base_dyn.cmodel,
            cdata=base_dyn.cdata,
            q_sym=base_dyn.q,
            v_sym=base_dyn.v,
            nv=base_dyn.nv,
            rho=RHO_WATER,
            **kwargs,
        )
        print(f"  Done in {time.time() - t0:.1f}s")

    print()
    return base_dyn, components


def make_eval_fd(base_dyn, components):
    """Return a forward-dynamics callable parametrised by
    (Cd_t, Cd_a, Ca_t, Ca_a, Cd_lin_t, Cd_lin_a).

    Assembles:
        M   = M_rb + Ca_t * M_added_t + Ca_a * M_added_a
        rhs = tau + tau_buoyancy
              + Cd_t * tau_drag_t + Cd_a * tau_drag_a
              + Cd_lin_t * tau_drag_lin_t + Cd_lin_a * tau_drag_lin_a
              - C_rb @ v - g_rb
        qdd = solve(M, rhs)
    """
    f_M_rb  = base_dyn.f_M_rb
    f_C_rb  = base_dyn.f_C_rb
    f_g_rb  = base_dyn.f_g_rb
    f_tau_b = base_dyn.f_tau_buoyancy
    f_dt    = components["drag_t"].f_tau_drag
    f_da    = components["drag_a"].f_tau_drag
    f_lt    = components["drag_lin_t"].f_tau_drag
    f_la    = components["drag_lin_a"].f_tau_drag
    f_at    = components["added_t"].f_M_added
    f_aa    = components["added_a"].f_M_added

    def eval_fd(q, v, tau, Cd_t, Cd_a, Ca_t, Ca_a, Cd_lin_t, Cd_lin_a):
        M = (np.array(f_M_rb(q))
             + Ca_t * np.array(f_at(q))
             + Ca_a * np.array(f_aa(q)))
        C = np.array(f_C_rb(q, v))
        g = np.array(f_g_rb(q)).flatten()
        tau_b = np.array(f_tau_b(q)).flatten()
        tau_d = (Cd_t * np.array(f_dt(q, v)).flatten()
                 + Cd_a * np.array(f_da(q, v)).flatten()
                 + Cd_lin_t * np.array(f_lt(q, v)).flatten()
                 + Cd_lin_a * np.array(f_la(q, v)).flatten())
        rhs = tau + tau_b + tau_d - C @ v - g
        return np.linalg.solve(M, rhs)

    return eval_fd


# ── Rollout ───────────────────────────────────────────────────────────────────

def _dq_base_dt(q_base, v_base):
    qx, qy, qz, qw = q_base[3], q_base[4], q_base[5], q_base[6]
    vx, vy, vz = v_base[0], v_base[1], v_base[2]
    wx, wy, wz = v_base[3], v_base[4], v_base[5]
    dp = np.array([
        (1-2*(qy**2+qz**2))*vx + 2*(qx*qy-qw*qz)*vy + 2*(qx*qz+qw*qy)*vz,
        2*(qx*qy+qw*qz)*vx + (1-2*(qx**2+qz**2))*vy + 2*(qy*qz-qw*qx)*vz,
        2*(qx*qz-qw*qy)*vx + 2*(qy*qz+qw*qx)*vy + (1-2*(qx**2+qy**2))*vz,
    ])
    dquat = np.array([
        0.5*(qw*wx+qy*wz-qz*wy),
        0.5*(qw*wy+qz*wx-qx*wz),
        0.5*(qw*wz+qx*wy-qy*wx),
        0.5*(-qx*wx-qy*wy-qz*wz),
    ])
    return np.concatenate([dp, dquat])


def rollout_fast(X_ocp, U, eval_fd, T, N, nq, Cd_t, Cd_a, Ca_t, Ca_a,
                 Cd_lin_t, Cd_lin_a):
    """RK4 base rollout with prescribed joints and parametric hydro.

    Returns xyz : (3, N+1) world-frame base position.
    """
    dt = T / N
    nv_base = 6
    i_vbase_end = nq + nv_base

    q_base = X_ocp[:7, 0].copy()
    v_base = X_ocp[nq:i_vbase_end, 0].copy()
    xyz = np.zeros((3, N + 1))
    xyz[:, 0] = q_base[:3]

    for k in range(N):
        q_joints = X_ocp[7:nq, k]
        v_joints = X_ocp[i_vbase_end:, k]
        tau_full = np.concatenate([np.zeros(nv_base), U[:, k]])

        def f(qb, vb):
            q_full = np.concatenate([qb, q_joints])
            v_full = np.concatenate([vb, v_joints])
            qdd = eval_fd(q_full, v_full, tau_full, Cd_t, Cd_a, Ca_t, Ca_a,
                          Cd_lin_t, Cd_lin_a)
            return _dq_base_dt(qb, vb), qdd[:nv_base]

        dq1, dv1 = f(q_base, v_base)
        dq2, dv2 = f(q_base + dt/2*dq1, v_base + dt/2*dv1)
        dq3, dv3 = f(q_base + dt/2*dq2, v_base + dt/2*dv2)
        dq4, dv4 = f(q_base + dt*dq3,   v_base + dt*dv3)

        q_base = q_base + (dt/6)*(dq1 + 2*dq2 + 2*dq3 + dq4)
        v_base = v_base + (dt/6)*(dv1 + 2*dv2 + 2*dv3 + dv4)
        q_base[3:7] /= np.linalg.norm(q_base[3:7])
        xyz[:, k+1] = q_base[:3]

    return xyz


# ── Data loading ──────────────────────────────────────────────────────────────

def load_ocp(path):
    d = np.load(path)
    return d["X"], d["U"], float(d["T"]), int(d["N"]), int(d["nq"])


def load_sph_bag(bag_path, start_time, t_ocp):
    """Read /gazebo/model_states from a rosbag and interpolate to OCP grid.

    Returns xyz_aligned : (3, len(t_ocp)) world-frame position [m].
    """
    try:
        import rosbag
    except ImportError:
        sys.path.insert(0, "/opt/ros/noetic/lib/python3/dist-packages")
        import rosbag

    times, xs, ys, zs = [], [], [], []
    model_idx = None

    with rosbag.Bag(bag_path) as bag:
        for _, msg, t in bag.read_messages(topics=["/gazebo/model_states"]):
            if model_idx is None:
                if "amph" not in msg.name:
                    raise ValueError(f"Model 'amph' not in bag. Available: {list(msg.name)}")
                model_idx = list(msg.name).index("amph")
            p = msg.pose[model_idx].position
            xs.append(p.x); ys.append(p.y); zs.append(p.z)
            times.append(t.to_sec())

    times = np.array(times) - times[0]
    xyz_raw = np.array([xs, ys, zs])
    t_query = t_ocp + start_time

    xyz_aligned = np.zeros((3, len(t_ocp)))
    for i in range(3):
        f = interp1d(times, xyz_raw[i], kind="linear",
                     bounds_error=False, fill_value=(xyz_raw[i, 0], xyz_raw[i, -1]))
        xyz_aligned[i] = f(t_query)
    return xyz_aligned


# ── Objective ─────────────────────────────────────────────────────────────────

def make_objective(X_ocp, U, eval_fd, T, N, nq, xyz_ref):
    """Return objective(params) -> scalar loss.

    Loss = weighted sum of per-axis RMSE on relative displacement (xyz - xyz[0]).
    """
    xyz_ref_rel = xyz_ref - xyz_ref[:, [0]]
    eval_count = [0]

    def objective(params):
        Cd_t, Cd_a, Ca_t, Ca_a, Cd_lin_t, Cd_lin_a = params
        xyz = rollout_fast(X_ocp, U, eval_fd, T, N, nq,
                           Cd_t, Cd_a, Ca_t, Ca_a, Cd_lin_t, Cd_lin_a)
        xyz -= xyz[:, [0]]
        rmse = np.sqrt(np.mean((xyz - xyz_ref_rel)**2, axis=1))  # (3,)
        loss = float(W_XYZ @ rmse)
        eval_count[0] += 1
        if eval_count[0] % 10 == 0:
            print(f"  eval {eval_count[0]:4d}: "
                  f"Cd_t={Cd_t:.3f} Cd_a={Cd_a:.3f} "
                  f"Ca_t={Ca_t:.3f} Ca_a={Ca_a:.3f} "
                  f"Cd_lin_t={Cd_lin_t:.3f} Cd_lin_a={Cd_lin_a:.3f}  "
                  f"rmse x={rmse[0]:.4f} y={rmse[1]:.4f} z={rmse[2]:.4f}  "
                  f"loss={loss:.5f}")
        return loss

    return objective


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    _DEFAULT_BAG = (
        "/home/ws/experiment_results/mlruns/1"
        "/1d12383d7c0f4d31b022574db8838084/artifacts/sim_log.bag"
    )
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--ocp",    default="/home/ws/task3_solution.npz")
    parser.add_argument("--bag",    default=_DEFAULT_BAG)
    parser.add_argument("--start",  type=float, default=0.0,
                        help="Bag time [s] corresponding to OCP t=0")
    parser.add_argument("--method", choices=["optimize", "grid"], default="optimize")
    parser.add_argument("--grid-n", type=int, default=5,
                        help="Grid points per parameter (grid mode only)")
    args = parser.parse_args()

    # ── Robot & component functions ────────────────────────────────────────
    robot = QuadrupedRobot(URDF_PATH)
    q_n = robot.neutral_config()
    robot.forward_kinematics(q_n)
    robot.build_cylinders()

    base_dyn, components = build_components(robot)
    eval_fd = make_eval_fd(base_dyn, components)

    # ── Load OCP & SPH data ────────────────────────────────────────────────
    X_ocp, U, T, N, nq = load_ocp(args.ocp)
    t_ocp = np.linspace(0.0, T, N + 1)
    print(f"OCP: N={N}, T={T:.3f}s, nq={nq}")

    print(f"\nLoading SPH bag: {args.bag}")
    xyz_sph = load_sph_bag(args.bag, args.start, t_ocp)
    xyz_rel = xyz_sph - xyz_sph[:, [0]]
    print(f"  SPH Δx={xyz_rel[0,-1]:.4f} m  Δy={xyz_rel[1,-1]:.4f} m  Δz={xyz_rel[2,-1]:.4f} m\n")

    objective = make_objective(X_ocp, U, eval_fd, T, N, nq, xyz_sph)

    # ── Optimise or grid ───────────────────────────────────────────────────
    if args.method == "optimize":
        print("Running differential_evolution …")
        print(f"  Bounds: {dict(zip(PARAM_NAMES, BOUNDS))}\n")
        result = differential_evolution(
            objective,
            BOUNDS,
            maxiter=300,
            tol=1e-5,
            seed=42,
            workers=1,
            disp=True,
            init="sobol",
        )
        best = result.x
        best_loss = result.fun

    else:  # grid
        n = args.grid_n
        ndim = len(BOUNDS)
        print(f"Grid sweep: {n}^{ndim} = {n**ndim} evaluations …\n")
        grids = [np.linspace(lo, hi, n) for (lo, hi) in BOUNDS]
        mesh = np.meshgrid(*grids, indexing="ij")
        shape = mesh[0].shape
        losses = np.empty(shape)
        all_params = np.stack([m.ravel() for m in mesh], axis=1)
        all_losses = np.array([objective(p) for p in all_params])
        losses = all_losses.reshape(shape)
        best_idx = np.unravel_index(np.argmin(losses), shape)
        best = np.array([mesh[i][best_idx] for i in range(ndim)])
        best_loss = float(losses[best_idx])


    # ── Report ─────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("Best parameters:")
    for name, val in zip(PARAM_NAMES, best):
        lo, hi = BOUNDS[PARAM_NAMES.index(name)]
        note = "  ← at bound!" if abs(val - lo) < 1e-4 or abs(val - hi) < 1e-4 else ""
        print(f"  {name:<6} = {val:.4f}  (range [{lo}, {hi}]){note}")
    print(f"Loss = {best_loss:.6f}")

    # Final rollout for comparison
    xyz_best = rollout_fast(X_ocp, U, eval_fd, T, N, nq, *best)
    xyz_best_rel = xyz_best - xyz_best[:, [0]]
    print("\nFinal vs SPH displacement:")
    print(f"  {'':6}  {'Model':>10}  {'SPH':>10}  {'Error':>10}")
    for i, axis in enumerate(["Δx", "Δy", "Δz"]):
        model_val = xyz_best_rel[i, -1]
        sph_val   = xyz_rel[i, -1]
        print(f"  {axis}     {model_val:10.4f}  {sph_val:10.4f}  {model_val-sph_val:10.4f} m")



if __name__ == "__main__":
    main()
