#!/usr/bin/env python3
"""Grid-search fit of the hydrodynamic coefficients to SPH replays.

Every force is linear in its coefficient, so each term is compiled once with a
unit coefficient and any parameter set is evaluated by scaling and summing:
    tau_drag  = Cd_t * tau_drag_t + Cd_a * tau_drag_a
              + Cd_lin_t * tau_drag_lin_t + Cd_lin_a * tau_drag_lin_a
    M_added   = Ca_t * M_added_t + Ca_a * M_added_a

Several (--ocp, --bag) pairs are fitted jointly; each run's loss is normalised
by its own SPH cycle distance. --start is the bag time of OCP t=0 and rarely 0,
since the replay first moves to the start pose.

Usage:
  python stage2_sim_validation/hydro_calibration/sweep_hydro_params.py \\
      --ocp run1.npz run2.npz --bag run1.bag run2.bag --start 1.49 --grid-n 5
  python stage2_sim_validation/hydro_calibration/sweep_hydro_params.py --fix Ca_t=1.0 Ca_a=0.05
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

ROBOT = "amph"   # registered robot name; see hydro_model/robots/
sys.path.insert(0, str(Path(__file__).parents[2]))

from stage1_gait_optimization.hydro_model import SymbolicDynamics, get_spec, load_robot
from stage1_gait_optimization.hydro_model.hydrodynamics import (
    RHO_WATER,
    SymbolicHydrodynamicModel,
)
from stage1_gait_optimization.hydro_model.trajectory import (
    expand_to_tree,
    load_solution,
)

# --- Search bounds ---
# Flow regime of the paddles: KC ~ 31-35, Re ~ 6e4. The legs act like flat
# plates rather than cylinders, so Cd_t is expected around 2 or higher and Ca_t
# around 1 (plate of width c, referenced to the circle of diameter c).
#
#   name      search        basis
#   Cd_t      2.0 - 5.0     flat plate normal to flow, ~2 steady, more when oscillating
#   Cd_a      0.0 - 2.0     axial blunt body, 0.1-1.0 expected
#   Ca_t      0.6 - 2.4     kept away from 0 so the fit cannot drop the fluid inertia
#   Ca_a      0.05 - 0.40   Lamb's k1 for the link fineness ratios (0.14-0.21)
#   Cd_lin_*  0.0 - 5.0     linearisation scale tied to V_LINEAR_THRESHOLD
BOUNDS = [(2.0, 5.0), (0.0, 2.0), (0.6, 2.4), (0.05, 0.40), (0.0, 5.0), (0.0, 5.0)]
PARAM_NAMES = ["Cd_t", "Cd_a", "Ca_t", "Ca_a", "Cd_lin_t", "Cd_lin_a"]

# Objective weights for (x, y, z) RMSE
W_XYZ = np.array([1.0, 0.01, 0.0])

# Loss for diverged rollouts; far above any converged loss (~0.1)
DIVERGED_LOSS = 1e3


# --- Model components ---

def build_components(robot):
    """Build the hydro-free base dynamics and the six unit hydro components.

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

    # Reuse base_dyn's CasADi model and symbols. Set every coefficient
    # explicitly; omitted ones would take their fitted defaults.
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
    """Base acceleration ``eval_fd(q, v, a_joints, *coefficients)`` with prescribed joints.

        M      = M_rb + Ca_t * M_added_t + Ca_a * M_added_a
        rhs    = tau_buoyancy + sum(Cd_i * tau_drag_i)
                 - C_rb @ v - (Ca_t * C_A_t_v + Ca_a * C_A_a_v) - g_rb
        a_base = solve(M[:6, :6], rhs[:6] - M[:6, 6:] @ a_joints)

    Joint torques are not used: the joints are position-controlled, so only the
    unactuated base rows apply (as in the OCP). C_A·v is tau_added at a = 0.
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
    f_ct    = components["added_t"].f_tau_added
    f_ca    = components["added_a"].f_tau_added
    zero_a  = np.zeros(base_dyn.nv)
    nv_base = 6

    diverged = np.full(nv_base, np.nan)

    def eval_fd(q, v, a_joints, Cd_t, Cd_a, Ca_t, Ca_a, Cd_lin_t, Cd_lin_a):
        # Diverged state: return NaN so rollout_fast aborts.
        if not (np.isfinite(q).all() and np.isfinite(v).all()):
            return diverged
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
        C_A_v = (Ca_t * np.array(f_ct(q, v, zero_a)).flatten()
                 + Ca_a * np.array(f_ca(q, v, zero_a)).flatten())
        rhs = tau_b + tau_d - C @ v - C_A_v - g
        try:
            return np.linalg.solve(M[:nv_base, :nv_base],
                                   rhs[:nv_base] - M[:nv_base, nv_base:] @ a_joints)
        except np.linalg.LinAlgError:
            # Singular mass matrix from a diverging state
            return diverged

    return eval_fd


# --- Rollout ---

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


def rollout_fast(X_ocp, eval_fd, T, N, nq, Cd_t, Cd_a, Ca_t, Ca_a,
                 Cd_lin_t, Cd_lin_a):
    """RK4 base rollout with prescribed joints; returns base position (3, N+1).

    Same scheme as simulate_ocp.rk4_base_step: joints interpolated at each
    stage, joint acceleration constant per interval.
    """
    dt = T / N
    nv_base = 6
    i_vbase_end = nq + nv_base

    q_base = X_ocp[:7, 0].copy()
    v_base = X_ocp[nq:i_vbase_end, 0].copy()
    xyz = np.zeros((3, N + 1))
    xyz[:, 0] = q_base[:3]

    for k in range(N):
        q_joints_k,  q_joints_k1 = X_ocp[7:nq, k],        X_ocp[7:nq, k+1]
        v_joints_k,  v_joints_k1 = X_ocp[i_vbase_end:, k], X_ocp[i_vbase_end:, k+1]
        a_joints = (v_joints_k1 - v_joints_k) / dt

        def f(qb, vb, s):
            q_joints = (1 - s) * q_joints_k + s * q_joints_k1
            v_joints = (1 - s) * v_joints_k + s * v_joints_k1
            q_full = np.concatenate([qb, q_joints])
            v_full = np.concatenate([vb, v_joints])
            a_base = eval_fd(q_full, v_full, a_joints, Cd_t, Cd_a, Ca_t, Ca_a,
                             Cd_lin_t, Cd_lin_a)
            return _dq_base_dt(qb, vb), a_base

        dq1, dv1 = f(q_base, v_base, 0.0)
        dq2, dv2 = f(q_base + dt/2*dq1, v_base + dt/2*dv1, 0.5)
        dq3, dv3 = f(q_base + dt/2*dq2, v_base + dt/2*dv2, 0.5)
        dq4, dv4 = f(q_base + dt*dq3,   v_base + dt*dv3,   1.0)

        q_base = q_base + (dt/6)*(dq1 + 2*dq2 + 2*dq3 + dq4)
        v_base = v_base + (dt/6)*(dv1 + 2*dv2 + 2*dv3 + dv4)
        q_base[3:7] /= np.linalg.norm(q_base[3:7])
        xyz[:, k+1] = q_base[:3]

        # Fixed-step RK4 is unstable for large damping (Cd_lin_*); abort.
        if not np.isfinite(q_base).all() or not np.isfinite(v_base).all():
            xyz[:, k+1:] = np.nan
            break

    return xyz


# --- Data loading ---

def load_ocp(path, robot=None):
    """``(X, U, T, N, nq)`` in tree coordinates."""
    d = load_solution(path)
    if robot is None:
        robot = load_robot(d["robot"])
    X = expand_to_tree(robot, d["X"], d["nq"])
    nq = robot.nq if X is not d["X"] else d["nq"]
    return X, d["U"], d["T"], d["N"], nq


def load_sph_bag(bag_path, start_time, t_ocp):
    """Base position (3, len(t_ocp)) from ``/gazebo/model_states``, resampled to the OCP grid."""
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
                model = get_spec(ROBOT).ros
                if model not in msg.name:
                    raise ValueError(f"Model {model!r} not in bag. Available: {list(msg.name)}")
                model_idx = list(msg.name).index(model)
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


# --- Objective ---

def dataset_loss(ds, eval_fd, params):
    """``(loss, rmse)``: weighted RMSE divided by the run's SPH cycle distance, and per-axis RMSE [m]."""
    xyz = rollout_fast(ds["X"], eval_fd, ds["T"], ds["N"], ds["nq"], *params)
    xyz -= xyz[:, [0]]
    rmse = np.sqrt(np.mean((xyz - ds["xyz_ref_rel"])**2, axis=1))  # (3,)
    return float(W_XYZ @ rmse) / ds["d_cycle"], rmse


def make_objective(datasets, eval_fd, expand=None):
    """Mean ``dataset_loss`` over all datasets as a function of the search vector.

    ``expand`` maps the search vector to all six coefficients (for ``--fix``).
    """
    eval_count = [0]

    def objective(search_vals):
        params = search_vals if expand is None else expand(search_vals)
        losses, rmses = [], []
        for ds in datasets:
            loss_i, rmse_i = dataset_loss(ds, eval_fd, params)
            losses.append(loss_i)
            rmses.append(rmse_i)

        loss = float(np.mean(losses))
        if not np.isfinite(loss):
            # Finite penalty; NaN would break argmin.
            loss = DIVERGED_LOSS

        eval_count[0] += 1
        if eval_count[0] % 10 == 0:
            rmse = np.mean(rmses, axis=0)
            print(f"  eval {eval_count[0]:4d}: "
                  f"Cd_t={params[0]:.3f} Cd_a={params[1]:.3f} "
                  f"Ca_t={params[2]:.3f} Ca_a={params[3]:.3f} "
                  f"Cd_lin_t={params[4]:.3f} Cd_lin_a={params[5]:.3f}  "
                  f"mean rmse x={rmse[0]:.4f} y={rmse[1]:.4f} z={rmse[2]:.4f}  "
                  f"loss={loss:.5f}")
        return loss

    return objective


def main():
    _DEFAULT_BAG = (
        "/home/ws/experiment_results/mlruns/1"
        "/1d12383d7c0f4d31b022574db8838084/artifacts/sim_log.bag"
    )
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--ocp",    nargs="+", default=["/home/ws/task3_solution.npz"],
                        help="One or more OCP solutions, paired with --bag by position")
    parser.add_argument("--bag",    nargs="+", default=[_DEFAULT_BAG],
                        help="One bag per --ocp; several are fitted jointly")
    parser.add_argument("--start",  nargs="+", type=float, default=[0.0],
                        help="Bag time [s] at OCP t=0; one value, or one per bag")
    parser.add_argument("--fix", nargs="*", default=[], metavar="NAME=VALUE",
                        help="Hold a coefficient at a value instead of fitting it, "
                             "e.g. --fix Ca_t=1.0 Ca_a=0.05")
    parser.add_argument("--grid-n", type=int, default=5,
                        help="Grid points per parameter")
    args = parser.parse_args()

    if len(args.ocp) != len(args.bag):
        parser.error(f"{len(args.ocp)} --ocp against {len(args.bag)} --bag; "
                     "they are paired by position")
    starts = args.start * len(args.bag) if len(args.start) == 1 else args.start
    if len(starts) != len(args.bag):
        parser.error(f"{len(args.start)} --start values for {len(args.bag)} bags; "
                     "give one value or one per bag")

    fixed = {}
    for item in args.fix:
        name, sep, value = item.partition("=")
        if not sep or name not in PARAM_NAMES:
            parser.error(f"--fix expects NAME=VALUE with NAME in {PARAM_NAMES}, "
                         f"got {item!r}")
        fixed[name] = float(value)
    free = [i for i, n in enumerate(PARAM_NAMES) if n not in fixed]
    if not free:
        parser.error("every coefficient is fixed; nothing left to fit")

    def expand(free_vals):
        """Search vector -> all six coefficients."""
        p = np.empty(len(PARAM_NAMES))
        for n, v in fixed.items():
            p[PARAM_NAMES.index(n)] = v
        p[free] = free_vals
        return p

    # --- Model ---
    robot = load_robot(ROBOT)
    q_n = robot.neutral_config()
    robot.forward_kinematics(q_n)
    robot.build_cylinders()

    base_dyn, components = build_components(robot)
    eval_fd = make_eval_fd(base_dyn, components)

    # --- Data ---
    datasets = []
    for ocp_path, bag_path, start in zip(args.ocp, args.bag, starts):
        X_ocp, _U, T, N, nq = load_ocp(ocp_path)
        t_ocp = np.linspace(0.0, T, N + 1)
        xyz_sph = load_sph_bag(bag_path, start, t_ocp)
        xyz_ref_rel = xyz_sph - xyz_sph[:, [0]]
        d_cycle = float(np.linalg.norm(xyz_ref_rel[:, -1]))
        if d_cycle < 1e-9:
            raise ValueError(
                f"{bag_path}: SPH moves {d_cycle:.2e} m over t=0…{T:.3f}s from "
                f"--start {start}. Nothing to normalise by — check the offset."
            )
        datasets.append(dict(name=Path(ocp_path).parent.parent.name[:8],
                             X=X_ocp, T=T, N=N, nq=nq,
                             xyz_ref_rel=xyz_ref_rel, d_cycle=d_cycle))
        print(f"{datasets[-1]['name']}: N={N}, T={T:.3f}s, start={start}  "
              f"SPH Δx={xyz_ref_rel[0,-1]:.4f} Δy={xyz_ref_rel[1,-1]:.4f} "
              f"Δz={xyz_ref_rel[2,-1]:.4f} m  |d|={d_cycle:.4f} m")

    print(f"\nFitting {len(datasets)} dataset(s) jointly.")
    if fixed:
        print("  Held constant: "
              + ", ".join(f"{n}={v:g}" for n, v in fixed.items()))
    print()
    objective = make_objective(datasets, eval_fd, expand)
    search_bounds = [BOUNDS[i] for i in free]

    # --- Grid search ---
    n = args.grid_n
    ndim = len(search_bounds)
    print(f"Grid sweep: {n}^{ndim} = {n**ndim} evaluations …\n")
    grids = [np.linspace(lo, hi, n) for (lo, hi) in search_bounds]
    mesh = np.meshgrid(*grids, indexing="ij")
    shape = mesh[0].shape
    all_params = np.stack([m.ravel() for m in mesh], axis=1)
    all_losses = np.array([objective(p) for p in all_params])
    losses = all_losses.reshape(shape)
    best_idx = np.unravel_index(np.argmin(losses), shape)
    best = expand(np.array([mesh[i][best_idx] for i in range(ndim)]))
    best_loss = float(losses[best_idx])


    # --- Report ---
    print("\n" + "="*60)
    print("Best parameters:")
    for i, (name, val) in enumerate(zip(PARAM_NAMES, best)):
        if name in fixed:
            print(f"  {name:<8} = {val:.4f}  (fixed)")
            continue
        lo, hi = BOUNDS[i]
        note = "  ← at bound!" if abs(val - lo) < 1e-4 or abs(val - hi) < 1e-4 else ""
        print(f"  {name:<8} = {val:.4f}  (range [{lo}, {hi}]){note}")
    print(f"Loss = {best_loss:.6f}")

    # Report each dataset separately; a good mean fit can hide a poor run.
    for ds in datasets:
        loss_i, _ = dataset_loss(ds, eval_fd, best)
        xyz_best = rollout_fast(ds["X"], eval_fd, ds["T"], ds["N"], ds["nq"], *best)
        xyz_best_rel = xyz_best - xyz_best[:, [0]]
        print(f"\n{ds['name']} — final vs SPH over t=0…{ds['T']:.3f}s "
              f"(loss {loss_i:.5f}):")
        print(f"  {'':6}  {'Model [m]':>10}  {'SPH [m]':>10}  {'Error [m]':>10}  {'Rel.':>9}")
        for i, axis in enumerate(["Δx", "Δy", "Δz"]):
            model_val = xyz_best_rel[i, -1]
            sph_val   = ds["xyz_ref_rel"][i, -1]
            err = model_val - sph_val
            # Relative to SPH; undefined near zero.
            rel = f"{100 * err / abs(sph_val):.2f}%" if abs(sph_val) > 1e-9 else "n/a"
            print(f"  {axis:<6}  {model_val:10.4f}  {sph_val:10.4f}  {err:10.4f}  {rel:>9}")



if __name__ == "__main__":
    main()
