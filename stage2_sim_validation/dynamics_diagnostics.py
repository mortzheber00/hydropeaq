#!/usr/bin/env python3
"""
Decompose forces/inertias acting on the robot through a saved OCP trajectory.

Per-step breakdown of:
  - M_rb vs M_added (sizes, off-diagonal coupling base↔joints)
  - tau_drag (per-link, axial vs transverse, x/y/z)
  - tau_buoyancy, gravity
  - Forward (world-x) base acceleration sources
  - Sanity check: midpoint-only vs integrated drag on a rotating link

Run:  python stage1_gait_optimization/dynamics_diagnostics.py [PATH]
"""

import sys
from pathlib import Path

import casadi as ca
import numpy as np
import pinocchio as pin
import pinocchio.casadi as cpin

sys.path.insert(0, str(Path(__file__).parents[1]))
from stage1_gait_optimization.hydro_model import SymbolicDynamics, fn_name, load_robot
from stage1_gait_optimization.hydro_model.hydrodynamics import (
    SymbolicHydrodynamicModel,
    _skew,
)
from stage1_gait_optimization.hydro_model.trajectory import (
    expand_to_tree,
    load_solution,
)


def build_per_link_diagnostics(robot, dyn):
    """For each link, build a ca.Function returning (F_drag, v_link, alpha)."""
    q = ca.SX.sym("q", dyn.nq)
    v = ca.SX.sym("v", dyn.nv)
    cmodel, cdata = dyn.cmodel, dyn.cdata
    cpin.forwardKinematics(cmodel, cdata, q, v, ca.SX.zeros(dyn.nv))
    cpin.updateFramePlacements(cmodel, cdata)

    hyd = SymbolicHydrodynamicModel(
        robot=robot, cmodel=cmodel, cdata=cdata,
        q_sym=q, v_sym=v, nv=dyn.nv,
        Cd_transverse=dyn.Cd_t, Cd_axial=dyn.Cd_a,
        Ca_transverse=dyn.Ca_t, Ca_axial=dyn.Ca_a,
        leg_thrust_scale=dyn.leg_thrust_scale,
    )

    per_link = {}
    for link in robot.links.values():
        cyl = link.cylinder
        if cyl is None:
            continue
        fid = link.frame_id
        oMf = cdata.oMf[fid]
        R_sym = oMf.rotation
        J_full = cpin.computeFrameJacobian(cmodel, cdata, q, fid,
                                           pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        r_offset = R_sym @ ca.SX(cyl.center_local)
        Jv = J_full[:3, :] - _skew(r_offset) @ J_full[3:, :]
        v_link = Jv @ v
        axis_sym = R_sym @ ca.SX(cyl.axis_local)
        axis_sym = axis_sym / (ca.norm_2(axis_sym) + 1e-15)
        alpha = hyd.submersion_ratio(cyl, axis_sym, oMf, R_sym)
        F_drag = hyd.drag_force(cyl, alpha, axis_sym, v_link)
        scale = 1.0 if link.name == "base_link" else dyn.leg_thrust_scale
        F_drag = scale * F_drag

        per_link[link.name] = ca.Function(
            fn_name("diag", link.name),
            [q, v],
            [F_drag, v_link, alpha],
            ["q", "v"],
            ["F_drag", "v_link", "alpha"],
        )
    return per_link


def quat_to_R_x_row(q4):
    """Return world-x row of rotation matrix from quaternion (qx,qy,qz,qw)."""
    qx, qy, qz, qw = q4
    return np.array([
        1 - 2*(qy**2 + qz**2),
        2*(qx*qy - qw*qz),
        2*(qx*qz + qw*qy),
    ])



def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "task3_solution.npz"
    print(f"Loading: {path}")
    d = load_solution(path)
    X, U, T, N, nq = d["X"], d["U"], d["T"], d["N"], d["nq"]

    print(f"Building robot+dynamics ({d['robot']}) …")
    robot = load_robot(d["robot"])
    dyn = SymbolicDynamics(robot)
    nv = dyn.nv

    # Two views of the same trajectory: the per-link hydrodynamic terms are
    # tree-space quantities, while the equations of motion are integrated in
    # the robot's own (possibly reduced) coordinates.  They coincide for a
    # serial robot.
    X_r, nq_r = X, nq
    X, nq = expand_to_tree(robot, X, nq), robot.nq

    per_link = build_per_link_diagnostics(robot, dyn)
    link_names = list(per_link.keys())
    n_links = len(link_names)

    # ── Static info ─────────────────────────────────────────────────────────
    print("\n" + "─" * 88)
    print("Per-link cylinder primitives:")
    print(f"  {'link':30s} {'L[m]':>7} {'D[m]':>7} {'V[L]':>8} "
          f"{'A_a[cm²]':>10} {'A_t[cm²]':>10}")
    for name in link_names:
        cyl = robot.links[name].cylinder
        print(f"  {name:30s} {cyl.length:7.3f} {2*cyl.radius:7.3f} "
              f"{cyl.volume_displaced*1000:8.3f} "
              f"{cyl.cross_section_axial*1e4:10.2f} "
              f"{cyl.cross_section_transverse*1e4:10.2f}")

    # ── Static mass matrix comparison ──────────────────────────────────────
    q0 = X[:nq, 0]
    M_rb_0 = np.array(dyn.f_M_rb(q0))
    M_A_0  = np.array(dyn.f_M_added(q0))
    print("\nMass matrix at t=0:")
    print(f"  M_rb base diag (m,m,m,Ix,Iy,Iz) = "
          f"{np.array2string(np.diag(M_rb_0[:6,:6]), precision=3)}")
    print(f"  M_A  base diag                  = "
          f"{np.array2string(np.diag(M_A_0[:6,:6]), precision=3)}")
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.diag(M_A_0[:6,:6]) / np.diag(M_rb_0[:6,:6])
    print(f"  M_A / M_rb (base)               = "
          f"{np.array2string(ratio, precision=3)}")
    print(f"  ‖M_A base↔joint coupling‖_F     = {np.linalg.norm(M_A_0[:6,6:],'fro'):.4f}")
    print(f"  ‖M_rb base↔joint coupling‖_F    = {np.linalg.norm(M_rb_0[:6,6:],'fro'):.4f}")

    # ── Walk the trajectory ────────────────────────────────────────────────
    F_drag_x  = np.zeros((n_links, N+1))
    v_link_x  = np.zeros((n_links, N+1))
    alpha_    = np.zeros((n_links, N+1))
    tau_drag  = np.zeros((nv, N+1))
    tau_buoy  = np.zeros((nv, N+1))
    grav      = np.zeros((nv, N+1))
    a_world_x = np.zeros(N+1)
    a_body    = np.zeros((6, N+1))
    vx_world  = np.zeros(N+1)

    for k in range(N+1):
        q = X[:nq, k]
        v = X[nq:, k]
        tau_r = np.concatenate([np.zeros(6), U[:, min(k, N - 1)]])

        tau_drag[:, k] = np.array(dyn.f_tau_drag(q, v)).flatten()
        tau_buoy[:, k] = np.array(dyn.f_tau_buoyancy(q)).flatten()
        grav[:, k]     = np.array(dyn.f_g_rb(q)).flatten()

        a = dyn.eval_reduced_forward_dynamics(
            X_r[:7, k], X_r[7:nq_r, k], X_r[nq_r:, k], tau_r
        )
        a_body[:, k] = a[:6]
        Rx_row = quat_to_R_x_row(q[3:7])
        a_world_x[k] = Rx_row @ a[:3]
        vx_world[k]  = Rx_row @ v[:3]

        for i, name in enumerate(link_names):
            F_d, v_l, al = per_link[name](q, v)
            F_drag_x[i, k] = float(F_d[0])
            v_link_x[i, k] = float(v_l[0])
            alpha_[i, k]   = float(al)

    # ── Drag breakdown ─────────────────────────────────────────────────────
    print("\n" + "─" * 88)
    print("Total drag force in world-x summed over links (N):")
    tot = F_drag_x.sum(axis=0)
    print(f"  mean = {tot.mean():+.4f}  min = {tot.min():+.4f}  max = {tot.max():+.4f}")
    print(f"  time-integrated impulse over T={T}s: {tot.mean()*T:+.4f} N·s")

    print("\nPer-link world-x drag (sorted by |mean|, N):")
    print(f"  {'link':30s} {'mean Fx':>10s} {'peak Fx':>10s} "
          f"{'⟨|v_x|⟩':>10s} {'⟨α⟩':>8s}")
    sorted_idx = np.argsort(-np.abs(F_drag_x.mean(axis=1)))
    for i in sorted_idx:
        name = link_names[i]
        mean_f = F_drag_x[i, :].mean()
        peak_f = F_drag_x[i, np.argmax(np.abs(F_drag_x[i,:]))]
        mean_v = np.abs(v_link_x[i, :]).mean()
        mean_a = alpha_[i, :].mean()
        print(f"  {name:30s} {mean_f:+10.4f} {peak_f:+10.4f} "
              f"{mean_v:10.4f} {mean_a:8.3f}")

    # ── Base-DOF forces ────────────────────────────────────────────────────
    print("\n" + "─" * 88)
    print("Generalized base forces (means, body frame): [surge, sway, heave, roll, pitch, yaw]")
    print(f"  drag      = {np.array2string(tau_drag[:6,:].mean(axis=1), precision=4)}")
    print(f"  buoyancy  = {np.array2string(tau_buoy[:6,:].mean(axis=1), precision=4)}")
    print(f"  gravity   = {np.array2string(grav[:6,:].mean(axis=1), precision=4)}")
    print(f"  buoy−grav = {np.array2string((tau_buoy[:6,:]-grav[:6,:]).mean(axis=1), precision=4)}")

    print(f"\nBase acceleration (body frame, mean): "
          f"{np.array2string(a_body.mean(axis=1), precision=4)}")
    print(f"World-x base accel: mean = {a_world_x.mean():+.4f} m/s², "
          f"std = {a_world_x.std():.4f} m/s², "
          f"implied Δvx = {a_world_x.mean()*T:+.4f} m/s over T")
    print(f"World-x base vel:   {vx_world[0]:+.4f} → {vx_world[-1]:+.4f} m/s "
          f"(periodic in OCP if Δ≈0)")

    # ── Sanity check: midpoint vs integrated drag on rotating leg ─────────
    print("\n" + "─" * 88)
    print("Midpoint-only vs integrated drag on a leg rotating at ω about its base:")
    print("  For transverse (perpendicular-to-axis) motion:")
    print("    integrated  |F|    = (1/3) · 0.5·ρ·Cd·D · ω²·L³")
    print("    midpoint    |F|    = (1/4) · 0.5·ρ·Cd·D · ω²·L³   →  75% of true")
    print("    integrated  |M|    = (1/4) · 0.5·ρ·Cd·D · ω²·L⁴")
    print("    midpoint    |M|    = (1/8) · 0.5·ρ·Cd·D · ω²·L⁴   →  50% of true")

    test = next((n for n in link_names if "Thigh" in n or "Calf" in n), None)
    if test:
        cyl = robot.links[test].cylinder
        L, D = cyl.length, 2 * cyl.radius
        omega, Cd, rho = 5.0, dyn.Cd_t, dyn.rho
        F_int = (1/3) * 0.5 * rho * Cd * D * omega**2 * L**3
        F_mid = (1/4) * 0.5 * rho * Cd * D * omega**2 * L**3
        M_int = (1/4) * 0.5 * rho * Cd * D * omega**2 * L**4
        M_mid = (1/8) * 0.5 * rho * Cd * D * omega**2 * L**4
        print(f"\n  Numerical example on {test} (L={L:.3f}m, D={D:.3f}m, "
              f"Cd_t={Cd}, ω={omega} rad/s):")
        print(f"    Force:  integrated={F_int:.3f} N   midpoint={F_mid:.3f} N "
              f"(model is {100*F_mid/F_int:.0f}% of true, underestimate {100-100*F_mid/F_int:.0f}%)")
        print(f"    Moment: integrated={M_int:.3f} Nm  midpoint={M_mid:.3f} Nm "
              f"(model is {100*M_mid/M_int:.0f}% of true, underestimate {100-100*M_mid/M_int:.0f}%)")


if __name__ == "__main__":
    main()
