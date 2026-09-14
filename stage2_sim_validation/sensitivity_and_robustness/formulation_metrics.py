#!/usr/bin/env python3
"""How large are the two approximations the OCP formulation makes?

Recomputes, on the stored thesis nominal, the numbers chapter 8.3 (limitation L4,
points ii and iii) quotes.  Both approximations exist to keep the NLP small, and
both are only defensible if what they drop is small next to what they keep.

**[1] Frozen submersion ratio in the added-mass force (L4 ii).**
``SymbolicHydrodynamicModel.added_mass_force`` writes the added-mass force per
link as Kirchhoff's equations with a constant body-frame added mass scaled by the
submersion ratio alpha.  That is exact for a fully submerged link, but the true
joint-space added mass is ``M_A(q) = sum_i alpha_i(q) A_i(q)``, and differentiating
it (the Christoffel/Coriolis term) also produces a term in ``d alpha / dq`` that
the code neglects:

    h_alpha = sum_i (g_i . v) A_i v - 1/2 g_i (v^T A_i v),   g_i = d alpha_i / dq

Section [1a] first checks that this expression really is the whole difference:
the full Christoffel Coriolis of ``M_A(q)``, minus the code's ``C_A(v) v``, should
equal ``h_alpha``.  It does to machine precision in the joint rows.  The floating
base rows do not close, because the Christoffel formula treats ``v`` as the time
derivative of the coordinates, which the base angular velocity is not; the
residual there is a property of the check, not of ``h_alpha``.  Section [1b] then
sizes ``h_alpha`` against the retained added-mass force, the drag, and the
applied joint torques.

**[2] Tangent-rate approximation (L4 iii).**  The collocation states carry the
base orientation as a tangent vector phi about a reference quaternion, and the
dynamics use ``phi_dot = omega_b``.  The exact kinematics are
``phi_dot = Jr^-1(phi) omega_b``; the deviation grows with |phi|, so it is
reported together with the largest |phi| the gait reaches.

Conventions.  ``X`` is the legacy node layout ``[q(nq); v(nv)]`` with a
quaternion base, 49 nodes.  ``Xc`` is ``[phi-tangent q(nv); v(nv)]`` at the
collocation points, so the base rotation is rows 3:6 and the base angular
velocity rows nv+3:nv+6.  Section [1] is evaluated on the nodes, with the
accelerations for the retained ``tau_A`` taken by central differences; section
[2] on the collocation points, where the approximation is actually imposed.

Usage:
  python formulation_metrics.py                     # the thesis nominal
  python formulation_metrics.py --solution <npz>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import casadi as ca
import numpy as np
import pinocchio as pin
import pinocchio.casadi as cpin

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "stage1_gait_optimization"))

from hydro_model import load_robot                                     # noqa: E402
from hydro_model.hydrodynamics import SymbolicHydrodynamicModel, _skew  # noqa: E402

# The thesis nominal: the v = 0.18 point of the cold co-design front.
DEFAULT_SOLUTION = (REPO_ROOT / "experiment_results" / "mlruns" / "2" /
                    "69943d2cffc94ca5b78271a7df24c546" / "artifacts" /
                    "TLPG50_v0p180_T1p400.npz")


def build_functions(robot):
    """CasADi functions for the neglected term and everything it is compared to.

    ``h_alpha`` and the full Christoffel Coriolis are differentiated with respect
    to a tangent perturbation ``dq`` of ``q0`` and evaluated at ``dq = 0``, which
    is how a derivative on the free-flyer configuration space is taken.
    """
    cmodel = cpin.Model(robot.model)
    cdata = cmodel.createData()
    nq, nv = robot.model.nq, robot.model.nv

    q0 = ca.SX.sym("q0", nq)
    dq = ca.SX.sym("dq", nv)
    v = ca.SX.sym("v", nv)
    hyd = SymbolicHydrodynamicModel(robot, cmodel, cdata, q0, v, nv)

    q = cpin.integrate(cmodel, q0, dq)
    cpin.forwardKinematics(cmodel, cdata, q, v)
    cpin.updateFramePlacements(cmodel, cdata)

    MA = ca.SX.zeros(nv, nv)
    h_alpha = ca.SX.zeros(nv, 1)
    for link in robot.links.values():
        cyl = link.cylinder
        if cyl is None:
            continue
        oMf = cdata.oMf[link.frame_id]
        R = oMf.rotation
        J_full = cpin.computeFrameJacobian(cmodel, cdata, q, link.frame_id,
                                           pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        Jv = J_full[:3, :] - _skew(R @ ca.SX(cyl.center_local)) @ J_full[3:, :]
        Jw = J_full[3:, :]
        axis = R @ ca.SX(cyl.axis_local)
        axis = axis / (ca.norm_2(axis) + 1e-15)
        alpha = hyd.submersion_ratio(cyl, axis, oMf, R)
        A_i = hyd.added_mass_matrix(cyl, ca.SX(1.0), axis, Jv, Jw)
        g_i = ca.jacobian(alpha, dq).T
        Av = A_i @ v
        MA += alpha * A_i
        h_alpha += ca.dot(g_i, v) * Av - 0.5 * g_i * ca.dot(v, Av)

    # Full Christoffel Coriolis of M_A(q):  (dM/dt) v - 1/2 d/dq (v^T M v)
    h_full = (ca.jacobian(MA @ v, dq) @ v
              - 0.5 * ca.jacobian(ca.dot(v, MA @ v), dq).T)

    return dict(
        h=ca.Function("h", [q0, dq, v], [h_alpha, h_full]),
        CAv=ca.Function("CAv", [q0, v], [hyd.f_tau_added(q0, v, ca.SX.zeros(nv))]),
        tau_A=hyd.f_tau_added,
        drag=hyd.f_tau_drag,
    )


def rms_norm(M):
    """Peak and rms of the per-node vector norm, rows = nodes."""
    n = np.linalg.norm(M, axis=1)
    return n.max(), np.sqrt((n ** 2).mean())


def rms(M):
    return np.sqrt((M ** 2).mean())


def frozen_alpha(robot, d):
    nq, nv = robot.model.nq, robot.model.nv
    f = build_functions(robot)
    X, U, T, N = d["X"], d["U"], float(d["T"]), int(d["N"])
    Q, V = X[:nq, :], X[nq:, :]
    dt = T / N
    z = np.zeros(nv)

    H, HF, C, A, D = [], [], [], [], []
    for k in range(Q.shape[1]):
        qk, vk = Q[:, k], V[:, k]
        lo, hi = max(k - 1, 0), min(k + 1, N)
        ak = (V[:, hi] - V[:, lo]) / ((hi - lo) * dt)
        h, hf = f["h"](qk, z, vk)
        H.append(np.array(h).ravel())
        HF.append(np.array(hf).ravel())
        C.append(np.array(f["CAv"](qk, vk)).ravel())
        A.append(np.array(f["tau_A"](qk, vk, ak)).ravel())
        D.append(np.array(f["drag"](qk, vk)).ravel())
    H, HF, C, A, D = map(np.array, (H, HF, C, A, D))

    res = HF - C - H
    print(f"\n=== [1a] cross-check: Christoffel(M_A) - C_A v == h_alpha, "
          f"{Q.shape[1]} nodes ===")
    print("  residual, joint rows : max %.3e" % np.linalg.norm(res[:, 6:], axis=1).max())
    print("  residual, base rows  : max %.3e   (not closed by construction, see docstring)"
          % np.linalg.norm(res[:, :6], axis=1).max())
    print("  ||h_alpha|| rms %.4f" % rms_norm(H)[1])

    print(f"\n=== [1b] size of the neglected h_alpha, full nv-vector norms ===")
    for name, M in (("neglected h_alpha", H), ("retained C_A v", C),
                    ("retained tau_A", A), ("drag tau", D)):
        print("  %-18s max %7.3f  rms %7.3f" % (name, *rms_norm(M)))
    print("  ratio h/C_A v  : rms %.1f %%" % (100 * rms_norm(H)[1] / rms_norm(C)[1]))
    print("  ratio h/tau_A  : rms %.1f %%" % (100 * rms_norm(H)[1] / rms_norm(A)[1]))
    print("  ratio h/drag   : rms %.1f %%" % (100 * rms_norm(H)[1] / rms_norm(D)[1]))

    print("\n  actuated joint rows only [N m]")
    Ha, Ca, Aa, Da = H[:, 6:], C[:, 6:], A[:, 6:], D[:, 6:]
    print("  neglected  : peak |component| %.4f   rms-per-joint %.4f"
          % (np.abs(Ha).max(), rms(Ha)))
    print("  C_A v      : peak %.4f  rms %.4f" % (np.abs(Ca).max(), rms(Ca)))
    print("  tau_A      : peak %.4f  rms %.4f" % (np.abs(Aa).max(), rms(Aa)))
    print("  drag       : peak %.4f  rms %.4f" % (np.abs(Da).max(), rms(Da)))
    print("  applied U  : peak %.4f  rms %.4f" % (np.abs(U).max(), rms(U)))
    print("  neglected / applied-torque rms = %.1f %%" % (100 * rms(Ha) / rms(U)))
    print("  neglected peak / peak applied  = %.1f %%"
          % (100 * np.abs(Ha).max() / np.abs(U).max()))


def tangent_rate(robot, d):
    nv = robot.model.nv
    Xc = d["Xc"]
    phi, om = Xc[3:6, :], Xc[nv + 3:nv + 6, :]

    dev, omn = [], []
    for k in range(phi.shape[1]):
        p, w = phi[:, k], om[:, k]
        th = np.linalg.norm(p)
        S = np.array([[0, -p[2], p[1]], [p[2], 0, -p[0]], [-p[1], p[0], 0]])
        coef = 1 / th ** 2 - (1 + np.cos(th)) / (2 * th * np.sin(th)) if th > 1e-9 else 0.0
        Jr_inv = np.eye(3) + 0.5 * S + coef * (S @ S)
        dev.append(np.linalg.norm(Jr_inv @ w - w))
        omn.append(np.linalg.norm(w))
    dev, omn = np.array(dev), np.array(omn)
    th_max = np.linalg.norm(phi, axis=0).max()

    print(f"\n=== [2] tangent-rate approximation phi_dot = omega_b, "
          f"{phi.shape[1]} collocation points ===")
    print("  |phi| max %.4f rad (%.1f deg)" % (th_max, np.degrees(th_max)))
    print("  deviation |Jr^-1(phi) w - w| : max %.4f rad/s, rms %.4f rad/s"
          % (dev.max(), np.sqrt((dev ** 2).mean())))
    print("  relative : peak %.1f %%, rms-norm %.1f %%, mean %.1f %%"
          % (100 * (dev / omn).max(), 100 * np.sqrt((dev ** 2).mean()) / np.sqrt((omn ** 2).mean()),
             100 * (dev / omn).mean()))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--solution", type=Path, default=DEFAULT_SOLUTION)
    ap.add_argument("--robot", default="amph", help="registered robot name")
    args = ap.parse_args()

    robot = load_robot(args.robot)
    d = np.load(args.solution, allow_pickle=True)
    print(f"solution: {args.solution.name}")
    frozen_alpha(robot, d)
    tangent_rate(robot, d)


if __name__ == "__main__":
    main()
