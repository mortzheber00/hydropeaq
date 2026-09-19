#!/usr/bin/env python3
"""Term-by-term forward force balance of the robot at the collocation points.

Decomposes the base linear rows of

    [M_rb + M_A] a  +  [C_rb + C_A] v  +  g_rb  =  tau + tau_buoy + tau_drag

rotated into the world frame (in the pitching body frame, gravity and buoyancy
would show up as spurious surge forces).

Pitfalls:
  - ``f_tau_added`` is already ``M_A a + C_A v``; do not add ``f_C_A_v`` to it.
  - ``f_C_rb`` returns the Coriolis matrix, not ``C v``.

Used by plot_thrust_budget.py and plot_mechanism_vs_speed.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pinocchio as pin

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from stage3_visualization.common.thesis_style import PALETTE  # noqa: E402
from stage3_visualization.common.collocation import D_COLLOC  # noqa: E402

# EoM residual tolerance. This only catches frame or sign errors: the balance
# holds by construction, since ``a`` comes from forward dynamics.
EOM_TOL = 1e-8

# Allowed change of forward momentum over the (periodic) cycle, see
# momentum_residual. Converged solves give ~1e-14.
MOMENTUM_TOL = 1e-6      # kg m/s

# Net external forward force above which plot_thrust_budget prints a note
EXT_FORCE_TOL = 5e-3     # N

# key: (label, colour, is_external), in plotting order. External = water and
# gravity forces; the rest is the inertial response.
TERMS = {
    "drag":  (r"drag $F_{\mathrm{drag}}$",            PALETTE[2], True),
    "buoy":  (r"buoyancy $F_{\mathrm{buoy}}$",        PALETTE[1], True),
    "grav":  (r"gravity $-g_{\mathrm{rb}}$",          PALETTE[3], True),
    "CAv":   (r"added-mass reactive $C_A v$",         PALETTE[0], False),
    "MA_a":  (r"added-mass inertial $M_A a$",         PALETTE[4], False),
    "Crb_v": (r"rigid-body Coriolis $C_{\mathrm{rb}} v$", "0.45", False),
    "Mrb_a": (r"rigid-body inertial $M_{\mathrm{rb}} a$", "0.15", False),
}


def force_terms(robot, dyn, Xc_leg, U, N, nq, d=D_COLLOC):
    """World-x component of each EoM term per collocation point: ``(terms, residual)``."""
    n_col = N * d
    out = {k: np.zeros(n_col) for k in TERMS}
    resid = np.zeros(n_col)

    for col in range(n_col):
        k = col // d                     # U is piecewise constant per interval
        q, v = Xc_leg[:nq, col], Xc_leg[nq:, col]
        R = pin.Quaternion(*np.roll(q[3:7], 1)).matrix()   # pin wants (w,x,y,z)

        tau_full = np.concatenate([np.zeros(6), U[:, k]])
        a = np.asarray(dyn.eval_reduced_forward_dynamics(
            q[:7], q[7:nq], v, tau_full)).flatten()
        v_flat = np.asarray(v, dtype=float).flatten()

        # Generalized forces; [:3] is the base linear part (body frame)
        vec = {
            "Mrb_a": np.array(dyn.f_M_rb(q)) @ a,
            "MA_a":  np.array(dyn.f_M_added(q)) @ a,
            "CAv":   np.array(dyn.f_C_A_v(q, v)).flatten(),
            "Crb_v": np.array(dyn.f_C_rb(q, v)) @ v_flat,
            "grav":  np.array(dyn.f_g_rb(q)).flatten(),
            "buoy":  np.array(dyn.f_tau_buoyancy(q)).flatten(),
            "drag":  np.array(dyn.f_tau_drag(q, v)).flatten(),
        }
        for name, gf in vec.items():
            out[name][col] = (R @ gf[:3])[0]

        # M a + C v + g - buoy - drag == 0 on the unactuated base rows
        resid[col] = (out["Mrb_a"][col] + out["MA_a"][col] + out["Crb_v"][col]
                      + out["CAv"][col] + out["grav"][col]
                      - out["buoy"][col] - out["drag"][col])

    # Gravity is on the left-hand side; flip it to an applied force.
    out["grav"] = -out["grav"]
    return out, resid


def forward_momentum(dyn, q, v) -> float:
    """World-x component of the rigid-body linear momentum."""
    p = np.array(dyn.f_M_rb(q)) @ np.asarray(v, dtype=float).flatten()
    R = pin.Quaternion(*np.roll(np.asarray(q)[3:7], 1)).matrix()
    return float((R @ p[:3])[0])


def momentum_residual(dyn, X, nq) -> float:
    """``p_x(T) - p_x(0)`` from the grid nodes; ~0 for a converged periodic solve."""
    return (forward_momentum(dyn, X[:nq, -1], X[nq:, -1])
            - forward_momentum(dyn, X[:nq, 0], X[nq:, 0]))
