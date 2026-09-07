#!/usr/bin/env python3
"""The whole-robot forward-force balance, term by term, at the collocation points.

The identity being decomposed (``hydro_model/dynamics.py:_build_eom``) is

    [M_rb + M_A] a  +  [C_rb + C_A] v  +  g_rb  =  tau + tau_buoy + tau_drag

with ``tau[:6] = 0`` on the floating base.  Rows 0:3 are the base linear rows;
rotated into the world frame they are the forward force balance.

Why the world frame.  Those rows are expressed in the *body* frame, and the base
pitches ~20 deg per stroke, so gravity and buoyancy each project ~1.5 N into
body-frame surge — an order of magnitude more than anything the legs do, and
entirely an artifact of the frame.  Rotated by R both fall to zero to machine
precision (1e-17), because a vertical force cannot push the robot forward.

Two traps this module exists to not fall into, both verified numerically:

  - ``f_tau_added(q,v,a)`` is *exactly* ``M_A(q) a + C_A(q,v) v`` (checked to
    4.7e-15).  It is the total added-mass force, so adding ``f_C_A_v`` to it
    double-counts the reactive term.  The two parts are separated here instead,
    because that split is the physics: ``M_A a`` is the inertia of the entrained
    water, ``C_A v`` is the Kirchhoff reactive force an oscillating limb feels.
  - ``f_C_rb(q,v)`` returns the Coriolis **matrix**, not the vector ``C v``
    (``dynamics.py`` builds it from ``cdata.C`` and uses ``C_rb @ v``).  Reading
    it as a vector silently takes ``C[0,0]`` and leaves a ~1.3 N hole in the
    balance that still looks plausible on a plot.

``plot_thrust_budget.py`` draws this and documents what the resulting numbers do
and do not mean; ``plot_mechanism_vs_speed.py`` reduces it to one point per
solve of a sweep.
"""
from __future__ import annotations

import numpy as np
import pinocchio as pin
from thesis_style import PALETTE

from collocation import D_COLLOC

# Residual of the equation of motion above which the decomposition is wrong
# rather than merely noisy.  Note this is an *identity* check, not evidence
# about the trajectory: ``a`` comes from forward dynamics, so the balance closes
# by construction wherever it is sampled.  It catches a mixed-up frame or sign,
# nothing more.  The check with physical content is MOMENTUM_TOL below.
EOM_TOL = 1e-8

# The rigid-body terms are a genuine momentum derivative, so over a periodic
# cycle the momentum itself must return.  That is tested on the momentum, by
# ``momentum_residual`` below, and not on the quadrature of its derivative.
#
# ``mean(M_rb a + C_rb v)`` was the earlier test, bounded at 5e-3 N.  It cannot
# work: the integrand is M_rb(q)a, not a polynomial, so the mean carries the
# scheme's truncation error rather than the physics.  Across a 14-point amph
# sweep at N=48 it ran 2.7e-4 to 7.9e-3 N on converged solves, against 2.1e-2 to
# 1.7e-1 N for the grid-node sampling it was meant to exclude — a separation of
# 2.6x, with the bound sitting inside the good population and rejecting three of
# the fourteen.  The endpoint difference is exact instead, and reads ~1e-14 kg
# m/s on those same solves; a solve whose periodicity closed only to the
# solver's own constraint tolerance would show ~1e-4.
MOMENTUM_TOL = 1e-6      # kg m/s

# Net external forward force below which ``plot_thrust_budget`` stays quiet.
# This is a force, and keeps the value the momentum bound used to carry so that
# the note it guards fires exactly as it did before.
EXT_FORCE_TOL = 5e-3     # N

# The terms of the balance, in the order they are drawn and tabulated.
#   key: (label, colour, is_external)
# External terms are forces the water and gravity apply; the rest are the
# inertial and Coriolis response they are balanced against.
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
    """World-frame forward (x) component of every term of the EoM, per sample.

    Returns ``(terms, residual)``.  The base linear rows (0:3) of each
    generalized force are a force 3-vector in the base's own frame, so rotating
    them by R gives world-frame forces and the identity survives term by term.
    """
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

        # Each of these is a generalized force; [:3] is its base linear part.
        vec = {
            "Mrb_a": np.array(dyn.f_M_rb(q)) @ a,
            "MA_a":  np.array(dyn.f_M_added(q)) @ a,
            "CAv":   np.array(dyn.f_C_A_v(q, v)).flatten(),
            # MATRIX, not a vector — see the module docstring.
            "Crb_v": np.array(dyn.f_C_rb(q, v)) @ v_flat,
            "grav":  np.array(dyn.f_g_rb(q)).flatten(),
            "buoy":  np.array(dyn.f_tau_buoyancy(q)).flatten(),
            "drag":  np.array(dyn.f_tau_drag(q, v)).flatten(),
        }
        for name, gf in vec.items():
            out[name][col] = (R @ gf[:3])[0]

        # M a + C v + g - buoy - drag == 0 on the unactuated base rows.
        resid[col] = (out["Mrb_a"][col] + out["MA_a"][col] + out["Crb_v"][col]
                      + out["CAv"][col] + out["grav"][col]
                      - out["buoy"][col] - out["drag"][col])

    # Gravity enters the EoM on the left; as an applied force it acts backwards.
    out["grav"] = -out["grav"]
    return out, resid


def forward_momentum(dyn, q, v) -> float:
    """World-frame forward component of the rigid-body linear momentum.

    Same rotation as ``force_terms``: the base linear rows are in the body
    frame, and only the world-frame component is a forward momentum.
    """
    p = np.array(dyn.f_M_rb(q)) @ np.asarray(v, dtype=float).flatten()
    R = pin.Quaternion(*np.roll(np.asarray(q)[3:7], 1)).matrix()
    return float((R @ p[:3])[0])


def momentum_residual(dyn, X, nq) -> float:
    """``p_x(T) - p_x(0)`` — the forward momentum the cycle fails to return.

    ``X`` is the grid-node block, whose first and last columns are the two ends
    of the cycle.  The OCP constrains the state to be periodic, so this vanishes
    on a converged solve and grows with whatever the periodicity constraints
    were left holding; see MOMENTUM_TOL for the sizes.
    """
    return (forward_momentum(dyn, X[:nq, -1], X[nq:, -1])
            - forward_momentum(dyn, X[:nq, 0], X[nq:, 0]))
