#!/usr/bin/env python3
"""Sample and integrate solutions at the collocation points.

Radau collocation enforces the dynamics at the collocation points, not at the
interval starts, so accelerations evaluated at grid nodes are wrong. Cycle
integrals use the Radau weights, like the OCP objective; a plain mean over the
points is biased. Always load ``Xc`` via ``collocation_states``, since it is
stored in tangent coordinates.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "stage1_gait_optimization"))
from stage1_gait_optimization.ocp_common import (  # noqa: E402
    collocation_coefficients,
    tangent_to_legacy,
)

D_COLLOC = 3  # Radau degree used by the OCP

# Allowed mismatch between Xc and the grid nodes it coincides with (~6e-13 in practice)
XC_TOL = 1e-8


def collocation_states(robot, X, Xc, nq, N, d=D_COLLOC):
    """Quaternion states at the collocation points: ``(Xc_leg, phase, xc_err)``.

    The tangent reference is the node-0 orientation (the OCP anchors phi there).
    The result is checked against the grid nodes that coincide with the last
    Radau point of each interval.
    """
    tau_root, _, _, _ = collocation_coefficients(d)
    q_ref = X[3:7, 0].copy()
    # nq is already the reduced size stored in the solution file.
    Xc_leg = tangent_to_legacy(Xc, q_ref, robot.model, nq=nq, nv=robot.nv_reduced)

    tau1 = np.array([Xc_leg[:, k * d + d - 1] for k in range(N)]).T
    xc_err = float(np.abs(tau1 - X[:, 1:]).max())
    if xc_err > XC_TOL:
        raise SystemExit(
            f"\nXc reconstruction disagrees with the grid nodes by {xc_err:.3e} "
            f"— the collocation states are in the wrong frame.")

    # Phase of each column (interval k, root i)
    phase = np.array([(k + tau_root[i + 1]) / N
                      for k in range(N) for i in range(d)])
    return Xc_leg, phase, xc_err


def solution_states(robot, meta, name, d=D_COLLOC):
    """``collocation_states`` for a loaded solution; exits if it has no ``Xc``."""
    if "Xc" not in meta:
        raise SystemExit(
            f"{name} has no Xc block, so it could only be sampled at the grid "
            f"nodes — where Radau never enforced the dynamics.  Re-solve with a "
            f"version-2 solution file.")
    return collocation_states(robot, meta["X"], meta["Xc"], meta["nq"],
                              meta["N"], d)


def power_spans(phase: np.ndarray, vx: np.ndarray):
    """``[(start, width), ...]`` phase spans where ``vx < 0`` (power stroke).

    ``phase`` is strictly increasing in (0, 1] and treated as circular. Zero
    crossings are interpolated in phase, since Radau points are unevenly
    spaced. A span wrapping past 1 is returned once with its full width.
    """
    neg = vx < 0
    if neg.all():
        return [(float(phase[0]), 1.0)]
    if not neg.any():
        return []

    m = len(phase)
    # Circular successor and phase gap
    nxt = [(i + 1) % m for i in range(m)]
    gap = np.array([phase[nxt[i]] + (1.0 if nxt[i] == 0 else 0.0) - phase[i]
                    for i in range(m)])

    def cross(i):
        """Phase where vx crosses zero between sample i and the next."""
        a, b = vx[i], vx[nxt[i]]
        return phase[i] + gap[i] * a / (a - b)

    starts = [cross(i) for i in range(m) if not neg[i] and neg[nxt[i]]]
    ends = [cross(i) for i in range(m) if neg[i] and not neg[nxt[i]]]
    spans = []
    for a in sorted(starts):
        later = [e for e in ends if e > a]
        b = min(later) if later else min(ends) + 1.0
        spans.append((a, b - a))
    return spans


def joint_power(Xc, U, n_act, N, d=D_COLLOC):
    """Joint power ``tau_j * qdot_j`` at the collocation points, ``(n_act, N*d)``.

    Takes the raw tangent ``Xc`` (joint velocities need no conversion) and
    uses the same points as ``ocp_common.cycle_energy``.
    """
    nv = 6 + n_act
    vc = np.asarray(Xc)[nv + 6:, :]
    return np.stack([U[:, col // d] * vc[:, col] for col in range(N * d)], axis=1)


def cycle_mean(a: np.ndarray, N: int, d: int = D_COLLOC) -> float:
    """Cycle mean of a collocation-point trace with Radau weights (independent of T)."""
    _, _, _, B = collocation_coefficients(d)
    return float(sum(B[i] * a[k * d + i] for k in range(N) for i in range(d)) / N)


def cycle_integral(a: np.ndarray, T: float, N: int, d: int = D_COLLOC) -> float:
    """``int a dt`` over one cycle."""
    return cycle_mean(a, N, d) * float(T)


def quadrature_weights(N: int, T: float, d: int = D_COLLOC):
    """Quadrature weights ``(w, W_cum)`` for collocation traces.

    ``w`` holds one weight per sample [s], so ``(f * w).sum()`` over any subset
    gives that subset's share of ``int f dt``. ``W_cum[j, i]`` integrates the
    Lagrange basis of point i from tau=0 to point j (its last row equals B),
    which gives correct running integrals within an interval.
    """
    tau_root, _, _, B = collocation_coefficients(d)
    tau = tau_root[1:]
    W_cum = np.zeros((d, d))
    for i in range(d):
        others = np.delete(tau, i)
        basis = (np.polynomial.polynomial.polyfromroots(others)
                 / np.prod(tau[i] - others))
        W_cum[:, i] = np.polynomial.polynomial.polyval(
            tau, np.polynomial.polynomial.polyint(basis))
    return np.tile(B, N) * (float(T) / N), W_cum


def cumulative_integral(a: np.ndarray, T: float, N: int, d: int = D_COLLOC):
    """Running integral ``int_0^t a dt'`` at every collocation sample."""
    _, W_cum = quadrature_weights(N, T, d)
    f = np.asarray(a, dtype=float).reshape(N, d) * (float(T) / N)
    # Completed intervals plus the partial current one
    done = np.concatenate([[0.0], np.cumsum(f @ W_cum[-1])[:-1]])
    return (f @ W_cum.T + done[:, None]).ravel()
