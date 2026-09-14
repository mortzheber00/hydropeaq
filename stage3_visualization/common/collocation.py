#!/usr/bin/env python3
"""Sampling a solution where its dynamics hold, and integrating it there.

Every stage-3 figure that puts a number on a trajectory — a cycle mean, an
impulse, a limit margin — reads it off the solution's ``Xc`` block rather than
its grid nodes ``X``, and integrates it on the Radau weights rather than a
rectangle rule.  Both choices are forced, and the reasons live here rather than
in each figure that depends on them.

**Where to sample.**  Degree-3 Radau collocation enforces the dynamics at the
three roots inside each interval — the last of which is tau=1, the *next* grid
node — and never at tau=0, which is where every grid node sits.  Asking the
model for an acceleration at a node therefore gets one the trajectory does not
have: on the reference solution the rigid-body terms then carry -0.082 N of
cycle-mean force that is pure artifact, the same size as the real result, while
at the collocation points they close to -2e-4 N as momentum conservation
requires.  Terms built from (q, v) alone — quasi-steady drag, joint limits —
are honest at a node, but they still have to be *integrated* correctly.

**How to integrate.**  The weights B are the quadrature the OCP's own objective
and ``ocp_common.cycle_energy`` use, so a mean taken with them is the same kind
of number the sweep reports.  A plain ``.mean()`` over the columns weights the
three roots of an interval equally, which Radau does not; a rectangle sum over
the N+1 grid nodes additionally counts the periodic endpoint twice, which biased
the per-leg impulses in ``gait_diagnostics`` by about 7%.

``Xc`` is stored in tangent coordinates about a reference quaternion the file
does not record, so it is unusable until ``collocation_states`` recovers it.
That is why every consumer comes through here instead of reading ``meta["Xc"]``.
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

# Collocation degree the transcription fixes (Radau, ocp_common's d_colloc=3).
D_COLLOC = 3

# Reconstruction of Xc against the grid nodes it shares (Radau's tau=1 point of
# interval k is grid node k+1).  Agreement to ~6e-13 in practice.
XC_TOL = 1e-8


def collocation_states(robot, X, Xc, nq, N, d=D_COLLOC):
    """``(q, v)`` columns at the collocation points, their phases, and the
    reconstruction error the tolerance above was checked against.

    ``Xc`` is stored in tangent coordinates about the reference quaternion the
    transcription anchored on, and the file does not record it.  It is
    recoverable: the OCP pins ``X[3:6, 0] == 0`` (the phi anchor), so the
    solution's own node-0 orientation *is* that reference.

    The check is here rather than in each caller because getting it wrong is
    silent: a reconstruction on the wrong reference still produces a plausible
    trajectory, just not this one.  Radau's tau=1 point of interval k is grid
    node k+1, so the two representations overlap at N points and must agree —
    to ~6e-13 in practice.
    """
    tau_root, _, _, _ = collocation_coefficients(d)
    q_ref = X[3:7, 0].copy()
    # Reduced dimensions, not the tree's: for a closed-chain robot the state the
    # transcription solved on is the actuated one, which is what ``nq`` from the
    # solution file already is.  Same pairing as run_collocation's own save.
    Xc_leg = tangent_to_legacy(Xc, q_ref, robot.model, nq=nq, nv=robot.nv_reduced)

    tau1 = np.array([Xc_leg[:, k * d + d - 1] for k in range(N)]).T
    xc_err = float(np.abs(tau1 - X[:, 1:]).max())
    if xc_err > XC_TOL:
        raise SystemExit(
            f"\nXc reconstruction disagrees with the grid nodes by {xc_err:.3e} "
            f"— the collocation states are in the wrong frame.")

    # Phase of each column: interval k, root tau_root[i+1] inside it.
    phase = np.array([(k + tau_root[i + 1]) / N
                      for k in range(N) for i in range(d)])
    return Xc_leg, phase, xc_err


def solution_states(robot, meta, name, d=D_COLLOC):
    """``collocation_states`` for a loaded solution, or exit saying what is missing.

    ``Xc`` is optional in the solution format, so a file written before the
    export existed loads fine and would silently be measured at the grid nodes.
    ``name`` is what to call the file in that message.
    """
    if "Xc" not in meta:
        raise SystemExit(
            f"{name} has no Xc block, so it could only be sampled at the grid "
            f"nodes — where Radau never enforced the dynamics.  Re-solve with a "
            f"version-2 solution file.")
    return collocation_states(robot, meta["X"], meta["Xc"], meta["nq"],
                              meta["N"], d)


def power_spans(phase: np.ndarray, vx: np.ndarray):
    """``[(start, width), ...]`` of the cycle where ``vx < 0``, in phase.

    ``phase`` are the sample phases in (0, 1], strictly increasing, and the
    sequence is treated as circular: the neighbour of the last sample is the
    first one a cycle later.  Crossings are placed by linear interpolation
    between the two bracketing samples *in phase*, so unevenly spaced samples
    land where the stroke actually turns — which is why this lives here.  The
    Radau roots are not evenly spaced inside an interval, and an index-based
    interpolation would put every transition in the wrong place.  Given the grid
    nodes it reproduces ``plot_solution_legs.power_segments`` to ~1e-12, which
    ``plot_structure_vs_speed --check`` asserts.

    Spans may wrap past 1; the width is what the duty factor sums, so a wrapped
    span is returned once with its true width rather than split.
    """
    neg = vx < 0
    if neg.all():
        return [(float(phase[0]), 1.0)]
    if not neg.any():
        return []

    m = len(phase)
    # Circular differences: sample i is followed by i+1, and the last by the
    # first one cycle on.
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
    """Per-joint mechanical power ``tau_j * qdot_j`` at the collocation points.

    ``Xc`` is the *tangent* state, whose joint-velocity block sits at ``[nv+6:]``
    — the same rows ``ocp_common.cycle_energy`` integrates, so a power needs no
    conversion out of tangent coordinates and this one takes the raw block rather
    than a reconstruction.  ``U`` is piecewise constant, so every collocation
    point of interval k takes ``U[:, k]``.

    Sampling here rather than at the grid nodes puts the split into delivered and
    absorbed work on the same points, and under the same weights, as the energy
    total ``cycle_energy`` returns.  Node sampling is a left-rectangle rule and
    reads 30-40% low — the error that made the old grid-node energy figures wrong.
    """
    nv = 6 + n_act
    vc = np.asarray(Xc)[nv + 6:, :]
    return np.stack([U[:, col // d] * vc[:, col] for col in range(N * d)], axis=1)


def cycle_mean(a: np.ndarray, N: int, d: int = D_COLLOC) -> float:
    """Cycle mean of a collocation-point trace, on the Radau weights.

    ``(1/T) * sum_k sum_i B_i * dt * f == (1/N) * sum_k sum_i B_i * f``, so the
    period cancels and a mean needs no T.
    """
    _, _, _, B = collocation_coefficients(d)
    return float(sum(B[i] * a[k * d + i] for k in range(N) for i in range(d)) / N)


def cycle_integral(a: np.ndarray, T: float, N: int, d: int = D_COLLOC) -> float:
    """``int a dt`` over one cycle — ``cycle_mean`` carrying its period."""
    return cycle_mean(a, N, d) * float(T)


def quadrature_weights(N: int, T: float, d: int = D_COLLOC):
    """``(w, W_cum)`` — the Radau weights a collocation trace integrates on.

    ``w`` is one weight per sample, in seconds, so ``(f * w).sum()`` over any
    subset is that subset's share of ``int f dt`` and over all of them is
    ``cycle_integral(f, T, N)``.  Subsets are the point: the collocation points
    are not equally spaced, so an unweighted mean over half a cycle silently
    reweights it.

    ``W_cum[j, i]`` integrates one interval's samples from tau=0 to the j-th
    Radau point — the same interpolatory Lagrange basis the weights B come from,
    so its last row *is* B.  It exists because a running sum of ``B_i * f_i``
    charges every sample its whole-interval weight the moment the curve passes
    it, which lands on the right total through a visibly wrong path.
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
    """``int_0^t a dt'`` at every collocation sample, on the Radau basis."""
    _, W_cum = quadrature_weights(N, T, d)
    f = np.asarray(a, dtype=float).reshape(N, d) * (float(T) / N)
    # Whole intervals already behind this one, plus the partial one it is in.
    done = np.concatenate([[0.0], np.cumsum(f @ W_cum[-1])[:-1]])
    return (f @ W_cum.T + done[:, None]).ravel()
