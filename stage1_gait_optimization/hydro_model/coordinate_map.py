"""Mapping between reduced actuated coordinates and the Pinocchio tree.

A URDF is always a tree.  A robot with closed kinematic loops therefore has
more tree joints than it has degrees of freedom, and the cut pins show up
nowhere in the model.  A ``CoordinateMap`` is the one place that discrepancy
lives: it takes the independent actuated coordinates ``theta`` and produces the
full tree configuration, velocity and acceleration, plus the projection of tree
generalised forces back onto ``theta``.

For a serial robot the map is the identity and costs nothing (``IdentityMap``
returns its arguments unchanged, so the CasADi graph is untouched).  For a
closed-chain robot it carries the loop-closure solution.

Shapes, with ``n_theta`` reduced DOF and tree blocks ``nq_j = model.nq - 7``,
``nv_j = model.nv - 6``:

    q_joints(theta)                 -> (nq_j,)     tree joint configuration
    v_joints(theta, thd)            -> (nv_j,)     S @ thd
    a_joints(theta, thd, thdd)      -> (nv_j,)     S @ thdd + Sdot @ thd
    tau_joints(theta, tau_tree_j)   -> (n_theta,)  S.T @ tau
    S(theta)                        -> (nv_j, n_theta)
    feasibility(theta)              -> (m,)        required >= 0, m may be 0
    expand_numeric(theta)           -> ndarray (nq_j,)   numpy twin for plotting

The symbolic methods accept and return CasADi SX/MX; ``expand_numeric`` is
numpy.  ``nq_j != nv_j`` whenever the tree contains unbounded (continuous)
joints, whose configuration is a ``(cos, sin)`` pair rather than an angle.
"""

from __future__ import annotations

import casadi as ca
import numpy as np


class CoordinateMap:
    """Interface; see module docstring.  Subclasses must set the three sizes."""

    n_theta: int
    nq_j: int
    nv_j: int

    def q_joints(self, theta):
        raise NotImplementedError

    def v_joints(self, theta, thd):
        raise NotImplementedError

    def a_joints(self, theta, thd, thdd):
        raise NotImplementedError

    def tau_joints(self, theta, tau_tree_j):
        raise NotImplementedError

    def S(self, theta):
        raise NotImplementedError

    def feasibility(self, theta):
        """Expressions the OCP must keep >= 0 (e.g. loop assemblability)."""
        return ca.SX.zeros(0, 1)

    def expand_numeric(self, theta: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def v_numeric(self, theta: np.ndarray, thd: np.ndarray) -> np.ndarray:
        """Numpy twin of ``v_joints`` — used when replaying saved trajectories."""
        raise NotImplementedError


class IdentityMap(CoordinateMap):
    """Serial robot: theta *is* the tree joint vector.

    Every method returns its argument object rather than multiplying by an
    identity, so a robot using this map produces exactly the expression graph
    it did before coordinate maps existed.
    """

    def __init__(self, n_theta: int):
        self.n_theta = n_theta
        self.nq_j = n_theta
        self.nv_j = n_theta

    def q_joints(self, theta):
        return theta

    def v_joints(self, theta, thd):
        return thd

    def a_joints(self, theta, thd, thdd):
        return thdd

    def tau_joints(self, theta, tau_tree_j):
        return tau_tree_j

    def S(self, theta):
        return ca.SX.eye(self.n_theta)

    def expand_numeric(self, theta: np.ndarray) -> np.ndarray:
        return np.asarray(theta, dtype=float)

    def v_numeric(self, theta: np.ndarray, thd: np.ndarray) -> np.ndarray:
        return np.asarray(thd, dtype=float)
