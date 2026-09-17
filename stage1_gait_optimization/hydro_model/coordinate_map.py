"""Mapping between reduced actuated coordinates and the Pinocchio tree.

A closed-chain robot has more tree joints than degrees of freedom. A
``CoordinateMap`` maps the actuated coordinates ``theta`` to the full tree
configuration, velocity and acceleration, and projects tree forces back onto
``theta``. Serial robots use ``IdentityMap``.

Shapes, with ``nq_j = model.nq - 7`` and ``nv_j = model.nv - 6``:

    q_joints(theta)                 -> (nq_j,)     tree joint configuration
    v_joints(theta, thd)            -> (nv_j,)     S @ thd
    a_joints(theta, thd, thdd)      -> (nv_j,)     S @ thdd + Sdot @ thd
    tau_joints(theta, tau_tree_j)   -> (n_theta,)  S.T @ tau
    S(theta)                        -> (nv_j, n_theta)
    feasibility(theta)              -> (m,)        required >= 0, m may be 0
    expand_numeric(theta)           -> ndarray (nq_j,)   numpy twin for plotting

Symbolic methods work on CasADi SX/MX. ``nq_j != nv_j`` if the tree has
continuous joints, which are stored as ``(cos, sin)`` pairs.
"""

from __future__ import annotations

import casadi as ca
import numpy as np


class CoordinateMap:
    """Base interface; subclasses set ``n_theta``, ``nq_j`` and ``nv_j``."""

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
        """Numeric version of ``v_joints``."""
        raise NotImplementedError


class IdentityMap(CoordinateMap):
    """Serial robot: ``theta`` is the tree joint vector.

    Arguments are returned as-is so the CasADi graph stays unchanged.
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
