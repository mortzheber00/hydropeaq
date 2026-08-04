"""BODY2's initial guess, and (marked slow) the OCP solve it warm-starts."""

from __future__ import annotations

import numpy as np
import pytest
from hydro_model import SymbolicDynamics, load_robot
from hydro_model.robots.body2 import H_MIN
from hydro_model.trajectory import coords_of
from initial_guess import build_theta_sinusoid_guess

N, T = 12, 1.0


@pytest.fixture(scope="module")
def dyn():
    return SymbolicDynamics(load_robot("body2"))


def test_guess_is_assemblable_and_periodic(dyn):
    robot = dyn.robot
    X, U = build_theta_sinusoid_guess(dyn, N, T, 0.3, n_cycles=3)
    nq_r = robot.nq_reduced

    assert X.shape == (nq_r + robot.nv_reduced, N + 1)
    assert U.shape == (robot.n_actuated, N)
    assert np.all(np.isfinite(X)) and np.all(np.isfinite(U))

    theta = X[7:nq_r, :]
    margin = np.array([np.min(np.asarray(robot.coord_map.feasibility(theta[:, k])))
                       for k in range(N + 1)])
    assert margin.min() > H_MIN ** 2, "guess leaves the assemblable set"

    np.testing.assert_allclose(theta[:, -1], theta[:, 0], atol=1e-12)
    np.testing.assert_allclose(X[nq_r + 6:, -1], X[nq_r + 6:, 0], atol=1e-12)


def test_guess_rejects_an_infeasible_stroke(dyn):
    """A too-wide stroke must fail loudly rather than emit NaNs."""
    with pytest.raises(ValueError, match="assemblable"):
        build_theta_sinusoid_guess(dyn, N, T, 0.3,
                                   a_across=np.radians(45.0), n_cycles=1)


def test_coords_convention(dyn):
    assert coords_of(dyn.robot) == "reduced"
    assert coords_of(load_robot("amph")) == "tree"


@pytest.mark.slow
def test_body2_ocp_solves(dyn):
    """End-to-end: the reduced-coordinate OCP converges and stays assemblable."""
    from ocp_common import build_collocation_nlp, tangent_to_legacy

    robot = dyn.robot
    X_g, U_g = build_theta_sinusoid_guess(dyn, 16, T, 0.3, n_cycles=6)
    nlp = build_collocation_nlp(dyn, robot, X_g, U_g, 16, t_lo=T, t_hi=T, t_init=T,
                                v_target=0.0, f_c=20.0, heading_tol=0.05, d_colloc=3)
    opti, X, Tv = nlp["opti"], nlp["X"], nlp["T"]
    opti.minimize(2.0 * nlp["power_cost"] - 0.5 * (X[0, -1] - X[0, 0]) / Tv
                  + 20.0 * nlp["vel_smooth_cost"] + 10.0 * nlp["drift_cost"])
    opti.solver("ipopt", {"expand": False},
                {"max_iter": 400, "tol": 1e-4, "acceptable_tol": 1e-3,
                 "acceptable_iter": 15, "print_level": 0,
                 "mu_strategy": "adaptive", "nlp_scaling_method": "gradient-based"})
    sol = opti.solve()
    assert opti.stats()["return_status"] == "Solve_Succeeded"

    Xl = tangent_to_legacy(sol.value(X), nlp["q_ref_quat"], robot.model,
                           nq=robot.nq_reduced, nv=robot.nv_reduced)
    theta = Xl[7:robot.nq_reduced, :]
    margin = np.array([np.min(np.asarray(robot.coord_map.feasibility(theta[:, k])))
                       for k in range(17)])
    assert margin.min() > 0, "solution is not assemblable"
    # The h^2 >= H_MIN^2 inequality comes out active — the optimiser rides the
    # assembly boundary — so allow IPOPT's constraint tolerance on it.
    assert margin.min() >= H_MIN ** 2 * (1 - 1e-2), (
        f"assembly margin {np.sqrt(margin.min())*1e3:.4f} mm is below the "
        f"{H_MIN*1e3:.1f} mm floor by more than solver tolerance"
    )
    assert Xl[0, -1] - Xl[0, 0] > 0, "solution does not move forward"
