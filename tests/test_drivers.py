"""Smoke-test the driver scripts against every registered robot.

These are the entry points a person actually runs, and nothing else covers
them: importing a module catches syntax and import errors but not a name
resolved only at call time, nor a hardcoded leg name, nor a numerically
degenerate configuration.  Three separate bugs reached the user this way.

Only drivers that need neither MLflow nor a display are exercised here.
"""

from __future__ import annotations

import contextlib
import io
import warnings

import numpy as np
import pytest
from hydro_model import registry

ROBOTS = sorted(registry())


@pytest.mark.parametrize("name", ROBOTS)
def test_run_dynamics_validation(name):
    """Runs main() end to end, and treats numerical warnings as failures.

    The warning filter is the point: rebuilding cylinders from an invalid
    configuration (``np.zeros(nq)`` on a robot whose joint entries are
    ``(cos, sin)`` pairs) shows up only as ``invalid value encountered in
    divide`` and otherwise sails through with NaN geometry.
    """
    import run_dynamics_validation as drv

    original = drv.ROBOT
    try:
        drv.ROBOT = name
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            with contextlib.redirect_stdout(io.StringIO()):
                drv.main()
    finally:
        drv.ROBOT = original


@pytest.mark.parametrize("name", ROBOTS)
def test_symbolic_and_numeric_foot_positions_agree(name):
    """``dyn.f_foot_pos`` must match ``robot.foot_positions()``.

    They are computed by different code paths, and the foot is a fixed offset
    in a frame rather than the frame origin for a robot with no dedicated foot
    link — so the two drifted apart by 66 mm on BODY2 once one side learned
    about the offset and the other did not.
    """
    from hydro_model import SymbolicDynamics, load_robot

    robot = load_robot(name)
    dyn = SymbolicDynamics(robot)
    spec = robot.spec
    home = (np.zeros(robot.n_actuated) if spec.theta_home is None
            else np.asarray(spec.theta_home, dtype=float))

    rng = np.random.default_rng(3)
    for _ in range(5):
        theta = home + rng.uniform(-0.05, 0.05, size=robot.n_actuated)
        q = robot.neutral_config()
        q[0:3] = rng.normal(scale=0.05, size=3)
        q[7:] = robot.coord_map.expand_numeric(theta)
        robot.forward_kinematics(q)
        numeric = robot.foot_positions()
        for leg in spec.leg_names:
            symbolic = np.asarray(dyn.f_foot_pos[leg](q)).ravel()
            np.testing.assert_allclose(symbolic, numeric[leg], atol=1e-12,
                                       err_msg=f"{name} foot {leg}")
