"""The amph NLP must keep exactly the same shape through the refactor.

Builds the transcription (without solving) and pins the variable and constraint
counts.  This is what catches a constraint being silently added, dropped, or
relocated — in particular the side-joint pinning at ``ocp_common.py:321-322``
moving from grid points to collocation points when it becomes a spec hook.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import REPO_ROOT
from hydro_model import QuadrupedRobot, SymbolicDynamics
from ocp_common import build_collocation_nlp

URDF_PATH = REPO_ROOT / "src" / "amph" / "urdf" / "amph.urdf"

# Captured on pre-refactor source at N=4, d_colloc=3.
BASELINE = {"nx": 661, "ng": 1293, "np": 0}
N = 4
NLP_KWARGS = dict(t_lo=1.0, t_hi=1.0, t_init=1.0, v_target=0.2,
                  f_c=20.0, heading_tol=0.1, d_colloc=3)


@pytest.fixture(scope="module")
def nlp():
    robot = QuadrupedRobot(URDF_PATH)
    robot.forward_kinematics(np.zeros(robot.nq))
    robot.build_cylinders()
    dyn = SymbolicDynamics(robot)

    X = np.zeros((robot.nq + robot.nv, N + 1))
    X[3:7, :] = np.array([0.0, 0.0, 0.0, 1.0])[:, None]   # valid quaternion
    U = np.zeros((robot.n_actuated, N))
    return robot, build_collocation_nlp(dyn, robot, X, U, N, **NLP_KWARGS)


def test_problem_size_unchanged(nlp):
    _, built = nlp
    opti = built["opti"]
    got = {"nx": opti.nx, "ng": opti.ng, "np": opti.np}
    assert got == BASELINE, (
        f"NLP shape changed: {got} != {BASELINE}. A constraint was added, "
        f"dropped, or moved between grid and collocation points."
    )


def test_handles_and_slices(nlp):
    robot, built = nlp
    n_act = robot.n_actuated
    assert built["X"].shape == (2 * robot.nv, N + 1)
    assert built["U"].shape == (n_act, N)
    assert built["Xc"].shape == (2 * robot.nv, N * NLP_KWARGS["d_colloc"])
    assert built["Q_J"] == slice(6, 6 + n_act)
    assert built["V_B"] == slice(6 + n_act, 12 + n_act)
    assert built["V_J"] == slice(12 + n_act, 2 * robot.nv)
    for key in ("power_cost", "vel_smooth_cost", "drift_cost", "T", "alpha", "q_ref_quat"):
        assert key in built
