"""amph must survive the multi-robot refactor numerically unchanged.

Every quantity here was captured by ``tests/data/make_golden.py`` on the
pre-refactor source.  If one of these moves, the refactor is wrong — it is not
a reason to regenerate the golden file.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import GOLDEN, REPO_ROOT
from hydro_model import QuadrupedRobot, SymbolicDynamics
from ocp_common import legacy_to_tangent, tangent_to_legacy

URDF_PATH = REPO_ROOT / "src" / "amph" / "urdf" / "amph.urdf"
CYL_FIELDS = ("radius", "length", "volume_displaced",
              "center", "axis_world", "axis_local", "center_local")

# Pure geometry is a deterministic computation: demand exact equality.
# CasADi evaluations may legitimately reassociate, so allow 1e-14.
EXACT = dict(rtol=0.0, atol=0.0)
TIGHT = dict(rtol=1e-14, atol=1e-300)


@pytest.fixture(scope="module")
def golden():
    if not GOLDEN.exists():
        pytest.skip(f"{GOLDEN} missing — run `python tests/data/make_golden.py`")
    return np.load(GOLDEN, allow_pickle=False)


@pytest.fixture(scope="module")
def built():
    robot = QuadrupedRobot(URDF_PATH)
    robot.forward_kinematics(np.zeros(robot.nq))
    robot.build_cylinders()
    return robot, SymbolicDynamics(robot)


def test_dimensions_and_names(built, golden):
    robot, _ = built
    nq, nv, n_act, njoints, nframes = golden["dims"]
    assert (robot.nq, robot.nv, robot.n_actuated) == (nq, nv, n_act)
    assert (robot.model.njoints, robot.model.nframes) == (njoints, nframes)
    assert list(robot.actuated_joint_names) == list(golden["actuated_joint_names"])
    assert sorted(robot.links) == list(golden["link_names"])


def test_model_geometry_unchanged(built, golden):
    """Catches _recenter_base_y regressions and any joint-placement drift."""
    robot, _ = built
    got = np.array([np.asarray(p.translation) for p in robot.model.jointPlacements])
    np.testing.assert_allclose(got, golden["joint_placements"], **EXACT)
    got_R = np.array([np.asarray(p.rotation) for p in robot.model.jointPlacements])
    np.testing.assert_allclose(got_R, golden["joint_rotations"], **EXACT)
    np.testing.assert_allclose(robot.model.inertias[1].lever, golden["base_lever"], **EXACT)
    np.testing.assert_allclose(robot.total_mass(), golden["total_mass"], **EXACT)


def test_foot_frames_resolve(built, golden):
    robot, _ = built
    ids = np.array([robot.foot_frame_ids[leg] for leg in sorted(robot.foot_frame_ids)])
    np.testing.assert_array_equal(ids, golden["foot_frame_ids"])
    # a frame id equal to nframes is pinocchio's silent "not found"
    assert ids.max() < robot.model.nframes


def test_mesh_volumes(built, golden):
    robot, _ = built
    got = np.array([robot.link_mesh_volumes[k] for k in sorted(robot.link_mesh_volumes)])
    np.testing.assert_allclose(got, golden["link_mesh_volumes"], **EXACT)


def test_cylinders_unchanged(built, golden):
    robot, _ = built
    keys = [k for k in golden.files if k.startswith("cyl/")]
    assert keys, "golden file has no cylinders"
    seen = set()
    for key in keys:
        _, link, field = key.split("/")
        seen.add(link)
        cyl = robot.links[link].cylinder
        assert cyl is not None, f"{link} lost its cylinder"
        np.testing.assert_allclose(
            np.asarray(getattr(cyl, field), dtype=float), golden[key],
            err_msg=f"{link}.{field} changed", **EXACT,
        )
    built_links = {n for n, ld in robot.links.items() if ld.cylinder is not None}
    assert built_links == seen, "set of links carrying a cylinder changed"


@pytest.mark.parametrize("fname,args", [
    ("f_M_rb", ("q",)), ("f_C_rb", ("q", "v")), ("f_g_rb", ("q",)),
    ("f_M_added", ("q",)), ("f_tau_buoyancy", ("q",)), ("f_tau_drag", ("q", "v")),
    ("f_C_A_v", ("q", "v")), ("f_tau_added", ("q", "v", "a")),
    ("f_inverse_dynamics", ("q", "v", "a")), ("f_forward_dynamics", ("q", "v", "a")),
])
def test_casadi_functions_unchanged(built, golden, fname, args):
    _, dyn = built
    Q, V, A = golden["sample_q"], golden["sample_v"], golden["sample_a"]
    fn = getattr(dyn, fname)
    rows = []
    for i in range(len(Q)):
        call = {"q": Q[i], "v": V[i], "a": A[i]}
        rows.append(np.asarray(fn(*[call[a] for a in args])).ravel())
    np.testing.assert_allclose(np.array(rows), golden[f"fn/{fname}"], **TIGHT)


def test_tangent_dynamics_unchanged(built, golden):
    """The function the coordinate map injects into — the highest-risk surface."""
    _, dyn = built
    q_ref_quat = np.array([0.0, 0.0, 0.0, 1.0])
    f_kin, f_inv_dyn = dyn.build_tangent_dynamics(q_ref_quat)
    XT, AR = golden["sample_xt"], golden["sample_ar"]
    kin = np.array([np.asarray(f_kin(XT[i])).ravel() for i in range(len(XT))])
    inv = np.array([np.asarray(f_inv_dyn(XT[i], AR[i])).ravel() for i in range(len(XT))])
    np.testing.assert_allclose(kin, golden["fn/f_kin"], **TIGHT)
    np.testing.assert_allclose(inv, golden["fn/f_inv_dyn"], **TIGHT)


def test_tangent_conversions_unchanged(built, golden):
    robot, _ = built
    nq, nv = robot.nq, robot.nv
    q_ref_quat = np.array([0.0, 0.0, 0.0, 1.0])
    X_leg = np.zeros((nq + nv, len(golden["sample_q"])))
    X_leg[:nq, :] = golden["sample_q"].T
    X_leg[nq:, :] = golden["sample_v"].T
    X_tan = legacy_to_tangent(X_leg, q_ref_quat, robot.model)
    np.testing.assert_allclose(X_tan, golden["legacy_to_tangent"], **TIGHT)
    np.testing.assert_allclose(
        tangent_to_legacy(X_tan, q_ref_quat, robot.model),
        golden["tangent_to_legacy"], **TIGHT,
    )


def test_trim_and_feet_unchanged(built, golden):
    robot, dyn = built
    # scipy's optimizer path can wobble in the last bits; 1e-9 still pins the pose
    np.testing.assert_allclose(dyn.find_trim_state(), golden["trim_state"], rtol=1e-9, atol=1e-12)
    robot.forward_kinematics(golden["trim_state"])
    fp = robot.foot_positions()
    got = np.array([fp[leg] for leg in sorted(fp)])
    np.testing.assert_allclose(got, golden["foot_positions"], **TIGHT)
