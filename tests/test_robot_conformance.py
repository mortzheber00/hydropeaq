"""Contract every registered robot must satisfy.

Parametrized over the registry, so adding a robot automatically subjects it to
the same checks.  Several of these exist because pinocchio fails *silently* on
a bad name: ``getFrameId`` returns ``nframes`` rather than raising, which only
shows up much later as a wrong number.
"""

from __future__ import annotations

import casadi as ca
import numpy as np
import pinocchio as pin
import pytest
from hydro_model import LocalPoint, load_robot, registry
from hydro_model.robots import RobotSpec

ROBOTS = sorted(registry())


@pytest.fixture(scope="module")
def built():
    return {name: load_robot(name) for name in ROBOTS}


def _spec(name) -> RobotSpec:
    return registry()[name]


def _tree_angles(robot, q_joints) -> np.ndarray:
    """Joint angles of the tree, given the map's configuration block.

    Continuous joints store ``(cos, sin)``; revolute ones store the angle.
    Returns one angle per velocity DOF, in model joint order.
    """
    q = np.asarray(q_joints, dtype=float).ravel()
    model = robot.model
    out = np.zeros(model.nv - 6)
    for jid in range(1, model.njoints):
        j = model.joints[jid]
        if j.nv == 0:
            continue
        iq, iv = j.idx_q - 7, j.idx_v - 6
        if iv < 0:
            continue                       # the free-flyer root
        if j.nq == 2:
            out[iv] = np.arctan2(q[iq + 1], q[iq])
        else:
            out[iv] = q[iq]
    return out


def _sample_theta(robot, rng, n=12):
    """Random configurations that satisfy the robot's own feasibility test."""
    spec = robot.spec
    home = (np.zeros(robot.n_actuated) if spec.theta_home is None
            else np.asarray(spec.theta_home, float))
    cmap = robot.coord_map
    f_feas = None
    th = ca.SX.sym("th", robot.n_actuated)
    feas = cmap.feasibility(th)
    if feas.numel() > 0:
        f_feas = ca.Function("feas", [th], [feas])

    out = []
    while len(out) < n:
        cand = home + rng.uniform(-0.15, 0.15, size=robot.n_actuated)
        if f_feas is None or float(np.min(np.asarray(f_feas(cand)))) > 1e-6:
            out.append(cand)
    return np.array(out)


@pytest.mark.parametrize("name", ROBOTS)
def test_urdf_and_model_build(name):
    spec = _spec(name)
    assert spec.urdf_path.exists(), spec.urdf_path
    assert spec.package_dir.is_dir(), spec.package_dir
    model = pin.buildModelFromUrdf(str(spec.urdf_path), pin.JointModelFreeFlyer())
    assert model.nv > 6


@pytest.mark.parametrize("name", ROBOTS)
def test_every_spec_name_resolves(built, name):
    """The check that catches pinocchio's silent unknown-name behaviour."""
    robot = built[name]
    model, spec = robot.model, robot.spec

    for j in spec.actuated_joint_names:
        assert model.existJointName(j), f"no joint {j!r}"
    assert model.existFrame(spec.base_link), f"no frame {spec.base_link!r}"

    for leg, (frame, _) in spec.foot_points.items():
        assert model.existFrame(frame), f"foot frame {frame!r} for leg {leg}"
    for leg, jname in spec.leg_plane_joint.items():
        assert model.existJointName(jname), f"plane joint {jname!r} for leg {leg}"

    for cs in spec.cylinders:
        assert model.existFrame(cs.link), f"cylinder link {cs.link!r}"
        for ep in (cs.start, cs.end, cs.at):
            if ep is None:
                continue
            if isinstance(ep, LocalPoint):
                assert model.existFrame(ep.link), f"LocalPoint link {ep.link!r}"
            else:
                assert model.existJointName(ep) or model.existFrame(ep), (
                    f"endpoint {ep!r} is neither joint nor frame"
                )
                # _resolve_point prefers the joint.  Pinocchio always mirrors a
                # joint as a JOINT-type frame of the same name, which is benign;
                # a BODY frame shadowing an unrelated joint would not be.
                if model.existJointName(ep):
                    ftype = model.frames[model.getFrameId(ep)].type
                    assert ftype == pin.FrameType.JOINT, (
                        f"{ep!r} is a joint but also a {ftype} frame — "
                        f"_resolve_point would silently pick the joint"
                    )
        if cs.kind == "copy":
            assert cs.ref is not None and model.existFrame(cs.ref)


@pytest.mark.parametrize("name", ROBOTS)
def test_actuated_ordering_is_leg_major(built, name):
    """stage3 indexes joints as ``7 + leg*n_per_leg + j``; enforce that layout."""
    robot = built[name]
    spec = robot.spec
    n_legs, labels = len(spec.leg_names), spec.leg_joint_labels
    assert robot.n_actuated == robot.coord_map.n_theta
    assert robot.n_actuated == n_legs * len(labels), (
        f"{robot.n_actuated} actuated joints is not {n_legs} legs x {len(labels)} labels"
    )
    for i, leg in enumerate(spec.leg_names):
        block = spec.actuated_joint_names[i * len(labels):(i + 1) * len(labels)]
        assert all(leg in jn for jn in block), f"leg {leg} block is {block}"


def _leg_side(leg: str, which: str) -> bool:
    """Front/hind and left/right from the leg name.

    Both robots name legs so the first letter gives fore/aft (F vs B or H) and
    a later L/R gives the side: ``Front_Left`` / ``FL``.
    """
    if which == "front":
        return leg[0].upper() == "F"
    tail = leg.upper().split("_")[-1]
    return tail.startswith("L") or tail.endswith("L")   # "Left" or "FL"


@pytest.mark.parametrize("name", ROBOTS)
def test_base_frame_convention(built, name):
    """+x forward, +y to the robot's left, origin near the body.

    A URDF exported from CAD often has none of these — BODY2's base frame
    started 1.1 m away, facing -x, which silently made the pipeline reward
    swimming *backwards* because the objective rewards +x displacement.
    """
    robot = built[name]
    model, data = robot.model, robot.model.createData()
    q0 = pin.neutral(model)
    com = np.asarray(pin.centerOfMass(model, data, q0))
    pin.forwardKinematics(model, data, q0)

    hips = {}
    for leg in robot.spec.leg_names:
        jname = next(j for j in robot.spec.actuated_joint_names if leg in j)
        hips[leg] = np.asarray(data.oMi[model.getJointId(jname)].translation)
    span = np.ptp(np.array(list(hips.values())), axis=0).max()

    assert np.linalg.norm(com) < span, (
        f"base frame is {np.linalg.norm(com):.3f} m from the CoM, which is more "
        f"than the robot's own {span:.3f} m hip span — re-frame the base"
    )
    front = np.mean([p[0] for leg, p in hips.items() if _leg_side(leg, "front")])
    hind = np.mean([p[0] for leg, p in hips.items() if not _leg_side(leg, "front")])
    assert front > hind, f"front legs are at x={front:+.3f}, behind hind at {hind:+.3f}: +x is not forward"

    left = np.mean([p[1] for leg, p in hips.items() if _leg_side(leg, "side")])
    right = np.mean([p[1] for leg, p in hips.items() if not _leg_side(leg, "side")])
    assert left > right, f"left legs are at y={left:+.3f}, right at {right:+.3f}: +y is not left"


@pytest.mark.parametrize("name", ROBOTS)
def test_cylinders_are_physical(built, name):
    robot = built[name]
    for cs in robot.spec.cylinders:
        cyl = robot.links[cs.link].cylinder
        assert cyl is not None, f"{cs.link} has no cylinder"
        assert cyl.radius > 0, f"{cs.link} radius {cyl.radius}"
        assert cyl.length > 0, f"{cs.link} length {cyl.length}"
        assert cyl.volume_displaced > 0, f"{cs.link} volume {cyl.volume_displaced}"
        np.testing.assert_allclose(np.linalg.norm(cyl.axis_local), 1.0, atol=1e-9)
        np.testing.assert_allclose(np.linalg.norm(cyl.axis_world), 1.0, atol=1e-9)


@pytest.mark.parametrize("name", ROBOTS)
def test_map_shapes_and_unit_pairs(built, name):
    """Continuous joints need ``cos^2 + sin^2 == 1`` or pinocchio misbehaves."""
    robot = built[name]
    cmap, model = robot.coord_map, robot.model
    assert cmap.nq_j == model.nq - 7
    assert cmap.nv_j == model.nv - 6

    theta = (np.zeros(robot.n_actuated) if robot.spec.theta_home is None
             else np.asarray(robot.spec.theta_home, float))
    q_j = np.asarray(cmap.expand_numeric(theta), dtype=float).ravel()
    assert q_j.shape == (cmap.nq_j,)

    for jid in range(1, model.njoints):
        j = model.joints[jid]
        if j.nq == 2 and j.idx_q >= 7:
            iq = j.idx_q - 7
            np.testing.assert_allclose(q_j[iq] ** 2 + q_j[iq + 1] ** 2, 1.0, atol=1e-12)


@pytest.mark.parametrize("name", ROBOTS)
def test_S_matches_finite_differences(built, name):
    robot = built[name]
    cmap = robot.coord_map
    th = ca.SX.sym("th", robot.n_actuated)
    f_S = ca.Function("S", [th], [cmap.S(th)])
    f_q = ca.Function("q", [th], [cmap.q_joints(th)])

    rng = np.random.default_rng(4)
    h = 1e-6
    for theta in _sample_theta(robot, rng):
        S = np.asarray(f_S(theta))
        for j in range(robot.n_actuated):
            e = np.zeros(robot.n_actuated)
            e[j] = h
            a_p = _tree_angles(robot, f_q(theta + e))
            a_m = _tree_angles(robot, f_q(theta - e))
            d = np.arctan2(np.sin(a_p - a_m), np.cos(a_p - a_m)) / (2 * h)
            np.testing.assert_allclose(S[:, j], d, atol=1e-5,
                                       err_msg=f"S column {j} disagrees with FD")


@pytest.mark.parametrize("name", ROBOTS)
def test_a_joints_is_the_derivative_of_v_joints(built, name):
    """Checks the Sdot*thd term — the most error-prone expression in the map."""
    robot = built[name]
    cmap = robot.coord_map
    n = robot.n_actuated
    th, thd, thdd = (ca.SX.sym("th", n), ca.SX.sym("thd", n), ca.SX.sym("thdd", n))
    f_v = ca.Function("v", [th, thd], [cmap.v_joints(th, thd)])
    f_a = ca.Function("a", [th, thd, thdd], [cmap.a_joints(th, thd, thdd)])

    rng = np.random.default_rng(5)
    h = 1e-6
    zero = np.zeros(n)
    for theta in _sample_theta(robot, rng, n=6):
        thd_v = rng.normal(scale=0.3, size=n)
        got = np.asarray(f_a(theta, thd_v, zero)).ravel()
        fd = (np.asarray(f_v(theta + h * thd_v, thd_v)).ravel()
              - np.asarray(f_v(theta - h * thd_v, thd_v)).ravel()) / (2 * h)
        np.testing.assert_allclose(got, fd, atol=1e-5)


@pytest.mark.parametrize("name", ROBOTS)
def test_power_is_consistent(built, name):
    """Virtual work: thd . S^T tau == (S thd) . tau.  Catches a transposed S."""
    robot = built[name]
    cmap = robot.coord_map
    n, nv_j = robot.n_actuated, robot.coord_map.nv_j
    th, thd = ca.SX.sym("th", n), ca.SX.sym("thd", n)
    tau = ca.SX.sym("tau", nv_j)
    f_v = ca.Function("v", [th, thd], [cmap.v_joints(th, thd)])
    f_t = ca.Function("t", [th, tau], [cmap.tau_joints(th, tau)])

    rng = np.random.default_rng(6)
    for theta in _sample_theta(robot, rng, n=6):
        thd_v = rng.normal(size=n)
        tau_v = rng.normal(size=nv_j)
        lhs = float(thd_v @ np.asarray(f_t(theta, tau_v)).ravel())
        rhs = float(np.asarray(f_v(theta, thd_v)).ravel() @ tau_v)
        np.testing.assert_allclose(lhs, rhs, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("name", ROBOTS)
def test_reduced_dynamics_is_affine_in_acceleration(built, name):
    """``M_r a_r + b_r`` must reproduce the reduced inverse dynamics exactly.

    The whole extraction rests on inverse dynamics being affine in
    acceleration; if it were not, the Jacobian-and-substitute trick would give
    a silently wrong mass matrix rather than an error.
    """
    from hydro_model import SymbolicDynamics

    robot = built[name]
    dyn = SymbolicDynamics(robot)
    f_Mb = dyn.build_reduced_dynamics()

    # the same residual the extraction is taken from, kept independent of it
    cmap, n = robot.coord_map, robot.n_actuated
    qb, th = ca.SX.sym("qb", 7), ca.SX.sym("th", n)
    vr, ar = ca.SX.sym("vr", 6 + n), ca.SX.sym("ar", 6 + n)
    tau_tree = dyn.f_inverse_dynamics(
        ca.vertcat(qb, cmap.q_joints(th)),
        ca.vertcat(vr[:6], cmap.v_joints(th, vr[6:])),
        ca.vertcat(ar[:6], cmap.a_joints(th, vr[6:], ar[6:])),
    )
    f_res = ca.Function("res", [qb, th, vr, ar],
                        [ca.vertcat(tau_tree[:6], cmap.tau_joints(th, tau_tree[6:]))])

    rng = np.random.default_rng(9)
    q_base = np.array([0.0, 0.0, 0.03, 0.0, 0.0, 0.0, 1.0])
    for theta in _sample_theta(robot, rng, n=4):
        v_r = rng.normal(scale=0.2, size=6 + n)
        a_r = rng.normal(size=6 + n)
        M_r, b_r = (np.asarray(x) for x in f_Mb(q_base, theta, v_r))
        b_r = b_r.ravel()
        got = np.asarray(f_res(q_base, theta, v_r, a_r)).ravel()
        np.testing.assert_allclose(got, M_r @ a_r + b_r, atol=1e-12)

        # and solving it back recovers the acceleration
        tau_r = M_r @ a_r + b_r
        np.testing.assert_allclose(np.linalg.solve(M_r, tau_r - b_r), a_r, atol=1e-9)


@pytest.mark.parametrize("name", ROBOTS)
def test_reduced_forward_dynamics_matches_tree_for_serial_robots(built, name):
    """A serial robot must be dispatched to the untouched tree path."""
    from hydro_model import IdentityMap, SymbolicDynamics

    robot = built[name]
    if not isinstance(robot.coord_map, IdentityMap):
        pytest.skip("closed-chain robot has no tree-space equivalent")

    dyn = SymbolicDynamics(robot)
    rng = np.random.default_rng(10)
    for _ in range(5):
        q_base = np.array([0.0, 0.0, 0.03, 0.0, 0.0, 0.0, 1.0])
        theta = rng.uniform(-0.3, 0.3, size=robot.n_actuated)
        v_r = rng.normal(scale=0.2, size=robot.nv_reduced)
        tau_r = np.concatenate([np.zeros(6), rng.normal(scale=0.05, size=robot.n_actuated)])
        got = dyn.eval_reduced_forward_dynamics(q_base, theta, v_r, tau_r)
        want = dyn.eval_forward_dynamics(np.concatenate([q_base, theta]), v_r, tau_r)
        np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("name", ROBOTS)
def test_home_pose_is_strictly_feasible(built, name):
    robot = built[name]
    spec = robot.spec
    if spec.pose_constraints is None:
        pytest.skip("no pose constraints")
    theta = (np.zeros(robot.n_actuated) if spec.theta_home is None
             else np.asarray(spec.theta_home, float))
    th = ca.SX.sym("th", robot.n_actuated)
    eqs, ineqs = spec.pose_constraints(th)
    for expr in eqs:
        assert np.isfinite(float(ca.Function("e", [th], [expr])(theta)))
    for expr in ineqs:
        val = float(ca.Function("i", [th], [expr])(theta))
        assert val > 0, f"home pose violates an inequality ({val})"


@pytest.mark.parametrize("name", ROBOTS)
def test_trim_converges(built, name):
    from hydro_model import SymbolicDynamics

    robot = built[name]
    q_trim = SymbolicDynamics(robot).find_trim_state()
    assert q_trim.shape == (robot.nq,)
    assert np.all(np.isfinite(q_trim))
