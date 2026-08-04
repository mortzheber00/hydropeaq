"""BODY2's frozen linkage data and its CasADi coordinate map.

The pipeline reads pin geometry from ``body2_linkage.json`` rather than
re-deriving it from the STL hole walls at import time.  That is only safe if
something checks the two still agree, which is what this module is for.
"""

from __future__ import annotations

import json

import casadi as ca
import numpy as np
import pinocchio as pin
import pytest
from hydro_model import load_robot
from hydro_model.robots.body2 import H_MIN, LEG_NAMES, LINKAGE, LINKAGE_PATH
from hydro_model.robots.body2_map import JOINT_KEYS, _solve_leg
from leg_linkage_sim import Leg

# Feasible sample points around the home pose, in degrees.  The band at
# theta = 0 runs from -12 to +50 deg in theta1, so this stays well inside.
GRID = np.radians(np.linspace(-6.0, 6.0, 7))


@pytest.fixture(scope="module")
def robot():
    return load_robot("body2")


@pytest.mark.parametrize("leg", LEG_NAMES)
def test_frozen_geometry_matches_the_meshes(leg):
    """Re-derive from the STLs and compare — the guard that makes freezing safe."""
    ref = Leg(leg)
    got = LINKAGE["legs"][leg]
    for k, v in ref.pins0.items():
        np.testing.assert_allclose(got["pins"][k], v, atol=1e-9, err_msg=f"{leg} pin {k}")
    np.testing.assert_allclose(got["tip"], ref.tip0, atol=1e-9)
    for k, v in ref.L.items():
        np.testing.assert_allclose(got["lengths"][k], v, atol=1e-9, err_msg=f"{leg} len {k}")
    assert got["sgn"] == {k: float(v) for k, v in ref.sgn.items()}
    assert LINKAGE["branch"] == [float(b) for b in ref.branch]


@pytest.mark.parametrize("leg", LEG_NAMES)
def test_map_matches_the_reference_solver(leg):
    """The CasADi port must reproduce leg_linkage_sim to round-off."""
    ld = LINKAGE["legs"][leg]
    q = ca.SX.sym("q", 2)
    pairs, h_sq = _solve_leg(q[0], q[1], ld, LINKAGE["branch"])
    f = ca.Function("f", [q], [ca.vertcat(*[ca.vertcat(*pairs[k]) for k in JOINT_KEYS]),
                              ca.vertcat(*h_sq)])
    ref = Leg(leg)

    checked = 0
    for q1 in GRID:
        for q2 in GRID:
            sol = ref.solve(q1, q2)
            pair_v, h_v = (np.asarray(x).ravel() for x in f([q1, q2]))
            assert bool(sol["ok"]) == bool(np.all(h_v > 0)), "feasibility disagrees"
            if not sol["ok"]:
                continue
            checked += 1
            for i, key in enumerate(JOINT_KEYS):
                c, s = pair_v[2 * i], pair_v[2 * i + 1]
                np.testing.assert_allclose(np.hypot(c, s), 1.0, atol=1e-12)
                mine = np.arctan2(s, c)
                want = (q1 if key == "1.1" else q2 if key == "2.1"
                        else float(sol["joints"][f"Joint_{leg}{key}"]))
                err = np.arctan2(np.sin(mine - want), np.cos(mine - want))
                assert abs(err) < 1e-10, f"{leg} joint {key}: {mine} vs {want}"
    assert checked > 20, f"only {checked} feasible samples"


def test_forward_kinematics_closes_both_loops(robot):
    """The point of the whole exercise.

    Nothing in the URDF ties the two sub-chains of a leg together.  If the
    reduced coordinates are right, then feeding ``q_joints(theta)`` through
    pinocchio's FK must place the two loop-closure pins at the same world
    point from either side.  Nothing else in the suite checks this.
    """
    model, data, cmap = robot.model, robot.data, robot.coord_map

    # Local coordinates of each shared pin in *both* owning links, read off the
    # assembled zero pose.  The two links are stacked plates at different
    # depths on the same pin, so both are anchored to the point where the pin
    # axis crosses y = 0; otherwise they would differ by their depth offset.
    pin.forwardKinematics(model, data, pin.neutral(model))
    pin.updateFramePlacements(model, data)
    shared = {}
    for leg in LEG_NAMES:
        info = LINKAGE["legs"][leg]
        for pin_name, links in (("P6", ("2.1", "1.3")), ("P8", ("1.3", "2.3"))):
            xz = np.asarray(info["pins"][pin_name])
            axis_pt = np.array([xz[0], 0.0, xz[1]])
            local = []
            for lk in links:
                oMf = data.oMf[model.getFrameId(f"Link_{leg}{lk}")]
                R, o = np.asarray(oMf.rotation), np.asarray(oMf.translation)
                local.append(R.T @ (axis_pt - o))
            shared[(leg, pin_name)] = (links, local)

    rng = np.random.default_rng(11)
    worst = 0.0
    for _ in range(25):
        theta = rng.uniform(-0.08, 0.08, size=robot.n_actuated)
        if np.min(np.asarray(cmap.feasibility(theta))) <= H_MIN ** 2:
            continue
        q = pin.neutral(model)
        q[7:] = cmap.expand_numeric(theta)
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        for (leg, pin_name), (links, local) in shared.items():
            pts = []
            for lk, loc in zip(links, local):
                oMf = data.oMf[model.getFrameId(f"Link_{leg}{lk}")]
                pts.append(np.asarray(oMf.translation) + np.asarray(oMf.rotation) @ loc)
            worst = max(worst, float(np.linalg.norm(pts[0] - pts[1])))
    assert worst < 1e-9, f"loop closure violated by {worst * 1e3:.6f} mm"


def test_home_pose_is_assembled_and_matches_the_urdf(robot):
    """theta = 0 must reproduce the URDF's own zero configuration."""
    q_j = robot.coord_map.expand_numeric(np.zeros(robot.n_actuated))
    np.testing.assert_allclose(q_j, pin.neutral(robot.model)[7:], atol=1e-12)
    h = np.asarray(robot.coord_map.feasibility(np.zeros(robot.n_actuated))).ravel()
    assert np.all(h > H_MIN ** 2)
    assert np.sqrt(h.min()) > 5e-3, "home has less than 5 mm of assembly margin"


def test_linkage_file_is_committed_and_small():
    assert LINKAGE_PATH.exists()
    assert LINKAGE_PATH.stat().st_size < 40_000
    json.loads(LINKAGE_PATH.read_text())          # parses


def test_foot_cylinder_matches_the_paddle_mesh(robot):
    """``FOOT_RADIUS`` must still reproduce the blade's face-on silhouette.

    The foot is a stalk carrying a flat blade, so the generic mesh RMS radius
    understates the area the flow sees.  ``FOOT_RADIUS`` is a measured constant
    instead, which goes stale silently if the mesh is ever re-exported — hence
    this check.  The silhouette is taken along the blade normal, the direction
    the foot is permanently broadside to.
    """
    import trimesh

    for leg in LEG_NAMES:
        name = f"Link_{leg}2.3"
        cyl = robot.links[name].cylinder
        mesh = trimesh.load(robot.link_geom_objects[name].meshPath)

        # Work in the mesh's own principal frame: the silhouette is a property
        # of the shape, so nothing here may depend on the FK state the shared
        # fixture happens to be left in.  Halve the projected face area because
        # every ray through a closed mesh crosses it twice.
        pts = np.asarray(mesh.vertices) - np.asarray(mesh.vertices).mean(axis=0)
        _, _, axes = np.linalg.svd(pts, full_matrices=False)
        area = max(
            0.5 * float(np.abs(mesh.area_faces * (mesh.face_normals @ a)).sum())
            for a in axes
        )  # the blade normal is by definition the widest silhouette

        assert abs(area - 1287e-6) < 10e-6, f"{name} silhouette {area * 1e6:.0f} mm^2"
        np.testing.assert_allclose(
            cyl.cross_section_transverse, area, rtol=2e-3,
            err_msg=f"{name}: 2*r*L no longer matches the blade silhouette",
        )


def test_foot_added_mass_is_decoupled_from_buoyancy(robot):
    """Buoyancy keeps the mesh volume; added mass uses the swept volume."""
    for leg in LEG_NAMES:
        cyl = robot.links[f"Link_{leg}2.3"].cylinder
        assert cyl.volume_displaced == pytest.approx(4.461e-6, rel=1e-2)
        assert cyl.volume_entrained == pytest.approx(1.145e-5, rel=1e-9)
        assert cyl.volume_entrained > 2.0 * cyl.volume_displaced

    # Links without an override must be unaffected.
    base = robot.links["base_link"].cylinder
    assert base.volume_entrained == base.volume_displaced
