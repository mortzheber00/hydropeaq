"""BODY2 — four closed-loop planar legs, two hip servos each.

Each leg is a seven-link, 2-DOF planar mechanism: a five-bar
(P1-P2-P3-P6-P4) driving a parallelogram (P5-P6-P8-P7) that carries the foot.
URDF cannot express the two loop-closure pins, so the model pinocchio builds is
a 24-joint tree and the loops live in ``Body2CoordinateMap`` instead.

Two consequences show up in this file:

* the tree's ``continuous`` joints are *unbounded* in pinocchio, so its
  configuration block is 48 wide against 24 velocities and its position limits
  bound ``(cos, sin)`` rather than angles.  The OCP works in the 8 reduced
  coordinates, so the spec supplies its own limits.
* the legs are already planar, so no sagittal projection is needed and
  ``leg_plane_joint`` stays empty.

The pin geometry is frozen in ``body2_linkage.json``; regenerate it with
``src/BODY2/scripts/dump_linkage.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import REPO_ROOT, CylinderSpec, LocalPoint, OCPSettings, RobotSpec

LEG_NAMES = ("FL", "FR", "BL", "BR")
JOINTS_PER_LEG = ("1.1", "2.1")          # the two hip servos, both on base_link


def urdf_joint(leg: str, key: str) -> str:
    """URDF joint name for a linkage key: ``("FL", "1.1") -> "Joint_FL1_1"``.

    Pins are named ``1.1``-style here and in the coordinate map because that is
    the CAD's nomenclature, but ROS graph resource names forbid dots, so the
    URDF spells the same joints with an underscore.  Link names and mesh files
    keep the dots; they never become ROS names.

    ``src/BODY2/scripts/leg_linkage_sim.py`` carries the same one-liner; the
    two packages do not import each other.
    """
    return f"Joint_{leg}{key.replace('.', '_')}"

_PKG = REPO_ROOT / "src" / "BODY2"
LINKAGE_PATH = Path(__file__).resolve().parent / "body2_linkage.json"
LINKAGE = json.loads(LINKAGE_PATH.read_text())

# Reduced coordinates are (theta1, theta2) per leg.  The mechanism assembles
# over ~41% of the (theta1, theta2) square, in a diagonal band whose centreline
# runs at roughly d(theta1)/d(theta2) = 1.37.  Measured slices through the band
# that contains the zero pose:
#
#     theta2 = -30 deg  ->  theta1 in [-49, +5] deg
#     theta2 =   0 deg  ->  theta1 in [-12, +50] deg
#     theta2 = +10 deg  ->  theta1 in [ -1, +136] deg
#
# The URDF zero pose is the CAD assembly pose and sits inside the band with a
# 12.5 mm half-chord, so it is the home.  Note it is *not* centred: the band
# only extends 12 deg in the -theta1 direction, which is what BAND_DIR is for.
THETA_HOME = np.zeros(len(LEG_NAMES) * 2)

# Unit vector along the band centreline, in (theta1, theta2).  A gait that
# sweeps along this stays assemblable far longer than one that does not.
BAND_DIR = np.array([1.37, 1.0]) / np.linalg.norm([1.37, 1.0])

# Minimum circle-circle half-chord the OCP must keep.  Crossing zero is the
# only way the mechanism can flip assembly branch, and home has 12.5 mm of
# slack, so 2 mm is a wide margin.
H_MIN = 2e-3

# --- Foot (Link_*2.3) --------------------------------------------------------
# The foot is not a rod: it is a 3 mm stalk over the inner ~40% of its 66 mm
# span carrying a 32 mm wide, 8 mm thick blade over the outer ~45%.  The mesh
# RMS radius the generic fit produces (7.54 mm) gets both drag and added mass
# wrong, so both are measured off the mesh instead.
#
# Every leg joint turns about world y and the blade's 32 mm chord *is* world y,
# so leg motion cannot rotate the blade out of the sagittal plane: the foot is
# permanently broadside to its own crossflow.  One transverse direction is ever
# exercised, which is why a plain cylinder still suffices here.
#
# radius:  chosen so the transverse drag area 2*r*L reproduces the measured
#          face-on silhouette, 1287 mm^2.  (For a link that really is a
#          cylinder this is an identity, so the fit is unchanged elsewhere.)
# volume:  strip theory gives added mass rho*(pi/4)*k*int c(s)^2 ds = 24.7 g * k
#          for the measured chord distribution.  Stored as the volume the model
#          multiplies by Ca_transverse=1.3 to reach that figure.
#
# k is the finite-aspect-ratio factor: strip theory treats every slice as 2D,
# but a finite plate lets fluid escape past its tips, so k < 1.  The blade is
# ~30 mm span on a 32 mm chord (AR ~ 1), and two shapes with exact solutions at
# that aspect ratio agree closely there — a square plate gives 0.478/(pi/4) =
# 0.61, a circular disc (8/3)rho a^3 against its own strip integral gives 0.64.
# Hence 0.60.  It reaches ~0.70 by AR 2 and 1.0 in the 2D limit.
#
# TODO(calibration): treat k as a fit parameter, not a derived truth — the
# plausible band (~0.55-0.70) puts the entrained mass between 13.6 and 17.3 g.
FOOT_RADIUS = 9.729e-3              # [m]
FOOT_ADDED_MASS_VOLUME = 1.145e-5   # [m^3] -> 14.8 g entrained at Ca_t = 1.3

# --- Chassis (base_link) -----------------------------------------------------
# base_link.STL is 26 disjoint solids -- plates, rails, mounts -- with no inner
# walls: every component is positively oriented and they sum to exactly the
# mesh's signed volume, 191.35 cm^3.  It is an open frame, so the water between
# those parts is water, not displacement.
#
# The generic fit cannot see that.  The mesh fails ``is_watertight`` (85 of its
# 85524 edges are non-manifold where components touch, though the surface does
# close -- its vector area is exactly zero), so QuadrupedRobot falls back to the
# convex hull.  That hull spans the open frame as if it were solid: 1178.6 cm^3,
# 6.2x the material.  It put buoyancy at 11.9 N against a 1.41 N weight, floated
# the robot with ~12% of the hull wetted, and gave the chassis 1.53 kg of added
# mass against a 143 g robot -- which is what let gait optimisation extract
# thrust from the surface-crossing terms the hydro model does not model.
#
# 191.35 cm^3 still leaves the robot 1.58x positively buoyant, so it trims at
# the surface rather than submerged; that is a separate question from this one.
CHASSIS_VOLUME = 1.913528183066247e-4   # [m^3] signed volume of base_link.STL


def _leg_cylinders(leg: str):
    """One cylinder per link, spanning the pins that link physically joins.

    After ``rehome_urdf.py`` every leg joint origin sits exactly on its pin, so
    the joint names resolve to the right points; the three pins that carry no
    joint (P6, P8 and the foot tip) come from the frozen link-local offsets.
    """
    lp = LINKAGE["legs"][leg]["local_points"]
    return (
        CylinderSpec(f"Link_{leg}1.1", start=urdf_joint(leg, "1.1"), end=urdf_joint(leg, "1.2")),
        CylinderSpec(f"Link_{leg}1.2", start=urdf_joint(leg, "1.2"), end=urdf_joint(leg, "1.3")),
        CylinderSpec(f"Link_{leg}1.3", start=urdf_joint(leg, "1.3"),
                     end=LocalPoint(f"Link_{leg}1.3", tuple(lp["P8"]))),
        # Link 2.1 carries both P5 and P6; P4->P6 is the longer, more
        # hydrodynamically relevant span (50 mm vs 30 mm).
        CylinderSpec(f"Link_{leg}2.1", start=urdf_joint(leg, "2.1"),
                     end=LocalPoint(f"Link_{leg}2.1", tuple(lp["P6"]))),
        CylinderSpec(f"Link_{leg}2.2", start=urdf_joint(leg, "2.2"), end=urdf_joint(leg, "2.3")),
        CylinderSpec(f"Link_{leg}2.3", start=urdf_joint(leg, "2.3"),
                     end=LocalPoint(f"Link_{leg}2.3", tuple(lp["tip"])),
                     radius=FOOT_RADIUS,
                     added_mass_volume=FOOT_ADDED_MASS_VOLUME),
    )


def _coordinate_map(robot):
    from .body2_map import Body2CoordinateMap

    return Body2CoordinateMap(robot)


def _pose_constraints(theta):
    """Keep both loops assemblable: h^2 >= H_MIN^2 at every configuration.

    Crossing h = 0 is the only way the mechanism can change assembly branch,
    so this doubles as the guard that keeps the closed-form solution valid.

    Written as ``h^2/H_MIN^2 - 1`` rather than ``h^2 - H_MIN^2``.  The two have
    the same zero set, so the feasible region and the active set are identical,
    but the raw form carries units of m^2 and runs from 5e-6 to 4e-4 over a
    typical gait -- five orders below the O(1) scale an interior-point method
    assumes.  IPOPT starts at mu = 1 and drives every slack toward it, so a
    slack that small leaves a barrier gradient of order mu/s ~ 1e5 on each of
    these rows.  There are 776 of them in the OCP (8 loops x 97 configuration
    points), and together they swamp every other term in the KKT system.
    """
    from .body2_map import feasibility_expr

    return ([], [expr / H_MIN ** 2 - 1.0
                 for expr in feasibility_expr(theta, LEG_NAMES, LINKAGE)])


SPEC = RobotSpec(
    name="body2",
    urdf_path=_PKG / "urdf" / "BODY2.urdf",
    package_dir=_PKG,
    leg_names=LEG_NAMES,
    actuated_joint_names=tuple(
        urdf_joint(leg, j) for leg in LEG_NAMES for j in JOINTS_PER_LEG
    ),
    leg_joint_labels=JOINTS_PER_LEG,
    ros_name="BODY2",               # the URDF's robot name, unlike the registry key
    foot_points={
        leg: (f"Link_{leg}2.3", tuple(LINKAGE["legs"][leg]["local_points"]["tip"]))
        for leg in LEG_NAMES
    },
    leg_plane_joint={},                  # legs are already planar; no projection
    recenter_base_y=False,               # base frame sits ~1.1 m from the body
    cylinders=(CylinderSpec("base_link", kind="inertia"),)
    + tuple(c for leg in LEG_NAMES for c in _leg_cylinders(leg)),
    volume_overrides={"base_link": CHASSIS_VOLUME},
    coordinate_map=_coordinate_map,
    # The hips are continuous servos; travel is limited by the linkage (via
    # pose_constraints), not by a joint stop.
    theta_lower=np.full(len(LEG_NAMES) * 2, -np.pi),
    theta_upper=np.full(len(LEG_NAMES) * 2, np.pi),
    # TODO(hardware): placeholders carried over from amph.  BODY2 is a 143 g
    # robot with 0.2 g leg links, so 3.5 N*m is roughly 15x too large and any
    # gait optimised against it is qualitative only.  Replace with the servo
    # datasheet figures before trusting a solution.
    theta_vel_limit=np.full(len(LEG_NAMES) * 2, 12.0),
    theta_effort_limit=np.full(len(LEG_NAMES) * 2, 3.5),
    theta_home=THETA_HOME,
    pose_constraints=_pose_constraints,
    # Firmware-gait stroke, in metres of foot travel.  Measured on the FL foot
    # over the assemblable band: the reachable box about the home foot trades
    # fore-aft against vertical travel — +-20 mm in x allows +-26 mm in z,
    # +-30 mm allows +-16 mm, and past +-45 mm there is none left.  25 mm of
    # stroke with a 20 mm downward swing therefore sits inside the box with
    # room to spare, and the swing only ever goes down, using half of it.
    #
    # amph's defaults are 2x the stroke and 3x the depth of BODY2's *entire*
    # reachable box, so leaving them in place would put every foot target off
    # the workspace.
    firmware_gait={
        "stroke_len": 0.025,
        "stand_h": 0.0958,       # home foot depth below the base frame origin
        "depth_surface": 0.0958,  # recovery runs at the trim depth
        "depth_deep": 0.1158,     # power stroke 20 mm below it
        "center_x_front": 0.0,    # stroke centred on the trim foot, both ends
        "center_x_rear": 0.0,
    },
    # The firmware IK gait needs a foot path; BODY2's is written straight in
    # theta instead, because theta *is* the actuated coordinate once the
    # coordinate map has the loops (see initial_guess/theta_sinusoid.py).
    #
    # The three weights that differ from amph's defaults were tuned against
    # this robot: it is 143 g against amph's, so the same 0.2 m/cycle target
    # asks for a stroke the linkage's assemblable band cannot reach, and the
    # power and smoothing terms have to be rebalanced around the smaller
    # torques that go with it.
    ocp=OCPSettings(
        gait="ThetaSinusoid",
        d_target=0.1,
        w_power=200.0,
        w_vel_smooth=0.2,
    ),
)
