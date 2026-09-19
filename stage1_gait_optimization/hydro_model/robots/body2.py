"""BODY2: four closed-loop planar legs with two hip servos each.

Each leg is a 2-DOF five-bar (P1-P2-P3-P6-P4) driving a parallelogram
(P5-P6-P8-P7) that carries the foot. The URDF only holds the spanning tree;
loop closure is handled by ``Body2CoordinateMap``, so joint limits are given
here in the 8 reduced coordinates.

Pin geometry is stored in ``body2_linkage.json``
(regenerate with ``src/BODY2/scripts/dump_linkage.py``).
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

    ROS names cannot contain dots. Duplicated in leg_linkage_sim.py.
    """
    return f"Joint_{leg}{key.replace('.', '_')}"

_PKG = REPO_ROOT / "src" / "BODY2"
LINKAGE_PATH = Path(__file__).resolve().parent / "body2_linkage.json"
LINKAGE = json.loads(LINKAGE_PATH.read_text())

# The leg only assembles in a diagonal band of the (theta1, theta2) plane, e.g.
#     theta2 = -30 deg  ->  theta1 in [-49, +5] deg
#     theta2 =   0 deg  ->  theta1 in [-12, +50] deg
#     theta2 = +10 deg  ->  theta1 in [ -1, +136] deg
# Home is the CAD pose (zeros): inside the band, but off-centre.
THETA_HOME = np.zeros(len(LEG_NAMES) * 2)

# Hip travel (lower, upper) [deg] of joints 1.1 and 2.1; mirrored for the right side.
_HIP_BOX_DEG = {"L": ((-66.0, 180.0), (-40.0, 105.0)),
                "R": ((-180.0, 66.0), (-105.0, 40.0))}

# Grouped by leg to match actuated_joint_names: FL1.1, FL2.1, FR1.1, ...
HIP_BOX = np.radians([_HIP_BOX_DEG[leg[1]][i]
                      for leg in LEG_NAMES for i in range(len(JOINTS_PER_LEG))])

# Minimum loop half-chord [m]; reaching zero would flip the assembly branch
# (home has 12.5 mm).
H_MIN = 2e-3

# --- Foot (Link_*2.3) ---
# The foot is a thin stalk carrying a 32 x 8 mm blade whose chord always points
# along world y, so it is always broadside to the flow. Its drag and added mass
# are therefore measured from the mesh instead of using the generic fit:
#   radius: 2*r*L matches the measured face-on area (1287 mm^2)
#   volume: strip-theory added mass 24.7 g * k with aspect-ratio factor k = 0.6
#           (square plate 0.61, disc 0.64), expressed as a volume for Ca_t = 1.3
# TODO(calibration): fit k (plausible 0.55-0.70, i.e. 13.6-17.3 g).
FOOT_RADIUS = 9.729e-3              # [m]
FOOT_ADDED_MASS_VOLUME = 1.145e-5   # [m^3] -> 14.8 g entrained at Ca_t = 1.3

# --- Chassis (base_link) ---
# The chassis mesh is an open frame of disjoint parts and not watertight, so the
# convex-hull fallback would overestimate its volume ~6x. Use the mesh's signed
# volume instead (the robot is still ~1.6x positively buoyant).
CHASSIS_VOLUME = 1.913528183066247e-4   # [m^3] signed volume of base_link.STL


def _leg_cylinders(leg: str):
    """One cylinder per link between the pins it connects.

    Joint origins sit on their pins; P6, P8 and the foot tip have no joint and
    use the link-local offsets from the linkage file.
    """
    lp = LINKAGE["legs"][leg]["local_points"]
    return (
        CylinderSpec(f"Link_{leg}1.1", start=urdf_joint(leg, "1.1"), end=urdf_joint(leg, "1.2")),
        CylinderSpec(f"Link_{leg}1.2", start=urdf_joint(leg, "1.2"), end=urdf_joint(leg, "1.3")),
        CylinderSpec(f"Link_{leg}1.3", start=urdf_joint(leg, "1.3"),
                     end=LocalPoint(f"Link_{leg}1.3", tuple(lp["P8"]))),
        # Link 2.1 carries P5 and P6; use the longer P4->P6 span.
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
    """Keep both loops of every leg assemblable: h^2 >= H_MIN^2.

    Scaled as ``h^2/H_MIN^2 - 1`` so the constraint is O(1); the raw form in
    m^2 is badly scaled for IPOPT.
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
    # Both hips flip sign between left and right.
    lr_leg_pairs=(("FR", "FL"), ("BR", "BL")),
    mirror_joint_sign=(-1.0, -1.0),
    ros_name="BODY2",               # robot name in the URDF
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
    # Servo travel; assemblability is enforced by pose_constraints.
    theta_lower=HIP_BOX[:, 0],
    theta_upper=HIP_BOX[:, 1],
    # TODO(hardware): placeholders, not servo datasheet values; the effort limit
    # is likely ~15x too high for this robot.
    theta_vel_limit=np.full(len(LEG_NAMES) * 2, 15*12.0),
    theta_effort_limit=np.full(len(LEG_NAMES) * 2, 3.5),
    theta_home=THETA_HOME,
    pose_constraints=_pose_constraints,
    # Stroke sized to BODY2's much smaller reach; amph's defaults are far out of
    # range. Around home, +-20 mm in x leaves +-26 mm in z, +-30 mm leaves +-16 mm.
    firmware_gait={
        "stroke_len": 0.042,
        "stand_h": 0.085,       # home foot depth below the base frame origin
        "depth_surface": 0.085,  # recovery runs at the trim depth
        "depth_deep": 0.1158,     # power stroke depth
        "center_x_front": -0.03,
        "center_x_rear": -0.03,
    },
    # Weights rebalanced for BODY2's much smaller torques.
    ocp=OCPSettings(
        gait="Prototype",
        d_target=0.15,
        w_power=200.0,
        w_vel_smooth=0.2,
    ),
)
