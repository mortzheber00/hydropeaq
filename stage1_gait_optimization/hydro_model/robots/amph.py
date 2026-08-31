"""amph — 4 serial legs, 3 revolute joints each, foot fixed to the calf.

This spec reproduces the behaviour the pipeline had before the registry
existed; ``tests/test_amph_regression.py`` is the proof.
"""

from __future__ import annotations

from functools import lru_cache

from . import REPO_ROOT, CylinderSpec, OCPSettings, RobotSpec

LEG_NAMES = ("Front_Left", "Front_Right", "Hind_Left", "Hind_Right")
JOINTS_PER_LEG = ("Side_joint", "Thigh_joint", "Calf_joint")

_PKG = REPO_ROOT / "src" / "amph"

# Longitudinal clearance held between a front shank and the hind shank behind
# it [m].  Shank against shank is the only pair that ever closes up -- the
# thighs never do -- so requiring the front one to lie wholly ahead of the hind
# one in body x separates them: a vertical line then passes between the two
# segments, which proves they cannot intersect.
SHANK_CLEARANCE = 0.015

# (right, left) is irrelevant here; these are the two same-side front/hind pairs
# sharing a sagittal plane, and so the only ones that can touch.
_SHANK_PAIRS = (("Front_Left", "Hind_Left"), ("Front_Right", "Hind_Right"))


@lru_cache(maxsize=1)
def _shank_x():
    """``theta -> body-frame x`` of every leg's calf joint and foot.

    Built on its own pinocchio model because pose_constraints has to run before
    any QuadrupedRobot exists.  Safe for an x-only constraint: the robot's
    ``_recenter_base_y`` fixup moves the base frame in y alone.

    Returned as a ``ca.Function`` so one expression serves both callers -- the
    OCP passes CasADi symbols, ``foot_ik.pose_margin`` passes floats.
    """
    import casadi as ca
    import pinocchio as pin
    import pinocchio.casadi as cpin

    model = pin.buildModelFromUrdf(
        str(_PKG / "urdf" / "amph.urdf"), pin.JointModelFreeFlyer()
    )
    cmodel = cpin.Model(model)
    cdata = cmodel.createData()
    theta = ca.SX.sym("theta", len(JOINTS_PER_LEG) * len(LEG_NAMES))
    # Base at the identity, so world coordinates already are body coordinates.
    cpin.forwardKinematics(cmodel, cdata, ca.vertcat(0, 0, 0, 0, 0, 0, 1, theta))
    cpin.updateFramePlacements(cmodel, cdata)
    out = []
    for leg in LEG_NAMES:
        out.append(cdata.oMi[cmodel.getJointId(f"{leg}_Calf_joint")].translation[0])
        out.append(cdata.oMf[cmodel.getFrameId(f"{leg}_Foot_link")].translation[0])
    return ca.Function("shank_x", [theta], [ca.vertcat(*out)])


def _pose_constraints(theta):
    """Pin every side joint to zero, and keep front and hind shanks apart.

    The side joints are the planar-gait assumption, previously hardcoded as
    ``ocp_common.py:321-322``:
        for side_idx in range(6, 6 + n_act, 3): X[side_idx, k] == 0
    """
    eqs = [theta[i] for i in range(0, 3 * len(LEG_NAMES), 3)]

    x = _shank_x()(theta)               # calf x, foot x per leg, in LEG_NAMES order
    at = {leg: 2 * i for i, leg in enumerate(LEG_NAMES)}
    ineqs = [
        x[at[front] + a] - x[at[hind] + b] - SHANK_CLEARANCE
        for front, hind in _SHANK_PAIRS
        for a in (0, 1)                 # front calf joint, front foot
        for b in (0, 1)                 # hind calf joint, hind foot
    ]
    return (eqs, ineqs)


def _leg_cylinders(leg: str):
    return (
        CylinderSpec(f"{leg}_Side_link", start=f"{leg}_Side_joint",
                     end=f"{leg}_Thigh_joint", project_leg=leg),
        CylinderSpec(f"{leg}_Thigh_link", start=f"{leg}_Thigh_joint",
                     end=f"{leg}_Calf_joint", project_leg=leg),
        CylinderSpec(f"{leg}_Calf_link", start=f"{leg}_Calf_joint",
                     end=f"{leg}_Foot_link", project_leg=leg),
        # thin cap sharing the calf's axis
        CylinderSpec(f"{leg}_Foot_link", kind="copy", ref=f"{leg}_Calf_link",
                     at=f"{leg}_Foot_link", length=0.01, project_leg=leg),
    )


SPEC = RobotSpec(
    name="amph",
    urdf_path=_PKG / "urdf" / "amph.urdf",
    package_dir=_PKG,
    leg_names=LEG_NAMES,
    actuated_joint_names=tuple(f"{leg}_{j}" for leg in LEG_NAMES for j in JOINTS_PER_LEG),
    leg_joint_labels=("Side", "Thigh", "Calf"),
    # Only the out-of-plane Side joint flips; Thigh and Calf face the same way
    # on both sides.  Side is pinned to zero anyway, so its sign never binds --
    # and for the same reason it is left out of symmetry_joints.
    lr_leg_pairs=(("Front_Right", "Front_Left"), ("Hind_Right", "Hind_Left")),
    mirror_joint_sign=(-1.0, 1.0, 1.0),
    symmetry_joints=(1, 2),          # Thigh, Calf; Side is pinned by pose_constraints
    foot_points={leg: (f"{leg}_Foot_link", (0.0, 0.0, 0.0)) for leg in LEG_NAMES},
    leg_plane_joint={leg: f"{leg}_Side_joint" for leg in LEG_NAMES},
    recenter_base_y=True,
    cylinders=(CylinderSpec("base_link", kind="inertia"),)
    + tuple(c for leg in LEG_NAMES for c in _leg_cylinders(leg)),
    coordinate_map=None,          # IdentityMap
    pose_constraints=_pose_constraints,
    # The paper path as published leaves amph's hind reach: 59% of it sits
    # outside, and the unconstrained IK tracks it anyway by driving the hind
    # thigh 29 deg through its lower stop.  Rotating the path 40 deg clockwise
    # about the hip and shifting it 3 cm aft brings it into range.
    paper_gait={"hind_rotation_deg": 45.0, "hind_dx": 0.01},
    # OCPSettings' defaults are amph's own numbers, so this is spelled out only
    # to say the gait explicitly.  Do not tune them for another robot here.
    ocp=OCPSettings(gait="LSPG25"),
)
