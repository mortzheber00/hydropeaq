"""amph — 4 serial legs, 3 revolute joints each, foot fixed to the calf.

This spec reproduces the behaviour the pipeline had before the registry
existed; ``tests/test_amph_regression.py`` is the proof.
"""

from __future__ import annotations

from . import REPO_ROOT, CylinderSpec, RobotSpec

LEG_NAMES = ("Front_Left", "Front_Right", "Hind_Left", "Hind_Right")
JOINTS_PER_LEG = ("Side_joint", "Thigh_joint", "Calf_joint")

_PKG = REPO_ROOT / "src" / "amph"


def _pose_constraints(theta):
    """Pin every side joint to zero — the planar-gait assumption.

    Previously hardcoded as ``ocp_common.py:321-322``:
        for side_idx in range(6, 6 + n_act, 3): X[side_idx, k] == 0
    """
    return ([theta[i] for i in range(0, 3 * len(LEG_NAMES), 3)], [])


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
    foot_points={leg: (f"{leg}_Foot_link", (0.0, 0.0, 0.0)) for leg in LEG_NAMES},
    leg_plane_joint={leg: f"{leg}_Side_joint" for leg in LEG_NAMES},
    recenter_base_y=True,
    cylinders=(CylinderSpec("base_link", kind="inertia"),)
    + tuple(c for leg in LEG_NAMES for c in _leg_cylinders(leg)),
    coordinate_map=None,          # IdentityMap
    pose_constraints=_pose_constraints,
)
