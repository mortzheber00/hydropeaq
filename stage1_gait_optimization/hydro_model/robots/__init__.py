"""Robot registry.

Each robot is described by a ``RobotSpec`` in its own module here. To add a
robot, write the module and list it in ``_build_registry``.

    from hydro_model import load_robot
    robot = load_robot("amph")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class LocalPoint:
    """Point fixed in a link, in link coordinates (e.g. loop-closure pins, foot tips)."""

    link: str
    xyz: Tuple[float, float, float]


Endpoint = Union[str, LocalPoint]


@dataclass(frozen=True)
class CylinderSpec:
    """How to build one link's cylinder.

    ``kind="segment"``  axis and length span ``start`` -> ``end``
    ``kind="copy"``     cap of fixed ``length`` at ``at`` with ``ref``'s axis
    ``kind="inertia"``  axis and length from the inertia tensor, at the CoM

    ``project_leg`` projects the endpoints onto that leg's sagittal plane.
    ``radius`` and ``added_mass_volume`` override the mesh-derived values for
    links the cylinder fits badly.
    """

    link: str
    kind: str = "segment"
    start: Optional[Endpoint] = None
    end: Optional[Endpoint] = None
    ref: Optional[str] = None
    at: Optional[Endpoint] = None
    length: Optional[float] = None
    project_leg: Optional[str] = None
    radius: Optional[float] = None              # None -> mesh RMS radius
    added_mass_volume: Optional[float] = None   # None -> displaced volume


@dataclass(frozen=True)
class OCPSettings:
    """Gait-OCP discretisation and cost weights for one robot (defaults: amph)."""

    gait: str = "Prototype"   # initial-guess builder; see run_collocation.build_ocp
    n: int = 48               # collocation intervals
    d_colloc: int = 3         # polynomial degree (Radau collocation points)
    t_init: float = 1.0       # initial-guess cycle period [s]
    t_min: float = 0.8        # cycle-period bounds [s]
    t_max: float = 1.2
    d_target: float = 0.15     # forward distance per nominal cycle [m]
    tau_max: float = 3.5      # joint torque limit [Nm]
    f_c: float = 20.0         # actuator bandwidth [Hz], first-order filter cutoff
    w_power: float = 2.0      # squared joint power (τ·q̇)²
    w_dist: float = 0.0       # forward distance reward
    w_vel_smooth: float = 20.0  # base acceleration
    # Joint acceleration; ~15 % of the power term at the slow corner (v=0.10,
    # T=1.4). Recheck if w_power, tau_max or the speed range change.
    w_joint_smooth: float = 0.2
    w_drift: float = 10.0     # lateral/vertical drift
    heading_tol: float = 0.05   # max yaw angle at endpoint [rad]

    # Constrain each lr_leg_pairs right leg to mirror the left with a phase shift.
    enforce_symmetry: bool = True
    # Initial phase lag of the right leg [cycles]: scalar, one per pair, or
    # None to detect it from the initial guess (preferred).
    symmetry_phase: Optional[Union[float, Tuple[float, ...]]] = None

    @property
    def v_target(self) -> float:
        """Required average forward speed [m/s]."""
        return self.d_target / self.t_init


@dataclass(frozen=True)
class RobotSpec:
    """Declarative description of one robot."""

    name: str
    urdf_path: Path
    package_dir: Path            # ROS package directory

    # --- Naming ---
    leg_names: Tuple[str, ...]
    actuated_joint_names: Tuple[str, ...]   # grouped by leg
    leg_joint_labels: Tuple[str, ...]       # per-leg joint labels for plots
    base_link: str = "base_link"
    ros_name: Optional[str] = None          # ROS namespace / Gazebo model; defaults to name

    # --- Left-right symmetry ---
    # (right, left) leg pairs and the per-joint sign mapping a left leg onto its
    # right partner. Empty disables enforce_symmetry.
    lr_leg_pairs: Tuple[Tuple[str, str], ...] = ()
    mirror_joint_sign: Optional[Tuple[float, ...]] = None   # length = joints per leg
    # Per-leg joints to constrain (None = all). Exclude joints already pinned by
    # pose_constraints, otherwise the constraints become rank-deficient.
    symmetry_joints: Optional[Tuple[int, ...]] = None

    # --- Geometry ---
    # leg -> (frame, offset in that frame) of the foot point
    foot_points: Dict[str, Tuple[str, Tuple[float, float, float]]] = field(default_factory=dict)
    # leg -> joint whose origin and local y-axis define the leg's sagittal plane
    leg_plane_joint: Dict[str, str] = field(default_factory=dict)
    recenter_base_y: bool = False
    cylinders: Tuple[CylinderSpec, ...] = ()
    volume_overrides: Dict[str, float] = field(default_factory=dict)

    # --- Reduced coordinates ---
    coordinate_map: Optional[Callable[[object], object]] = None   # None -> IdentityMap

    # Actuated-joint limits (length n_theta); None -> URDF limits
    theta_lower: Optional[np.ndarray] = None
    theta_upper: Optional[np.ndarray] = None
    theta_vel_limit: Optional[np.ndarray] = None
    theta_effort_limit: Optional[np.ndarray] = None
    theta_home: Optional[np.ndarray] = None

    # theta -> (equalities == 0, inequalities >= 0) at one configuration
    pose_constraints: Optional[Callable] = None

    # --- Initial guesses and OCP ---
    # Overrides for build_robot_ik_initial_guess parameters (stroke sizes in metres).
    firmware_gait: Dict[str, float] = field(default_factory=dict)
    # Transform of the paper gait's foot path for the hind legs, about the hip:
    # hind_rotation_deg (clockwise, +x forward, +z up), hind_dx, hind_dz [m].
    paper_gait: Dict[str, float] = field(default_factory=dict)

    ocp: OCPSettings = field(default_factory=OCPSettings)

    @property
    def package_root(self) -> Path:
        """Parent of the package directory, as needed to resolve ``package://`` URIs."""
        return self.package_dir.parent

    @property
    def n_theta(self) -> int:
        return len(self.actuated_joint_names)

    @property
    def ros(self) -> str:
        """ROS namespace / Gazebo model name."""
        return self.ros_name or self.name


# --- Registry ---

def _build_registry() -> Dict[str, RobotSpec]:
    from . import amph, body2
    return {s.name: s for s in (amph.SPEC, body2.SPEC)}


_REGISTRY: Optional[Dict[str, RobotSpec]] = None


def registry() -> Dict[str, RobotSpec]:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _build_registry()
    return _REGISTRY


def get_spec(name_or_path) -> RobotSpec:
    """Look up a spec by registry name or URDF path."""
    if isinstance(name_or_path, RobotSpec):
        return name_or_path
    reg = registry()
    key = str(name_or_path)
    if key in reg:
        return reg[key]
    resolved = Path(key).resolve()
    for spec in reg.values():
        if spec.urdf_path.resolve() == resolved:
            return spec
    raise KeyError(
        f"no robot registered as {key!r}; known robots: {sorted(reg)}. "
        f"To add one, write hydro_model/robots/<name>.py and list it in _build_registry()."
    )


def load_robot(name_or_path, *, q: Optional[Sequence[float]] = None):
    """Create a robot, run FK at ``q`` (default: neutral) and build its cylinders."""
    import pinocchio as pin

    from ..robot import QuadrupedRobot

    robot = QuadrupedRobot(get_spec(name_or_path))
    robot.forward_kinematics(pin.neutral(robot.model) if q is None
                             else np.asarray(q, dtype=float))
    robot.build_cylinders()
    return robot
