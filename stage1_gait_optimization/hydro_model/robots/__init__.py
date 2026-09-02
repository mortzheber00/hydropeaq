"""Robot registry.

Everything the pipeline needs to know about a particular robot lives in a
``RobotSpec``.  Adding a model means writing one module here and registering
it — no pipeline code changes.

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
    """A point fixed in a link, given in that link's local body frame.

    Needed for geometry that no joint sits on — loop-closure pins and foot
    tips.  Plain ``str`` endpoints are resolved as a joint name first and a
    frame name second.
    """

    link: str
    xyz: Tuple[float, float, float]


Endpoint = Union[str, LocalPoint]


@dataclass(frozen=True)
class CylinderSpec:
    """Recipe for one link's hydrodynamic cylinder primitive.

    ``kind="segment"``  axis and length span ``start`` -> ``end``.
    ``kind="copy"``     thin cap at ``at`` reusing ``ref``'s axis and radius
                        source, with a fixed ``length`` (amph's feet).
    ``kind="inertia"``  axis and length from the inertia eigenvectors, centred
                        on the link CoM (the base link).

    ``project_leg`` projects both endpoints onto that leg's sagittal plane
    before measuring, which is what removes amph's lateral offset.

    ``radius`` and ``added_mass_volume`` override the two quantities that are
    otherwise measured off the mesh.  Both exist for links the cylinder fits
    badly — see the foot note in ``body2.py``.
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
    """Gait-OCP transcription and cost weights for one robot.

    These used to be module globals in ``trajopt/run_collocation.py``, which
    meant retuning them for one robot silently retuned the other.  The defaults
    below are amph's committed values, so a robot that omits the field gets the
    behaviour the pipeline had before this type existed.
    """

    gait: str = "Prototype"   # initial-guess builder; see run_collocation.build_ocp
    n: int = 48               # collocation intervals
    d_colloc: int = 3         # polynomial degree (Radau collocation points)
    t_init: float = 1.0       # initial-guess cycle period [s] (warm start)
    t_min: float = 0.8        # cycle-period bounds [s]
    t_max: float = 1.2
    d_target: float = 0.1     # forward distance per nominal cycle [m]
    tau_max: float = 3.5      # joint torque limit [Nm]
    f_c: float = 20.0         # actuator bandwidth [Hz] — first-order filter cutoff
    w_power: float = 2.0      # weight for sum-of-squared per-joint power (τ·q̇)²
    w_dist: float = 0.0       # weight for forward distance reward
    w_vel_smooth: float = 20.0  # weight for base velocity smoothing
    # Joint-acceleration smoothing.  Sized so the term contributes ~15% of the
    # power term at the low-speed corner (v=0.10, T=1.4), matching the share
    # w_vel_smooth carries there; that corner is where low torque utilisation
    # leaves the joint motion otherwise unregularised.  Retune with the power
    # weight, and check the share again if tau_max or the speed range moves.
    w_joint_smooth: float = 0.2
    w_drift: float = 10.0     # weight for drift penalty
    heading_tol: float = 0.05   # max yaw angle at endpoint [rad]

    # Mirror each RobotSpec.lr_leg_pairs left leg onto its right, phase-shifted.
    enforce_symmetry: bool = True
    # Seed for the cycle fraction the right leg lags the left; the OCP solves
    # for it from there.  None reads it off the initial guess (prefer that);
    # set a scalar, or one value per pair, to force a different phasing.
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
    package_dir: Path            # pinocchio buildGeomFromUrdf root (the package itself)

    # ordering / naming
    leg_names: Tuple[str, ...]
    actuated_joint_names: Tuple[str, ...]   # ordered, grouped leg-major
    leg_joint_labels: Tuple[str, ...]       # per-leg column labels for stage3
    base_link: str = "base_link"
    # ROS namespace and Gazebo model name (stage2).  Defaults to ``name``;
    # set it when the URDF's robot name differs from the registry key.
    ros_name: Optional[str] = None

    # (right, left) mirror pairs, plus the per-joint sign carrying a left leg's
    # block onto its mirrored right one.  The sign follows from how the legs are
    # mounted; check a candidate by confirming both trace the same hip-relative
    # foot path.  Empty -> no mirror, and enforce_symmetry cannot be used.
    lr_leg_pairs: Tuple[Tuple[str, str], ...] = ()
    mirror_joint_sign: Optional[Tuple[float, ...]] = None   # length = joints per leg
    # Per-leg joint indices the symmetry parametrisation covers; None -> all.
    # Leave out any joint pose_constraints already pins outright: the two would
    # stack into rank-deficient rows and overdetermine the NLP.
    symmetry_joints: Optional[Tuple[int, ...]] = None

    # leg -> (frame carrying the foot, offset in that frame's local coordinates)
    foot_points: Dict[str, Tuple[str, Tuple[float, float, float]]] = field(default_factory=dict)
    # leg -> joint whose origin and local Y define that leg's sagittal plane
    leg_plane_joint: Dict[str, str] = field(default_factory=dict)

    # geometry / model fixups
    recenter_base_y: bool = False
    cylinders: Tuple[CylinderSpec, ...] = ()
    volume_overrides: Dict[str, float] = field(default_factory=dict)

    # None -> IdentityMap
    coordinate_map: Optional[Callable[[object], object]] = None

    # Reduced-coordinate limits, each length n_theta.  None -> read from the
    # URDF via the model, which is what the pipeline did before specs existed.
    theta_lower: Optional[np.ndarray] = None
    theta_upper: Optional[np.ndarray] = None
    theta_vel_limit: Optional[np.ndarray] = None
    theta_effort_limit: Optional[np.ndarray] = None
    theta_home: Optional[np.ndarray] = None

    # theta -> (equalities == 0, inequalities >= 0) at one configuration
    pose_constraints: Optional[Callable] = None

    # Overrides for the firmware IK gait's shape, which is specified in metres
    # of foot travel and so does not transfer between robots of different size.
    # Keys are the tunable parameters of ``build_robot_ik_initial_guess``;
    # anything omitted keeps that function's default.
    firmware_gait: Dict[str, float] = field(default_factory=dict)

    # Rigid transform placing the paper gait's foot path into the *hind* legs'
    # reach.  paper.py builds one path from the front leg's kinematics and asks
    # every leg to trace it, but the hind legs are mounted rotated 180 deg, so
    # their reachable set is the front one mirrored in x and the shared path can
    # fall outside it.  Applied in the sagittal hip-relative plane, about the
    # hip, rotation before translation.  Keys: ``hind_rotation_deg`` (clockwise
    # positive, viewed with +x forward and +z up), ``hind_dx`` and ``hind_dz``
    # in metres.  Empty -> the hind feet trace the front path unchanged, which
    # is what paper.py did before this field existed.
    paper_gait: Dict[str, float] = field(default_factory=dict)

    # Gait-OCP horizon and cost weights; see OCPSettings.
    ocp: OCPSettings = field(default_factory=OCPSettings)

    @property
    def package_root(self) -> Path:
        """Directory *containing* the package — what ``package://`` resolvers want."""
        return self.package_dir.parent

    @property
    def n_theta(self) -> int:
        return len(self.actuated_joint_names)

    @property
    def ros(self) -> str:
        """ROS namespace / Gazebo model name."""
        return self.ros_name or self.name


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

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
    """Look a spec up by registry name, or by the URDF path it declares.

    The path form keeps the pre-registry call sites (``QuadrupedRobot(URDF_PATH)``)
    working unchanged.
    """
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
    """Build a robot and bring it to the state every driver expects.

    Replaces the ``QuadrupedRobot(...); forward_kinematics(...); build_cylinders()``
    triple that was copied into eight scripts.

    ``q`` defaults to ``pin.neutral``.  The old scripts passed ``np.zeros(nq)``,
    which for amph gives bit-identical cylinders (its zero quaternion is
    renormalised and its joint block is zeros either way).  It is *not*
    equivalent for a robot with continuous joints, whose configuration entries
    are ``(cos, sin)`` pairs: zeros is not a rotation and silently collapses the
    kinematics onto the parent joint.
    """
    import pinocchio as pin

    from ..robot import QuadrupedRobot

    robot = QuadrupedRobot(get_spec(name_or_path))
    robot.forward_kinematics(pin.neutral(robot.model) if q is None
                             else np.asarray(q, dtype=float))
    robot.build_cylinders()
    return robot
