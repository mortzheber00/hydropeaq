"""
Quadruped robot model built on Pinocchio with cylinder-primitive approximations.

Pinocchio handles URDF parsing, kinematic tree, forward kinematics, and
Jacobians.  On top of that, each link is approximated as a solid cylinder
(dimensions inferred from the URDF inertia tensor) for the hydrodynamic model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pinocchio as pin


# ---------------------------------------------------------------------------
# Geometric primitive
# ---------------------------------------------------------------------------

@dataclass
class CylinderPrimitive:
    """Solid-cylinder approximation of a robot link.

    For leg links the cylinder spans between two skeleton joints
    (centerline-projected).  The radius is derived from the smallest
    principal moment of inertia: I_sym = m*r^2/2  →  r = sqrt(2*I_sym/m).
    """
    radius: float           # [m]
    length: float           # [m]
    center: np.ndarray      # midpoint of the cylinder in world frame [m]
    axis_world: np.ndarray  # unit vector along the cylinder axis (world frame)
    axis_local: np.ndarray  # same axis expressed in the link body frame

    @property
    def volume(self) -> float:
        return np.pi * self.radius**2 * self.length

    @property
    def cross_section_axial(self) -> float:
        """End-cap area (flow along the cylinder axis)."""
        return np.pi * self.radius**2

    @property
    def cross_section_transverse(self) -> float:
        """Projected area for flow perpendicular to the cylinder axis."""
        return 2.0 * self.radius * self.length

    @staticmethod
    def radius_from_inertia(mass: float, inertia: np.ndarray) -> float:
        """Estimate cylinder radius from the smallest principal inertia.

        I_sym = m * r^2 / 2  →  r = sqrt(2 * I_min / m)
        """
        I_min = np.linalg.eigvalsh(inertia).min()
        return np.sqrt(max(2.0 * I_min / mass, 1e-10))

    @classmethod
    def from_segment(
        cls,
        p_start: np.ndarray,
        p_end: np.ndarray,
        mass: float,
        inertia: np.ndarray,
        R_frame: np.ndarray,
    ) -> CylinderPrimitive:
        """Build a cylinder spanning from p_start to p_end (world frame).

        Length = distance between the two points.
        Radius = derived from the inertia tensor.
        R_frame : 3x3 rotation of the link frame at build time (body to world frame).
        """
        diff = p_end - p_start
        length = float(np.linalg.norm(diff))
        length = max(length, 1e-6)
        axis_world = diff / length
        axis_local = R_frame.T @ axis_world
        center = (p_start + p_end) / 2.0
        radius = cls.radius_from_inertia(mass, inertia)
        return cls(radius=radius, length=length, center=center,
                   axis_world=axis_world, axis_local=axis_local)

    @classmethod
    def from_inertia_only(
        cls,
        mass: float,
        inertia: np.ndarray,
        center: np.ndarray,
        R_world: np.ndarray,
    ) -> CylinderPrimitive:
        """Fallback for links without a clear joint-to-joint segment (e.g. base).

        Axis and length are inferred from the inertia tensor eigenvalues.
        """
        eigvals, eigvecs = np.linalg.eigh(inertia)
        idx_min = np.argmin(eigvals)
        I_sym = eigvals[idx_min]
        I_trans = np.mean([eigvals[i] for i in range(3) if i != idx_min])

        # For a solid cylinder, the inertia about the central axis is I_sym = m*r^2/2,
        # and the inertia about any transverse axis is I_trans = m*(3*r^2 + h^2)/12.
        r_sq = 2.0 * I_sym / mass
        radius = np.sqrt(max(r_sq, 1e-10))
        h_sq = 12.0 * I_trans / mass - 3.0 * r_sq
        length = np.sqrt(max(h_sq, 1e-10))

        axis_local = eigvecs[:, idx_min]
        axis_world = R_world @ axis_local
        return cls(radius=radius, length=length, center=center,
                   axis_world=axis_world, axis_local=axis_local)


# ---------------------------------------------------------------------------
# Per-link data combining Pinocchio inertia with cylinder geometry
# ---------------------------------------------------------------------------

@dataclass
class LinkData:
    """Physical properties of a single link, derived from Pinocchio."""
    name: str
    frame_id: int          # Pinocchio frame index (BODY type)
    parent_joint: int      # Pinocchio joint index that moves this link
    mass: float            # [kg]
    com_local: np.ndarray  # center of mass in link frame [m]
    inertia: np.ndarray    # 3x3 inertia tensor at CoM, link frame
    cylinder: CylinderPrimitive | None = field(default=None, init=False)


# ---------------------------------------------------------------------------
# Canonical ordering
# ---------------------------------------------------------------------------

LEG_NAMES = ["Front_Left", "Front_Right", "Hind_Left", "Hind_Right"]
JOINTS_PER_LEG = ["Side_joint", "Thigh_joint", "Calf_joint"]


# ---------------------------------------------------------------------------
# Main robot class
# ---------------------------------------------------------------------------

class QuadrupedRobot:
    """Quadruped robot backed by Pinocchio with cylinder-approximated links.

    Parameters
    ----------
    urdf_path : str | Path
        Path to the URDF file.
    """

    def __init__(self, urdf_path: str | Path):
        self.urdf_path = Path(urdf_path)

        # Build Pinocchio model (kinematic + dynamic, no geometry meshes)
        self.model: pin.Model = pin.buildModelFromUrdf(str(self.urdf_path))
        self.data: pin.Data = self.model.createData()

        # Extract per-link data from Pinocchio frames
        self.links: dict[str, LinkData] = {}
        self._extract_links()

        # Ordered actuated joint names (12 total: 3 per leg × 4 legs)
        self.actuated_joint_names: list[str] = []
        self._build_joint_order()

        # Map from our ordered joint index to Pinocchio joint index
        self.joint_pin_ids: list[int] = [
            self.model.getJointId(name) for name in self.actuated_joint_names
        ]

        # Foot frame IDs for easy access
        self.foot_frame_ids: dict[str, int] = {}
        for leg in LEG_NAMES:
            fname = f"{leg}_Foot_link"
            fid = self.model.getFrameId(fname)
            self.foot_frame_ids[leg] = fid

    # ------------------------------------------------------------------
    # Internal setup
    # ------------------------------------------------------------------

    def _extract_links(self):
        """Pull mass, CoM, and inertia from Pinocchio BODY frames."""
        for frame in self.model.frames:
            if frame.type != pin.FrameType.BODY:
                continue
            name = frame.name
            joint_id = frame.parentJoint

            # Pinocchio stores inertia per joint, not per frame (frame corresponds to urdf link).
            # This holds for standard URDFs where each link has a single parent joint.
            inertia_pin = self.model.inertias[joint_id]
            mass = inertia_pin.mass
            if mass < 1e-8:
                continue

            com_local = np.array(inertia_pin.lever)  # CoM in joint frame
            # Inertia tensor at CoM, expressed in joint frame
            I_matrix = np.array(inertia_pin.inertia)

            self.links[name] = LinkData(
                name=name,
                frame_id=self.model.getFrameId(name),
                parent_joint=joint_id,
                mass=mass,
                com_local=com_local,
                inertia=I_matrix,
            )

    def _build_joint_order(self):
        """Build canonical actuated joint ordering."""
        for leg in LEG_NAMES:
            for suffix in JOINTS_PER_LEG:
                jname = f"{leg}_{suffix}"
                if self.model.existJointName(jname):
                    self.actuated_joint_names.append(jname)

    # ------------------------------------------------------------------
    # Forward kinematics (delegated to Pinocchio)
    # ------------------------------------------------------------------

    @property
    def nq(self) -> int:
        """Number of configuration variables."""
        return self.model.nq

    @property
    def nv(self) -> int:
        """Number of velocity variables (= nq for revolute joints)."""
        return self.model.nv

    @property
    def n_actuated(self) -> int:
        return len(self.actuated_joint_names)

    def forward_kinematics(
        self,
        q: np.ndarray,
        v: np.ndarray | None = None,
        a: np.ndarray | None = None,
    ) -> pin.Data:
        """Run Pinocchio FK and update frame placements.

        Parameters
        ----------
        q : (nq,) joint configuration.
        v : (nv,) joint velocities (optional, needed for velocity-level FK).
        a : (nv,) joint accelerations (optional, needed for acceleration-level FK).

        Returns
        -------
        data : pinocchio.Data with updated oMi and oMf placements.
        """
        if v is None and a is None:
            pin.forwardKinematics(self.model, self.data, q)
        elif a is None:
            pin.forwardKinematics(self.model, self.data, q, v)
        else:
            pin.forwardKinematics(self.model, self.data, q, v, a)
        pin.updateFramePlacements(self.model, self.data)
        return self.data

    def frame_placement(self, frame_id: int) -> pin.SE3:
        """World-frame SE3 placement of a frame (call FK first)."""
        return self.data.oMf[frame_id]

    def link_world_poses(self) -> dict[str, np.ndarray]:
        """4x4 world-frame transforms for every link (call FK first).

        Returns dict mapping link name → (4,4) homogeneous transform.
        """
        poses = {}
        for name, link in self.links.items():
            oMf = self.data.oMf[link.frame_id]
            poses[name] = oMf.homogeneous
        return poses

    def link_world_coms(self) -> dict[str, np.ndarray]:
        """World-frame CoM position for each link (call FK first)."""
        coms = {}
        for name, link in self.links.items():
            oMf = self.data.oMf[link.frame_id]
            # CoM is stored relative to the joint frame; the BODY frame
            # may have an offset, but for standard URDFs they coincide.
            oMj = self.data.oMi[link.parent_joint]
            coms[name] = np.array(oMj.act(pin.SE3.Identity().translation + link.com_local))
        return coms

    def foot_positions(self) -> dict[str, np.ndarray]:
        """World-frame foot positions (call FK first)."""
        return {
            leg: np.array(self.data.oMf[fid].translation)
            for leg, fid in self.foot_frame_ids.items()
        }

    # ------------------------------------------------------------------
    # Cylinder primitives (skeleton-aligned)
    # ------------------------------------------------------------------

    def build_cylinders(self):
        """Assign a CylinderPrimitive to every link (call FK first).

        Leg links get cylinders spanning between centerline-projected joint
        positions.  The base link uses an inertia-only fallback.  Foot links
        share their calf link's radius with a short nominal length.

        This mapping decides which skeleton segment each link belongs to:
          - Side link  → side joint  → thigh joint
          - Thigh link → thigh joint → calf joint
          - Calf link  → calf joint  → foot frame
          - Foot link  → same as calf (thin cap at the foot)
          - Base link   → inertia-only fallback
        """
        # -- Base link (no joint-to-joint segment) --
        base = self.links.get("base_link")
        if base is not None:
            # joint placement in world frame 4x4 transformation (populated by FK)
            oMj = self.data.oMi[base.parent_joint]
            # CoM in world frame = joint translation + rotated local CoM offset
            com_world = np.array(oMj.translation) + np.array(oMj.rotation) @ base.com_local
            # Geometry from inertia alone (no clear axis from joint-to-joint segment since it's the root link)
            base.cylinder = CylinderPrimitive.from_inertia_only(
                base.mass, base.inertia, com_world, np.array(oMj.rotation),
            )

        # -- Leg links --
        for leg in LEG_NAMES:
            proj = self.leg_centerline_positions(leg)

            # Segment endpoints for each link type
            segments = {
                "Side":  (proj["side"],  proj["thigh"]),
                "Thigh": (proj["thigh"], proj["calf"]),
                "Calf":  (proj["calf"],  proj["foot"]),
            }

            for link_type, (p_start, p_end) in segments.items():
                link_name = f"{leg}_{link_type}_link"
                link = self.links.get(link_name)
                if link is None:
                    continue
                R_frame = np.array(self.data.oMf[link.frame_id].rotation)
                link.cylinder = CylinderPrimitive.from_segment(
                    p_start, p_end, link.mass, link.inertia, R_frame,
                )

            # Foot link: thin cylinder at the foot with calf's radius
            foot_name = f"{leg}_Foot_link"
            calf_name = f"{leg}_Calf_link"
            foot_link = self.links.get(foot_name)
            calf_link = self.links.get(calf_name)
            if foot_link is not None and calf_link is not None and calf_link.cylinder is not None:
                foot_pos = proj["foot"]
                foot_link.cylinder = CylinderPrimitive(
                    radius=calf_link.cylinder.radius,
                    length=0.01,  # nominal thin cap
                    center=foot_pos,
                    axis_world=calf_link.cylinder.axis_world,
                    axis_local=calf_link.cylinder.axis_local,
                )

    def compute_jacobian(self, frame_id: int, q: np.ndarray) -> np.ndarray:
        """6×nv world-frame Jacobian for a given frame."""
        return pin.computeFrameJacobian(
            self.model, self.data, q, frame_id, pin.ReferenceFrame.WORLD,
        )

    # ------------------------------------------------------------------
    # Dynamics helpers (thin wrappers for convenience)
    # ------------------------------------------------------------------

    def mass_matrix(self, q: np.ndarray) -> np.ndarray:
        """Joint-space mass matrix M(q)."""
        return pin.crba(self.model, self.data, q)

    def nonlinear_effects(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Coriolis + gravity vector C(q,v)*v + g(q)."""
        return pin.nle(self.model, self.data, q, v)

    def gravity_torque(self, q: np.ndarray) -> np.ndarray:
        """Pure gravity torque g(q)."""
        return pin.computeGeneralizedGravity(self.model, self.data, q)

    # ------------------------------------------------------------------
    # Sagittal-plane (centerline) projection
    # ------------------------------------------------------------------

    def leg_sagittal_plane(self, leg: str) -> tuple[np.ndarray, np.ndarray]:
        """Compute the sagittal plane for a leg (call FK first).

        Parameters
        ----------
        leg : one of LEG_NAMES, e.g. "Front_Left".

        Returns
        -------
        point : (3,) a point on the plane (the side joint position).
        normal : (3,) outward-pointing unit normal (world y direction).
        """
        side_jid = self.model.getJointId(f"{leg}_Side_joint")
        oMj = self.data.oMi[side_jid]
        point = np.array(oMj.translation)
        # Normal is the side joint's local Y-axis rotated into world frame
        normal = np.array(oMj.rotation[:, 1])   # second column = local Y
        normal /= np.linalg.norm(normal)
        return point, normal


    def leg_centerline_positions(
        self, leg: str,
    ) -> dict[str, np.ndarray]:
        """Project leg joint/foot positions onto the leg's sagittal plane.

        Removes the lateral (y) offset so the kinematic chain lies in a
        single plane — useful for 2D trajectory visualization and for
        formulating planar optimal control.

        Call FK first.

        Returns
        -------
        positions : dict with keys "side", "thigh", "calf", "foot",
                    each a (3,) world-frame position projected onto the
                    sagittal plane.
        """
        plane_pt, normal = self.leg_sagittal_plane(leg)

        def _project(pos: np.ndarray) -> np.ndarray:
            offset = pos - plane_pt
            return pos - np.dot(offset, normal) * normal


        side_jid = self.model.getJointId(f"{leg}_Side_joint")
        thigh_jid = self.model.getJointId(f"{leg}_Thigh_joint")
        calf_jid = self.model.getJointId(f"{leg}_Calf_joint")
        foot_fid = self.foot_frame_ids[leg]

        return {
            "side":  _project(np.array(self.data.oMi[side_jid].translation)),
            "thigh": _project(np.array(self.data.oMi[thigh_jid].translation)),
            "calf":  _project(np.array(self.data.oMi[calf_jid].translation)),
            "foot":  _project(np.array(self.data.oMf[foot_fid].translation)),
        }

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def total_mass(self) -> float:
        return sum(link.mass for link in self.links.values())

    def __repr__(self) -> str:
        return (
            f"QuadrupedRobot(links={len(self.links)}, "
            f"nq={self.nq}, nv={self.nv}, "
            f"actuated={self.n_actuated}, "
            f"total_mass={self.total_mass():.4f} kg)"
        )
