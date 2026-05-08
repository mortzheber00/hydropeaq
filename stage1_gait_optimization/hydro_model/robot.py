"""
Quadruped robot model built on Pinocchio with cylinder-primitive approximations.

Pinocchio handles URDF parsing, kinematic tree, forward kinematics, and
Jacobians.  On top of that, each link is approximated as a solid cylinder
(radius = mesh RMS radius, length = joint-to-joint distance) for the
hydrodynamic model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pinocchio as pin
import trimesh

# ---------------------------------------------------------------------------
# Geometric primitive
# ---------------------------------------------------------------------------


@dataclass
class CylinderPrimitive:
    """Solid-cylinder approximation of a robot link.

    For leg links the cylinder spans between two skeleton joints
    (centerline-projected).  The radius is the RMS perpendicular distance of
    mesh vertices from the cylinder axis.

    ``center_local`` is the offset of the cylinder midpoint from the link's
    BODY-frame origin, expressed in the link LOCAL frame.  It is populated by
    ``QuadrupedRobot.build_cylinders()`` and used by the CasADi symbolic
    model to compute the correct world-frame cylinder centre from FK:

        p_center_world = R_link @ center_local + t_link_world
    """

    radius: float  # [m] — mesh RMS radius, drives drag areas
    length: float  # [m]
    volume_displaced: float  # [m^3] actual displaced water — drives buoyancy and added mass
    center: np.ndarray  # midpoint of the cylinder in world frame [m]
    axis_world: np.ndarray  # unit vector along the cylinder axis (world frame)
    axis_local: np.ndarray  # same axis expressed in the link body frame
    center_local: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # offset of cylinder midpoint from link frame origin, in link LOCAL frame

    @property
    def cross_section_axial(self) -> float:
        """End-cap area (flow along the cylinder axis)."""
        return np.pi * self.radius**2

    @property
    def cross_section_transverse(self) -> float:
        """Projected area for flow perpendicular to the cylinder axis."""
        return 2.0 * self.radius * self.length

    @classmethod
    def from_segment(
        cls,
        p_start: np.ndarray,
        p_end: np.ndarray,
        radius: float,
        R_frame: np.ndarray,
        volume_displaced: float,
    ) -> CylinderPrimitive:
        """Build a cylinder spanning from p_start to p_end (world frame).

        Length = distance between the two points.
        radius: mesh RMS radius (pre-computed by the caller).
        volume_displaced: actual mesh volume used for buoyancy and added mass.
        R_frame : 3x3 rotation of the link frame at build time (body to world frame).
        """
        diff = p_end - p_start
        length = float(np.linalg.norm(diff))
        length = max(length, 1e-6)
        axis_world = diff / length
        axis_local = R_frame.T @ axis_world
        center = (p_start + p_end) / 2.0
        return cls(
            radius=radius,
            length=length,
            volume_displaced=volume_displaced,
            center=center,
            axis_world=axis_world,
            axis_local=axis_local,
        )


# ---------------------------------------------------------------------------
# Per-link data combining Pinocchio kinematics with cylinder geometry
# ---------------------------------------------------------------------------


@dataclass
class LinkData:
    """Physical properties of a single link, derived from Pinocchio."""

    name: str
    frame_id: int  # Pinocchio frame index (BODY type)
    parent_joint: int  # Pinocchio joint index that moves this link
    mass: float  # [kg]
    com_local: np.ndarray  # center of mass in joint frame [m]
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

        # Build Pinocchio model with a free-floating base so the robot body
        # can translate and rotate in the world frame.
        # Configuration layout: q = [x, y, z, qx, qy, qz, qw, joint_angles...]
        #   nq = 7 (base) + 12 (joints) = 19
        #   nv = 6 (base twist) + 12 (joints) = 18
        # The base velocity v[0:3] is the linear velocity in the LOCAL (body)
        # frame, and v[3:6] is the angular velocity in the LOCAL frame.
        self.model: pin.Model = pin.buildModelFromUrdf(
            str(self.urdf_path), pin.JointModelFreeFlyer()
        )
        self._recenter_base_y()
        self.data: pin.Data = self.model.createData()

        # Build mesh-volume map: link_name -> displaced volume [m^3] from STL meshes.
        # Geometry object names have a numeric suffix (e.g. "base_link_0"); strip it.
        _geom_model = pin.GeometryModel()
        pin.buildGeomFromUrdf(
            self.model,
            str(self.urdf_path),
            pin.GeometryType.COLLISION,
            _geom_model,
            [str(self.urdf_path.parent.parent)],
        )
        self.link_mesh_volumes: dict[str, float] = {}
        self.link_geom_objects: dict[str, pin.GeometryObject] = {}
        for go in _geom_model.geometryObjects:
            link_name = go.name.rsplit("_", 1)[0]
            _m = trimesh.load(go.meshPath)
            # Non-watertight meshes (e.g. base_link has an open surface) fall back to
            # convex hull so the displaced volume is a sensible upper bound.
            self.link_mesh_volumes[link_name] = float(_m.volume if _m.is_watertight else _m.convex_hull.volume)
            self.link_geom_objects[link_name] = go

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

    def _recenter_base_y(self):
        """Shift the base frame origin so the base CoM lies at y = 0.

        The SolidWorks URDF export places the base frame origin ~4 mm off
        the geometric centerline in y.  This makes left/right side-joint
        origins asymmetric (±0.057 vs ±0.065) even though the physical
        robot is symmetric.

        Fix: move the base frame by dy (the base CoM y-offset) and
        compensate every base-child joint placement by -dy so that all
        joints remain at their original world-frame positions.
        """
        # Joint 0 is the free-flyer "universe → base" virtual joint.
        # Joint 1 is the first real joint (root_joint in Pinocchio).
        # Base inertia is stored at joint index 1 for a free-flyer model.
        base_jid = 1
        dy = self.model.inertias[base_jid].lever[1]
        if abs(dy) < 1e-6:
            return

        # Shift all joints whose parent is the base (the 4 side joints)
        for jid in range(2, self.model.njoints):
            if self.model.parents[jid] == base_jid:
                self.model.jointPlacements[jid].translation[1] -= dy

        # Zero out the base CoM y-offset
        self.model.inertias[base_jid].lever[1] = 0.0

    def _extract_links(self):
        """Pull mass and CoM from Pinocchio BODY frames."""
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

            self.links[name] = LinkData(
                name=name,
                frame_id=self.model.getFrameId(name),
                parent_joint=joint_id,
                mass=mass,
                com_local=np.array(inertia_pin.lever),
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
        """Number of configuration variables (19 = 7 base + 12 joints)."""
        return self.model.nq

    @property
    def nv(self) -> int:
        """Number of velocity variables (18 = 6 base + 12 joints)."""
        return self.model.nv

    @property
    def n_base_q(self) -> int:
        """Configuration variables for the floating base (3 pos + 4 quat = 7)."""
        return 7

    @property
    def n_base_v(self) -> int:
        """Velocity variables for the floating base (3 lin + 3 ang = 6)."""
        return 6

    @property
    def n_actuated(self) -> int:
        return len(self.actuated_joint_names)

    def neutral_config(self) -> np.ndarray:
        """Return the neutral configuration.

        The floating base is placed at the origin with identity orientation
        (quaternion w=1), and all joint angles are zero.  Always prefer this
        over ``np.zeros(robot.nq)`` — a zero quaternion is not a valid rotation.
        """
        return pin.neutral(self.model)

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
            # CoM is stored relative to the joint frame; the BODY frame
            # may have an offset, but for standard URDFs they coincide.
            oMj = self.data.oMi[link.parent_joint]
            coms[name] = np.array(
                oMj.act(pin.SE3.Identity().translation + link.com_local)
            )
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

    def _mesh_rms_radius(self, link_name: str, center: np.ndarray, axis_world: np.ndarray) -> float:
        """RMS perpendicular distance of mesh vertices from the cylinder axis.

        Vertices are transformed to world frame using the current FK and the
        geometry object's placement, then projected perpendicular to axis_world.
        """
        go = self.link_geom_objects.get(link_name)
        if go is None:
            raise ValueError(f"No geometry object for link {link_name!r}")

        oMj = self.data.oMi[go.parentJoint]
        T = oMj * go.placement  # geometry local frame → world
        R = np.array(T.rotation)
        t = np.array(T.translation)

        verts_local = np.array([go.geometry.vertex(i) for i in range(go.geometry.num_vertices)])
        verts_world = verts_local @ R.T + t  # (N, 3)

        v_rel = verts_world - center[None, :]  # (N, 3)
        perp = v_rel - (v_rel @ axis_world)[:, None] * axis_world[None, :]  # (N, 3)
        return float(np.sqrt(np.mean(np.sum(perp**2, axis=1))))

    def build_cylinders(self):
        """Assign a CylinderPrimitive to every link (call FK first).

        Leg links get cylinders spanning between centerline-projected joint
        positions.  The base link uses inertia eigenvectors for axis/length.
        Foot links share their calf link's axis with a short nominal length.

        This mapping decides which skeleton segment each link belongs to:
          - Side link  → side joint  → thigh joint
          - Thigh link → thigh joint → calf joint
          - Calf link  → calf joint  → foot frame
          - Foot link  → same as calf (thin cap at the foot)
          - Base link   → inertia eigenvectors for axis/length
        """
        # -- Base link (no joint-to-joint segment) --
        base = self.links.get("base_link")
        if base is not None:
            oMj = self.data.oMi[base.parent_joint]
            R_world = np.array(oMj.rotation)
            com_world = np.array(oMj.translation) + R_world @ base.com_local

            # Axis and length from inertia eigenvectors (no joint-to-joint segment for the root link).
            # For a solid cylinder: I_axial = m*r^2/2, I_transverse = m*(3*r^2 + h^2)/12.
            inertia = np.array(self.model.inertias[base.parent_joint].inertia)
            eigvals, eigvecs = np.linalg.eigh(inertia)
            idx_min = int(np.argmin(eigvals))
            I_sym = eigvals[idx_min]
            I_trans = float(np.mean([eigvals[i] for i in range(3) if i != idx_min]))
            r_sq = 2.0 * I_sym / base.mass
            h_sq = 12.0 * I_trans / base.mass - 3.0 * r_sq
            length = np.sqrt(max(h_sq, 1e-10))
            axis_local = eigvecs[:, idx_min]
            axis_world = R_world @ axis_local

            r_rms = self._mesh_rms_radius("base_link", com_world, axis_world)
            base.cylinder = CylinderPrimitive(
                radius=r_rms,
                length=length,
                volume_displaced=self.link_mesh_volumes.get("base_link", 0.0),
                center=com_world,
                axis_world=axis_world,
                axis_local=axis_local,
            )
            self._set_center_local(base)

        # -- Leg links --
        for leg in LEG_NAMES:
            proj = self.leg_centerline_positions(leg)

            segments = {
                "Side": (proj["side"], proj["thigh"]),
                "Thigh": (proj["thigh"], proj["calf"]),
                "Calf": (proj["calf"], proj["foot"]),
            }

            for link_type, (p_start, p_end) in segments.items():
                link_name = f"{leg}_{link_type}_link"
                link = self.links.get(link_name)
                if link is None:
                    continue
                diff = p_end - p_start
                axis_world = diff / max(float(np.linalg.norm(diff)), 1e-6)
                center = (p_start + p_end) / 2.0
                r_rms = self._mesh_rms_radius(link_name, center, axis_world)
                R_frame = np.array(self.data.oMf[link.frame_id].rotation)
                link.cylinder = CylinderPrimitive.from_segment(
                    p_start, p_end, r_rms, R_frame,
                    volume_displaced=self.link_mesh_volumes.get(link_name, 0.0),
                )
                self._set_center_local(link)

            # Foot link: thin cylinder at the foot with calf's radius
            foot_name = f"{leg}_Foot_link"
            calf_name = f"{leg}_Calf_link"
            foot_link = self.links.get(foot_name)
            calf_link = self.links.get(calf_name)
            if (
                foot_link is not None
                and calf_link is not None
                and calf_link.cylinder is not None
            ):
                foot_pos = proj["foot"]
                foot_vol = self.link_mesh_volumes.get(foot_name, 0.0)
                foot_length = 0.01  # nominal thin cap
                r_rms = self._mesh_rms_radius(foot_name, foot_pos, calf_link.cylinder.axis_world)
                foot_link.cylinder = CylinderPrimitive(
                    radius=r_rms,
                    length=foot_length,
                    volume_displaced=foot_vol,
                    center=foot_pos,
                    axis_world=calf_link.cylinder.axis_world,
                    axis_local=calf_link.cylinder.axis_local,
                )
                self._set_center_local(foot_link)

    def _set_center_local(self, link: "LinkData") -> None:
        """Compute cylinder.center_local from the current FK placement.

        center_local = R_frame^T @ (center_world - frame_origin_world)

        This stores the cylinder midpoint as a fixed offset in the link's
        LOCAL body frame so the symbolic model can reconstruct the correct
        world-frame position via  p = R_sym @ center_local + t_sym.
        """
        oMf = self.data.oMf[link.frame_id]
        R = np.array(oMf.rotation)  # world_R_local
        t = np.array(oMf.translation)  # frame origin in world
        link.cylinder.center_local = R.T @ (link.cylinder.center - t)

    def compute_jacobian(self, frame_id: int, q: np.ndarray) -> np.ndarray:
        """6×nv world-frame Jacobian for a given frame."""
        return pin.computeFrameJacobian(
            self.model,
            self.data,
            q,
            frame_id,
            pin.ReferenceFrame.WORLD,
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
        normal = np.array(oMj.rotation[:, 1])  # second column = local Y
        normal /= np.linalg.norm(normal)
        return point, normal

    def leg_centerline_positions(
        self,
        leg: str,
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
            "side": _project(np.array(self.data.oMi[side_jid].translation)),
            "thigh": _project(np.array(self.data.oMi[thigh_jid].translation)),
            "calf": _project(np.array(self.data.oMi[calf_jid].translation)),
            "foot": _project(np.array(self.data.oMf[foot_fid].translation)),
        }

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def total_mass(self) -> float:
        """Total robot mass [kg] from the Pinocchio model."""
        return pin.computeTotalMass(self.model)

    def __repr__(self) -> str:
        return (
            f"QuadrupedRobot(links={len(self.links)}, "
            f"nq={self.nq} (base_q={self.n_base_q} + joints={self.n_actuated}), "
            f"nv={self.nv} (base_v={self.n_base_v} + joints={self.n_actuated}), "
            f"total_mass={self.total_mass():.4f} kg)"
        )
