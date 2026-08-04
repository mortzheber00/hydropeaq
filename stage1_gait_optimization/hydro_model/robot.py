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

from .coordinate_map import IdentityMap
from .robots import LocalPoint, RobotSpec, get_spec

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
    volume_added: float | None = None
    # fluid volume driving added mass; None -> volume_displaced

    @property
    def volume_entrained(self) -> float:
        """Fluid volume the added-mass terms act on.

        Defaults to the displaced volume, which is right whenever the cylinder
        actually fits the link.  It is *not* right for a flat link: a plate
        entrains the fluid in the cylinder it sweeps, which is several times
        its own volume, so such links override this via the spec.
        """
        return self.volume_displaced if self.volume_added is None else self.volume_added

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
# Main robot class
# ---------------------------------------------------------------------------


class QuadrupedRobot:
    """Quadruped robot backed by Pinocchio with cylinder-approximated links.

    Parameters
    ----------
    spec : RobotSpec | str | Path
        A registered robot, its registry name, or the path to its URDF (the
        path form keeps pre-registry call sites working).
    """

    def __init__(self, spec: "RobotSpec | str | Path"):
        self.spec = get_spec(spec)
        self.urdf_path = Path(self.spec.urdf_path)

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
            # convex hull so the displaced volume is a sensible upper bound.  That
            # bound is only useful for a link that is sealed and roughly fills its
            # hull; one built from disjoint parts with open water between them needs
            # RobotSpec.volume_overrides instead.
            self.link_mesh_volumes[link_name] = float(_m.volume if _m.is_watertight else _m.convex_hull.volume)
            self.link_geom_objects[link_name] = go
        self.link_mesh_volumes.update(self.spec.volume_overrides)

        # Extract per-link data from Pinocchio frames
        self.links: dict[str, LinkData] = {}
        self._extract_links()

        # Ordered actuated joint names, straight from the spec.
        self.actuated_joint_names: list[str] = list(self.spec.actuated_joint_names)
        missing = [n for n in self.actuated_joint_names if not self.model.existJointName(n)]
        if missing:
            raise ValueError(f"{self.spec.name}: URDF has no joint(s) {missing}")

        # Map from our ordered joint index to Pinocchio joint index
        self.joint_pin_ids: list[int] = [
            self.model.getJointId(name) for name in self.actuated_joint_names
        ]

        # Foot frames.  getFrameId returns nframes for an unknown name instead
        # of raising, which silently produces an out-of-range lookup later.
        self.foot_frame_ids: dict[str, int] = {}
        self.foot_offsets: dict[str, np.ndarray] = {}
        for leg, (frame_name, offset) in self.spec.foot_points.items():
            self.foot_frame_ids[leg] = self._frame_id(frame_name)
            self.foot_offsets[leg] = np.asarray(offset, dtype=float)

        self.coord_map = (
            IdentityMap(self.n_actuated) if self.spec.coordinate_map is None
            else self.spec.coordinate_map(self)
        )

    def _frame_id(self, name: str) -> int:
        if not self.model.existFrame(name):
            raise ValueError(f"{self.spec.name}: URDF has no frame {name!r}")
        return self.model.getFrameId(name)

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
        # Opt-in per robot: it assumes every base child is a hip and that the
        # base CoM offset is a millimetre-scale export artefact.  Neither holds
        # for a robot whose exported base frame sits far from the body.
        if not self.spec.recenter_base_y:
            return

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

    @property
    def nq_reduced(self) -> int:
        """Configuration size in reduced coordinates (7 base + n_actuated).

        Equal to ``nq`` for a serial robot.  For a closed-chain robot the tree
        is larger, and larger still when it contains continuous joints, whose
        configuration is a ``(cos, sin)`` pair.
        """
        return self.n_base_q + self.n_actuated

    @property
    def nv_reduced(self) -> int:
        """Velocity size in reduced coordinates (6 base + n_actuated)."""
        return self.n_base_v + self.n_actuated

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
        """World-frame foot positions (call FK first).

        The foot is a fixed offset in some frame's local coordinates; robots
        with a dedicated ``*_Foot_link`` use a zero offset, so this reduces to
        the frame origin exactly.
        """
        out = {}
        for leg, fid in self.foot_frame_ids.items():
            oMf = self.data.oMf[fid]
            offset = self.foot_offsets[leg]
            p = np.array(oMf.translation)
            if offset.any():
                p = p + np.array(oMf.rotation) @ offset
            out[leg] = p
        return out

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

    def _resolve_point(self, endpoint, project_leg: str | None) -> np.ndarray:
        """World-frame position of a CylinderSpec endpoint (call FK first).

        ``str`` names a joint (its origin) or, failing that, a frame.
        ``LocalPoint`` names a point in a link's local body frame.
        With ``project_leg`` the result is projected onto that leg's sagittal
        plane, which is what removes the lateral offset of an asymmetric export.
        """
        if isinstance(endpoint, LocalPoint):
            oMf = self.data.oMf[self._frame_id(endpoint.link)]
            p = np.array(oMf.translation) + np.array(oMf.rotation) @ np.asarray(
                endpoint.xyz, dtype=float
            )
        elif self.model.existJointName(endpoint):
            p = np.array(self.data.oMi[self.model.getJointId(endpoint)].translation)
        elif self.model.existFrame(endpoint):
            p = np.array(self.data.oMf[self.model.getFrameId(endpoint)].translation)
        else:
            raise ValueError(
                f"{self.spec.name}: cylinder endpoint {endpoint!r} is neither a joint nor a frame"
            )

        if project_leg is not None:
            plane_pt, normal = self.leg_sagittal_plane(project_leg)
            p = p - np.dot(p - plane_pt, normal) * normal
        return p

    def build_cylinders(self):
        """Assign a CylinderPrimitive to every link named in the spec.

        Call FK first.  Each ``CylinderSpec`` is one of three kinds:
          - ``segment``  axis and length span two resolved endpoints
          - ``copy``     thin cap at a point, reusing another link's axis
          - ``inertia``  axis and length from the inertia eigenvectors

        Specs are processed in order, so a ``copy`` may reference any link
        built before it.
        """
        for cs in self.spec.cylinders:
            link = self.links.get(cs.link)
            if link is None:
                raise ValueError(
                    f"{self.spec.name}: cylinder spec names link {cs.link!r}, "
                    f"which has no mass-carrying frame in the model"
                )
            volume = self.link_mesh_volumes.get(cs.link, 0.0)

            if cs.kind == "inertia":
                oMj = self.data.oMi[link.parent_joint]
                R_world = np.array(oMj.rotation)
                center = np.array(oMj.translation) + R_world @ link.com_local

                # For a solid cylinder: I_axial = m*r^2/2,
                # I_transverse = m*(3*r^2 + h^2)/12.
                inertia = np.array(self.model.inertias[link.parent_joint].inertia)
                eigvals, eigvecs = np.linalg.eigh(inertia)
                idx_min = int(np.argmin(eigvals))
                I_sym = eigvals[idx_min]
                I_trans = float(np.mean([eigvals[i] for i in range(3) if i != idx_min]))
                r_sq = 2.0 * I_sym / link.mass
                h_sq = 12.0 * I_trans / link.mass - 3.0 * r_sq
                axis_local = eigvecs[:, idx_min]
                axis_world = R_world @ axis_local
                link.cylinder = CylinderPrimitive(
                    radius=self._mesh_rms_radius(cs.link, center, axis_world),
                    length=np.sqrt(max(h_sq, 1e-10)),
                    volume_displaced=volume,
                    center=center,
                    axis_world=axis_world,
                    axis_local=axis_local,
                )

            elif cs.kind == "segment":
                p_start = self._resolve_point(cs.start, cs.project_leg)
                p_end = self._resolve_point(cs.end, cs.project_leg)
                diff = p_end - p_start
                axis_world = diff / max(float(np.linalg.norm(diff)), 1e-6)
                center = (p_start + p_end) / 2.0
                link.cylinder = CylinderPrimitive.from_segment(
                    p_start, p_end,
                    self._mesh_rms_radius(cs.link, center, axis_world),
                    np.array(self.data.oMf[link.frame_id].rotation),
                    volume_displaced=volume,
                )

            elif cs.kind == "copy":
                ref = self.links[cs.ref].cylinder
                if ref is None:
                    raise ValueError(
                        f"{self.spec.name}: {cs.link!r} copies {cs.ref!r}, "
                        f"which is not built yet — reorder spec.cylinders"
                    )
                center = self._resolve_point(cs.at, cs.project_leg)
                link.cylinder = CylinderPrimitive(
                    radius=self._mesh_rms_radius(cs.link, center, ref.axis_world),
                    length=cs.length,
                    volume_displaced=volume,
                    center=center,
                    axis_world=ref.axis_world,
                    axis_local=ref.axis_local,
                )

            else:
                raise ValueError(f"unknown CylinderSpec kind {cs.kind!r}")

            if cs.radius is not None:
                link.cylinder.radius = cs.radius
            if cs.added_mass_volume is not None:
                link.cylinder.volume_added = cs.added_mass_volume

            self._set_center_local(link)

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
        jname = self.spec.leg_plane_joint.get(leg)
        if jname is None:
            raise ValueError(
                f"{self.spec.name}: no leg_plane_joint for {leg!r}; a robot whose "
                f"legs are already planar should leave CylinderSpec.project_leg unset"
            )
        oMj = self.data.oMi[self.model.getJointId(jname)]
        point = np.array(oMj.translation)
        # Normal is the plane joint's local Y-axis rotated into world frame
        normal = np.array(oMj.rotation[:, 1])  # second column = local Y
        normal /= np.linalg.norm(normal)
        return point, normal

    def leg_skeleton(self, leg: str) -> list[tuple[np.ndarray, np.ndarray]]:
        """World-frame segments drawing one leg (call FK first).

        The skeleton *is* the cylinder chain, so this introduces no geometry of
        its own.  A serial leg yields one connected run; a closed-chain leg
        yields both sub-chains, which is why the return type is a list of
        segments rather than a single polyline.  Zero-length caps (a ``copy``
        cylinder such as amph's foot) are omitted — use ``foot_positions`` for
        those.
        """
        out = []
        for cs in self.spec.cylinders:
            if cs.kind != "segment" or leg not in cs.link:
                continue
            out.append((self._resolve_point(cs.start, cs.project_leg),
                        self._resolve_point(cs.end, cs.project_leg)))
        return out

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
