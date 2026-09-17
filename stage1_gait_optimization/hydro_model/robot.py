"""Pinocchio robot model with a cylinder approximation of every link.

The cylinders (radius = mesh RMS radius, length from the spec) feed the
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

@dataclass
class CylinderPrimitive:
    """Solid-cylinder approximation of a link.

    ``center_local`` is the midpoint relative to the link frame, in link
    coordinates; set by ``QuadrupedRobot.build_cylinders()``.
    """

    radius: float  # [m] mesh RMS radius, drives drag areas
    length: float  # [m]
    volume_displaced: float  # [m^3] drives buoyancy and added mass
    center: np.ndarray  # midpoint, world frame [m]
    axis_world: np.ndarray  # unit axis, world frame
    axis_local: np.ndarray  # unit axis, link frame
    center_local: np.ndarray = field(default_factory=lambda: np.zeros(3))
    volume_added: float | None = None  # added-mass volume; None -> volume_displaced

    @property
    def volume_entrained(self) -> float:
        """Fluid volume for the added-mass terms.

        Flat links entrain much more than their own volume, so the spec can
        override it.
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
        """Cylinder from ``p_start`` to ``p_end`` (world frame).

        ``R_frame`` is the link's body-to-world rotation at build time.
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


@dataclass
class LinkData:
    """Mass properties and cylinder of one link."""

    name: str
    frame_id: int  # Pinocchio BODY frame
    parent_joint: int  # Pinocchio joint moving this link
    mass: float  # [kg]
    com_local: np.ndarray  # CoM in joint frame [m]
    cylinder: CylinderPrimitive | None = field(default=None, init=False)


class QuadrupedRobot:
    """Pinocchio robot with cylinder-approximated links.

    Parameters
    ----------
    spec : RobotSpec | str | Path
        A spec, its registry name, or its URDF path.
    """

    def __init__(self, spec: "RobotSpec | str | Path"):
        self.spec = get_spec(spec)
        self.urdf_path = Path(self.spec.urdf_path)

        # Free-flyer base: q = [x, y, z, qx, qy, qz, qw, joints...],
        # v = [body-frame linear velocity, body-frame angular velocity, joints...]
        self.model: pin.Model = pin.buildModelFromUrdf(
            str(self.urdf_path), pin.JointModelFreeFlyer()
        )
        self._recenter_base_y()
        self.data: pin.Data = self.model.createData()

        # Displaced volume per link from the collision meshes. Geometry names
        # carry a numeric suffix (e.g. "base_link_0").
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
            # Open meshes fall back to the convex hull (an upper bound). Links
            # made of disjoint parts need RobotSpec.volume_overrides instead.
            self.link_mesh_volumes[link_name] = float(_m.volume if _m.is_watertight else _m.convex_hull.volume)
            self.link_geom_objects[link_name] = go
        self.link_mesh_volumes.update(self.spec.volume_overrides)

        self.links: dict[str, LinkData] = {}
        self._extract_links()

        self.actuated_joint_names: list[str] = list(self.spec.actuated_joint_names)
        missing = [n for n in self.actuated_joint_names if not self.model.existJointName(n)]
        if missing:
            raise ValueError(f"{self.spec.name}: URDF has no joint(s) {missing}")

        self.joint_pin_ids: list[int] = [
            self.model.getJointId(name) for name in self.actuated_joint_names
        ]

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
        # getFrameId does not raise for unknown names.
        if not self.model.existFrame(name):
            raise ValueError(f"{self.spec.name}: URDF has no frame {name!r}")
        return self.model.getFrameId(name)

    # --- Setup ---

    def _recenter_base_y(self):
        """Move the base frame so the base CoM lies at y = 0.

        Fixes the ~4 mm lateral offset of amph's SolidWorks export; child joint
        placements are compensated so their world positions stay unchanged.
        Only enabled via ``spec.recenter_base_y``.
        """
        if not self.spec.recenter_base_y:
            return

        base_jid = 1  # free-flyer joint; holds the base inertia
        dy = self.model.inertias[base_jid].lever[1]
        if abs(dy) < 1e-6:
            return

        for jid in range(2, self.model.njoints):
            if self.model.parents[jid] == base_jid:
                self.model.jointPlacements[jid].translation[1] -= dy

        self.model.inertias[base_jid].lever[1] = 0.0

    def _extract_links(self):
        """Pull mass and CoM from Pinocchio BODY frames."""
        for frame in self.model.frames:
            if frame.type != pin.FrameType.BODY:
                continue
            name = frame.name
            joint_id = frame.parentJoint

            # Pinocchio stores inertia per joint; assumes one link per joint.
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

    # --- Dimensions and kinematics ---

    @property
    def nq(self) -> int:
        """Tree configuration size."""
        return self.model.nq

    @property
    def nv(self) -> int:
        """Tree velocity size."""
        return self.model.nv

    @property
    def n_base_q(self) -> int:
        """Base configuration size (3 position + 4 quaternion)."""
        return 7

    @property
    def n_base_v(self) -> int:
        """Base velocity size (3 linear + 3 angular)."""
        return 6

    @property
    def n_actuated(self) -> int:
        return len(self.actuated_joint_names)

    @property
    def nq_reduced(self) -> int:
        """Configuration size in reduced coordinates (equals ``nq`` for serial robots)."""
        return self.n_base_q + self.n_actuated

    @property
    def nv_reduced(self) -> int:
        """Velocity size in reduced coordinates."""
        return self.n_base_v + self.n_actuated

    def neutral_config(self) -> np.ndarray:
        """Neutral configuration; use instead of ``np.zeros(nq)``, which is not a valid pose."""
        return pin.neutral(self.model)

    def forward_kinematics(
        self,
        q: np.ndarray,
        v: np.ndarray | None = None,
        a: np.ndarray | None = None,
    ) -> pin.Data:
        """Run FK (optionally with velocity and acceleration) and update frame placements."""
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
        """World-frame 4x4 transform per link (call FK first)."""
        poses = {}
        for name, link in self.links.items():
            oMf = self.data.oMf[link.frame_id]
            poses[name] = oMf.homogeneous
        return poses

    def link_world_coms(self) -> dict[str, np.ndarray]:
        """World-frame CoM position for each link (call FK first)."""
        coms = {}
        for name, link in self.links.items():
            oMj = self.data.oMi[link.parent_joint]
            coms[name] = np.array(
                oMj.act(pin.SE3.Identity().translation + link.com_local)
            )
        return coms

    def foot_positions(self) -> dict[str, np.ndarray]:
        """World-frame foot positions (call FK first)."""
        out = {}
        for leg, fid in self.foot_frame_ids.items():
            oMf = self.data.oMf[fid]
            offset = self.foot_offsets[leg]
            p = np.array(oMf.translation)
            if offset.any():
                p = p + np.array(oMf.rotation) @ offset
            out[leg] = p
        return out

    # --- Cylinder primitives ---

    def _mesh_rms_radius(self, link_name: str, center: np.ndarray, axis_world: np.ndarray) -> float:
        """RMS distance of the link's mesh vertices from the given axis (call FK first)."""
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
        """World position of a CylinderSpec endpoint (call FK first).

        A ``str`` is a joint name, or else a frame name. With ``project_leg``
        the point is projected onto that leg's sagittal plane.
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
        """Build a cylinder for every link in ``spec.cylinders`` (call FK first).

        Specs are processed in order, so a ``copy`` must come after its ``ref``.
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

                # Solid cylinder: I_axial = m*r^2/2, I_transverse = m*(3*r^2 + h^2)/12
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
        """Store the cylinder midpoint in link coordinates for the symbolic model."""
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

    # --- Numeric dynamics ---

    def mass_matrix(self, q: np.ndarray) -> np.ndarray:
        """Joint-space mass matrix M(q)."""
        return pin.crba(self.model, self.data, q)

    def nonlinear_effects(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Coriolis + gravity vector C(q,v)*v + g(q)."""
        return pin.nle(self.model, self.data, q, v)

    def gravity_torque(self, q: np.ndarray) -> np.ndarray:
        """Pure gravity torque g(q)."""
        return pin.computeGeneralizedGravity(self.model, self.data, q)

    # --- Leg planes ---

    def leg_sagittal_plane(self, leg: str) -> tuple[np.ndarray, np.ndarray]:
        """``(point, unit normal)`` of a leg's sagittal plane (call FK first).

        Defined by the origin and local y-axis of ``spec.leg_plane_joint[leg]``.
        """
        jname = self.spec.leg_plane_joint.get(leg)
        if jname is None:
            raise ValueError(
                f"{self.spec.name}: no leg_plane_joint for {leg!r}; a robot whose "
                f"legs are already planar should leave CylinderSpec.project_leg unset"
            )
        oMj = self.data.oMi[self.model.getJointId(jname)]
        point = np.array(oMj.translation)
        normal = np.array(oMj.rotation[:, 1])  # local y-axis
        normal /= np.linalg.norm(normal)
        return point, normal

    def leg_skeleton(self, leg: str) -> list[tuple[np.ndarray, np.ndarray]]:
        """World-frame segments of one leg's ``segment`` cylinders (call FK first).

        Returns separate segments so closed-chain legs can be drawn too.
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
        """Side, thigh, calf and foot positions projected onto the leg plane (call FK first).

        Only for amph-style serial legs.
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
