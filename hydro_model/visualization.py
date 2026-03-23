"""
3D visualization of the cylinder-approximated quadruped robot.

Uses matplotlib for a static 3D view showing each link as a cylinder
at the pose computed by Pinocchio forward kinematics.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from .robot import QuadrupedRobot


def _cylinder_mesh(
    radius: float,
    length: float,
    n_facets: int = 16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unit cylinder mesh centered at origin, aligned along z-axis."""
    theta = np.linspace(0, 2 * np.pi, n_facets + 1)
    z = np.linspace(-length / 2, length / 2, 2)
    theta, z = np.meshgrid(theta, z)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return x, y, z


def _rotation_align_z_to(target: np.ndarray) -> np.ndarray:
    """Rotation matrix that maps [0,0,1] to the given target direction."""
    target = target / (np.linalg.norm(target) + 1e-15)
    z_axis = np.array([0.0, 0.0, 1.0])

    if np.allclose(target, z_axis):
        return np.eye(3)
    if np.allclose(target, -z_axis):
        return np.diag([1.0, -1.0, -1.0])

    v = np.cross(z_axis, target)
    s = np.linalg.norm(v)
    c = np.dot(z_axis, target)
    vx = np.array([
        [    0, -v[2],  v[1]],
        [ v[2],     0, -v[0]],
        [-v[1],  v[0],     0],
    ])
    return np.eye(3) + vx + vx @ vx * (1 - c) / (s * s + 1e-15)


# Color palette for different link types
LINK_COLORS = {
    "base":  "#4A90D9",
    "side":  "#E8A838",
    "thigh": "#5CB85C",
    "calf":  "#D9534F",
    "foot":  "#9B59B6",
}


def _link_color(name: str) -> str:
    name_lower = name.lower()
    for key, color in LINK_COLORS.items():
        if key in name_lower:
            return color
    return "#888888"


def _set_equal_aspect(ax, points: np.ndarray, margin: float = 1.2):
    """Set equal aspect ratio on a 3D axis from a point cloud."""
    mid = points.mean(axis=0)
    span = (points.max(axis=0) - points.min(axis=0)).max() / 2 * margin
    ax.set_xlim(mid[0] - span, mid[0] + span)
    ax.set_ylim(mid[1] - span, mid[1] + span)
    ax.set_zlim(mid[2] - span, mid[2] + span)


def _draw_frame_axes(ax, origin: np.ndarray, R: np.ndarray, length: float = 0.01):
    """Draw small RGB axes at a frame origin."""
    for i, color in enumerate(["r", "g", "b"]):
        tip = origin + R[:, i] * length
        ax.plot(*zip(origin, tip), color=color, linewidth=1.5, alpha=0.8)


def visualize_skeleton(
    robot: QuadrupedRobot,
    q: np.ndarray | None = None,
    centerline: bool = True,
    title: str = "Robot Kinematic Skeleton",
    elev: float = 25.0,
    azim: float = -60.0,
    figsize: tuple[float, float] = (12, 9),
    save_path: str | None = None,
) -> plt.Figure:
    """Render the kinematic skeleton: joints as nodes, links as edges.

    Parameters
    ----------
    robot : QuadrupedRobot
        Pinocchio-backed robot model.
    q : joint angles (defaults to zeros).
    centerline : if True (default), project each leg's thigh/calf/foot
        onto the leg's sagittal plane so the chain appears planar.
        The raw (offset) positions are shown as faint ghost markers.
    title : plot title.
    elev, azim : camera angles.
    figsize : figure size.
    save_path : if given, save figure to this path.
    """
    from .robot import LEG_NAMES

    if q is None:
        q = np.zeros(robot.nq)

    robot.forward_kinematics(q)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    all_pts = []

    # -- Base (universe) joint --
    base_pos = np.array(robot.data.oMi[0].translation)
    base_R = np.array(robot.data.oMi[0].rotation)
    all_pts.append(base_pos)
    ax.scatter(*base_pos, s=50, c="k", zorder=5)
    _draw_frame_axes(ax, base_pos, base_R, length=0.015)
    ax.text(base_pos[0], base_pos[1], base_pos[2] + 0.005,
            "base", fontsize=5, ha="center", va="bottom", color="dimgray")

    # -- Draw each leg --
    for leg in LEG_NAMES:
        # Raw joint positions from Pinocchio
        side_jid = robot.model.getJointId(f"{leg}_Side_joint")
        thigh_jid = robot.model.getJointId(f"{leg}_Thigh_joint")
        calf_jid = robot.model.getJointId(f"{leg}_Calf_joint")
        foot_fid = robot.foot_frame_ids[leg]

        raw = {
            "side":  np.array(robot.data.oMi[side_jid].translation),
            "thigh": np.array(robot.data.oMi[thigh_jid].translation),
            "calf":  np.array(robot.data.oMi[calf_jid].translation),
            "foot":  np.array(robot.data.oMf[foot_fid].translation),
        }

        if centerline:
            proj = robot.leg_centerline_positions(leg)
            # Show raw positions as faint ghost markers
            for key in ["thigh", "calf", "foot"]:
                ax.scatter(*raw[key], s=12, c="gray", alpha=0.3, zorder=2)
                # Dashed line from ghost to projected
                ax.plot(*zip(raw[key], proj[key]), color="gray",
                        linewidth=0.5, linestyle=":", alpha=0.4)
            pts = proj
        else:
            pts = raw

        # Joint markers and frame axes
        chain = ["side", "thigh", "calf"]
        jids = [side_jid, thigh_jid, calf_jid]
        for key, jid in zip(chain, jids):
            pos = pts[key]
            R = np.array(robot.data.oMi[jid].rotation)
            all_pts.append(pos)
            ax.scatter(*pos, s=40, c="k", zorder=5)
            _draw_frame_axes(ax, pos, R, length=0.012)

        # Foot marker
        foot_pos = pts["foot"]
        foot_R = np.array(robot.data.oMf[foot_fid].rotation)
        all_pts.append(foot_pos)
        ax.scatter(*foot_pos, s=60, c="#9B59B6", marker="v",
                   edgecolors="k", linewidths=0.5, zorder=5)
        _draw_frame_axes(ax, foot_pos, foot_R, length=0.015)

        # Connecting lines: base→side→thigh→calf, calf--foot
        ax.plot(*zip(base_pos, pts["side"]),
                color=_link_color("side"), linewidth=2.5, alpha=0.8)
        ax.plot(*zip(pts["side"], pts["thigh"]),
                color=_link_color("side"), linewidth=2.5, alpha=0.8)
        ax.plot(*zip(pts["thigh"], pts["calf"]),
                color=_link_color("thigh"), linewidth=2.5, alpha=0.8)
        ax.plot(*zip(pts["calf"], pts["foot"]),
                color=_link_color("calf"), linewidth=2.0, linestyle="--", alpha=0.7)

        # Labels
        for key in ["side", "thigh", "calf"]:
            pos = pts[key]
            label = f"{leg}\n{key}".replace("_", "\n")
            ax.text(pos[0], pos[1], pos[2] + 0.005, label,
                    fontsize=4, ha="center", va="bottom", color="dimgray")

    all_pts = np.array(all_pts)
    _set_equal_aspect(ax, all_pts)

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)

    legend_elements = [
        Patch(facecolor=c, edgecolor="k", label=n.capitalize())
        for n, c in LINK_COLORS.items()
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=8)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def visualize_robot(
    robot: QuadrupedRobot,
    q: np.ndarray | None = None,
    title: str = "Cylinder-Approximated Robot",
    elev: float = 25.0,
    azim: float = -60.0,
    figsize: tuple[float, float] = (12, 9),
    save_path: str | None = None,
) -> plt.Figure:
    """Render the robot with each link drawn as a skeleton-aligned cylinder.

    Parameters
    ----------
    robot : QuadrupedRobot
        Pinocchio-backed robot model.
    q : joint angles (defaults to zeros).
    title : plot title.
    elev, azim : camera angles.
    figsize : figure size.
    save_path : if given, save figure to this path.
    """
    if q is None:
        q = np.zeros(robot.nq)

    robot.forward_kinematics(q)
    robot.build_cylinders()

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    all_pts = []

    for name, link in robot.links.items():
        cyl = link.cylinder
        if cyl is None:
            continue

        # Cylinder mesh is z-aligned; rotate z → axis_world
        R_cyl = _rotation_align_z_to(cyl.axis_world)

        X, Y, Z = _cylinder_mesh(cyl.radius, cyl.length)
        color = _link_color(name)

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                pt = R_cyl @ np.array([X[i, j], Y[i, j], Z[i, j]]) + cyl.center
                X[i, j], Y[i, j], Z[i, j] = pt
                all_pts.append(pt)

        ax.plot_surface(X, Y, Z, alpha=0.6, color=color, edgecolor="k",
                        linewidth=0.3)

        ax.text(cyl.center[0], cyl.center[1], cyl.center[2],
                name.replace("_link", "").replace("_", "\n"),
                fontsize=5, ha="center", va="bottom", color="k")

    # Draw centerline skeleton on top
    from .robot import LEG_NAMES
    base_pos = np.array(robot.data.oMi[0].translation)
    for leg in LEG_NAMES:
        proj = robot.leg_centerline_positions(leg)
        chain = [base_pos, proj["side"], proj["thigh"], proj["calf"], proj["foot"]]
        for a, b in zip(chain[:-1], chain[1:]):
            ax.plot(*zip(a, b), "k-", linewidth=0.8, alpha=0.5)

    all_pts = np.array(all_pts)
    _set_equal_aspect(ax, all_pts)

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)

    legend_elements = [
        Patch(facecolor=c, edgecolor="k", label=n.capitalize())
        for n, c in LINK_COLORS.items()
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=8)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
