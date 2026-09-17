"""Matplotlib 3D views of the robot skeleton and its cylinder approximation."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

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
    vx = np.array(
        [
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ]
    )
    return np.eye(3) + vx + vx @ vx * (1 - c) / (s * s + 1e-15)


LINK_COLORS = {
    "base": "#4A90D9",
    "side": "#E8A838",
    "thigh": "#5CB85C",
    "calf": "#D9534F",
    "foot": "#9B59B6",
}


# Cycled along a leg's cylinder chain; excludes the base colour.
_SEGMENT_COLORS = [LINK_COLORS["side"], LINK_COLORS["thigh"], LINK_COLORS["calf"],
                   LINK_COLORS["foot"], "#E17FB0", "#7FD4C1"]


def link_color_key(robot: QuadrupedRobot) -> tuple[dict[str, str], dict[str, str]]:
    """``({link name: colour}, {legend label: colour})`` for one robot.

    Links named like amph's (side, thigh, ...) use ``LINK_COLORS``; others are
    coloured by chain position and labelled by their name suffix (BODY2: 1.1 ...).
    """
    base = robot.spec.base_link
    colors = {base: LINK_COLORS["base"]}
    legend = {"Base": LINK_COLORS["base"]}
    for leg in robot.spec.leg_names:
        links = [cs.link for cs in robot.spec.cylinders if leg in cs.link]
        for idx, name in enumerate(links):
            key = next((k for k in LINK_COLORS if k in name.lower()), None)
            if key is not None:
                colors[name], label = LINK_COLORS[key], key.capitalize()
            else:
                colors[name] = _SEGMENT_COLORS[idx % len(_SEGMENT_COLORS)]
                label = name.split(leg, 1)[-1].strip("_")
            legend[label] = colors[name]
    return colors, legend


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
    title: str | None = None,
    elev: float = 25.0,
    azim: float = -60.0,
    figsize: tuple[float, float] = (12, 9),
    save_path: str | None = None,
) -> plt.Figure:
    """Draw the kinematic skeleton: joints as markers, links as lines.

    Parameters
    ----------
    q : configuration (default: neutral).
    centerline : unused; projection is set by ``CylinderSpec.project_leg``.
    title : optional title (thesis figures leave it to the caption).
    elev, azim : camera angles.
    save_path : if given, save the figure there.
    """
    LEG_NAMES = robot.spec.leg_names
    colors, legend = link_color_key(robot)

    if q is None:
        q = robot.neutral_config()

    robot.forward_kinematics(q)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    all_pts = []

    # --- Base ---
    base_pos = np.array(robot.data.oMi[0].translation)
    base_R = np.array(robot.data.oMi[0].rotation)
    all_pts.append(base_pos)
    ax.scatter(*base_pos, s=50, c="k", zorder=5)
    _draw_frame_axes(ax, base_pos, base_R, length=0.015)
    ax.text(
        base_pos[0],
        base_pos[1],
        base_pos[2] + 0.005,
        "base",
        fontsize=5,
        ha="center",
        va="bottom",
        color="dimgray",
    )

    # --- Legs ---
    feet = robot.foot_positions()
    for leg in LEG_NAMES:
        segments = robot.leg_skeleton(leg)
        foot_fid = robot.foot_frame_ids[leg]
        # Colour of the link carrying the foot point
        foot_color = colors.get(robot.spec.foot_points[leg][0], LINK_COLORS["foot"])

        # Joint markers at each segment start
        for p_start, _ in segments:
            all_pts.append(p_start)
            ax.scatter(*p_start, s=40, c="k", zorder=5)

        # Foot marker
        foot_pos = feet[leg]
        foot_R = np.array(robot.data.oMf[foot_fid].rotation)
        all_pts.append(foot_pos)
        ax.scatter(
            *foot_pos,
            s=60,
            c=foot_color,
            marker="v",
            edgecolors="k",
            linewidths=0.5,
            zorder=5,
        )
        _draw_frame_axes(ax, foot_pos, foot_R, length=0.015)

        # Segments, plus base -> first joint and last joint -> foot
        for idx, (p_start, p_end) in enumerate(segments):
            ax.plot(*zip(p_start, p_end), color=_SEGMENT_COLORS[idx % len(_SEGMENT_COLORS)],
                    linewidth=2.5, alpha=0.8)
        if segments:
            ax.plot(*zip(base_pos, segments[0][0]), color=_SEGMENT_COLORS[0],
                    linewidth=2.5, alpha=0.8)
            ax.plot(*zip(segments[-1][1], foot_pos), color=foot_color,
                    linewidth=2.0, linestyle="--", alpha=0.7)

    all_pts = np.array(all_pts)
    _set_equal_aspect(ax, all_pts)
    # Force 0.1 m ticks to match the geometry figures.
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))

    ax.set_xlabel("X [m]", fontsize=8)
    ax.set_ylabel("Y [m]", fontsize=8)
    ax.set_zlabel("Z [m]", fontsize=8)
    if title:
        ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)

    legend_elements = [
        Patch(facecolor=c, edgecolor="k", label=n) for n, c in legend.items()
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=8)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# --- Geometry representation figures ---


def _make_urdf_transform_manager(robot: QuadrupedRobot):
    """Load the robot URDF into a pytransform3d UrdfTransformManager."""
    from pytransform3d.urdf import UrdfTransformManager

    # Replaces "package://", hence the trailing slash.
    package_dir = str(robot.urdf_path.parent.parent.parent) + "/"
    with open(robot.urdf_path) as f:
        urdf_str = f.read()
    tm = UrdfTransformManager()
    tm.load_urdf(urdf_str, package_dir=package_dir)
    return tm


def _draw_cylinder(ax, center, axis_world, radius, length, color, alpha=0.6, n=16, edgecolor="k"):
    """Draw one cylinder and return its surface points."""
    X, Y, Z = _cylinder_mesh(radius, length, n_facets=n)
    R_cyl = _rotation_align_z_to(axis_world)
    pts = []
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            p = R_cyl @ np.array([X[i, j], Y[i, j], Z[i, j]]) + center
            X[i, j], Y[i, j], Z[i, j] = p
            pts.append(p)
    ax.plot_surface(X, Y, Z, alpha=alpha, color=color, edgecolor=edgecolor, linewidth=0.3)
    return pts


def visualize_robot_representations(
    robot: QuadrupedRobot,
    q: np.ndarray | None = None,
    title: str | None = None,
    elev: float = 25.0,
    azim: float = -60.0,
    figsize: tuple[float, float] = (7, 6),
    save_prefix: str | None = None,
    save_suffix: str = ".png",
) -> dict[str, plt.Figure]:
    """Mesh, drag cylinders, buoyancy cylinders and their overlay as four figures.

    Returns ``{"mesh", "drag", "buoyancy", "overlay": Figure}`` with a shared
    bounding box. With ``save_prefix`` each is saved as
    ``<save_prefix>_<key><save_suffix>``.
    """
    import warnings

    import matplotlib.colors as mcolors

    if q is None:
        q = robot.neutral_config()

    colors, legend = link_color_key(robot)
    robot.forward_kinematics(q)
    robot.build_cylinders()

    tm = _make_urdf_transform_manager(robot)
    # Set all tree joints (passive ones close the loops on closed-chain robots).
    # Continuous joints are stored as (cos, sin).
    for jid in range(1, robot.model.njoints):
        joint = robot.model.joints[jid]
        if joint.nq == 1:
            angle = float(q[joint.idx_q])
        elif joint.nq == 2:
            angle = float(np.arctan2(q[joint.idx_q + 1], q[joint.idx_q]))
        else:  # free-flyer base
            continue
        tm.set_joint(robot.model.names[jid], angle)
    for v in tm.visuals:
        link_name = v.frame.split("visual:")[1].rsplit("/", 1)[0]
        v.color = list(mcolors.to_rgba(colors.get(link_name, "#888888")))

    subtitles = {
        "mesh": "STL mesh",
        "drag": r"Drag cylinder ($r$ = RMS vertex distance from axis)",
        "buoyancy": r"Buoyancy / added-mass cylinder ($r = \sqrt{V_{\mathrm{disp}}/(\pi L)}$)",
        "overlay": "All representations overlaid (mesh, drag, buoyancy)",
    }

    figs: dict[str, plt.Figure] = {}
    axes: dict[str, plt.Axes] = {}
    all_pts: list = []

    for kind in ("mesh", "drag", "buoyancy", "overlay"):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

        if kind in ("mesh", "overlay"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tm.plot_visuals(
                    "base_link", ax=ax,
                    alpha=0.7 if kind == "mesh" else 0.45,
                    wireframe=False, convex_hull_of_mesh=False,
                )

        if kind in ("drag", "buoyancy", "overlay"):
            for name, link in robot.links.items():
                cyl = link.cylinder
                if cyl is None:
                    continue
                color = colors.get(name, "#888888")
                r_vol = np.sqrt(max(cyl.volume_displaced / (np.pi * cyl.length), 1e-8))
                if kind == "drag":
                    all_pts.extend(
                        _draw_cylinder(ax, cyl.center, cyl.axis_world, cyl.radius, cyl.length, color)
                    )
                elif kind == "buoyancy":
                    _draw_cylinder(ax, cyl.center, cyl.axis_world, r_vol, cyl.length, color)
                else:  # overlay: drag cylinder + buoyancy outline
                    _draw_cylinder(ax, cyl.center, cyl.axis_world, cyl.radius, cyl.length, color,
                                   alpha=0.30, edgecolor="#1a1a6e")
                    _draw_cylinder(ax, cyl.center, cyl.axis_world, r_vol, cyl.length, color,
                                   alpha=0.18, edgecolor="#6e1a1a")

        figs[kind] = fig
        axes[kind] = ax

    # Shared bounding box from the mesh vertices
    for name in robot.links:
        go = robot.link_geom_objects.get(name)
        if go is None:
            continue
        oMj = robot.data.oMi[go.parentJoint]
        T = oMj * go.placement
        R, t = np.array(T.rotation), np.array(T.translation)
        g = go.geometry
        verts = np.array([g.vertex(i) for i in range(g.num_vertices)]) @ R.T + t
        all_pts.extend(verts[::20].tolist())
    all_pts_arr = np.array(all_pts)

    legend_elements = [
        Patch(facecolor=c, edgecolor="k", label=n) for n, c in legend.items()
    ]
    overlay_legend = legend_elements + [
        Patch(facecolor="white", edgecolor="#1a1a6e", label="Drag cyl. edge"),
        Patch(facecolor="white", edgecolor="#6e1a1a", label="Buoyancy cyl. edge"),
    ]

    for kind, fig in figs.items():
        ax = axes[kind]
        _set_equal_aspect(ax, all_pts_arr)
        ax.set_xlabel("X [m]", fontsize=8)
        ax.set_ylabel("Y [m]", fontsize=8)
        ax.set_zlabel("Z [m]", fontsize=8)
        ax.view_init(elev=elev, azim=azim)
        ax.legend(handles=overlay_legend if kind == "overlay" else legend_elements,
                  loc="upper left", fontsize=8)
        if title:
            fig.suptitle(f"{title}: {subtitles[kind]}", fontsize=10)
        fig.tight_layout()
        if save_prefix is not None:
            fig.savefig(f"{save_prefix}_{kind}{save_suffix}", dpi=150, bbox_inches="tight")

    return figs
