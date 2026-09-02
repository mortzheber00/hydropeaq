#!/usr/bin/env python3
"""
Thrust heatmap: best-case thrust capability + optimal sweep direction per pose.

For each (q_thigh, q_calf) in the reachable workspace (side joint fixed at zero),
sweeps the joint-velocity direction (v_thigh, v_calf) on the unit circle and
finds the direction that maximises forward drag-thrust:

    thrust = max_{|v_joints|=1}  F_thrust_x(v_joints)    [N]
    u_opt  = foot-velocity direction induced by that v_joints (du_x, du_z)

The constraint |v_joints|=1 makes the metric pose-only and bounded everywhere
(unlike |v_foot|=1, which blows up at foot-singularities).  Heatmap colour
shows max thrust capability — always ≥ 0 because drag is direction-symmetric,
so red/green diverging doesn't apply, we use a sequential colormap.

Reading the map:

    power stroke aligned WITH arrows     → max forward thrust  (good)
    recovery stroke aligned WITH arrows  → max anti-thrust     (bad)
    stroke perpendicular to arrows       → neutral             (no drag impact)

Outputs:
  Top-left  — 2D contour map in foot (x, z) space, coloured by efficiency
  Top-right — 3D topographic surface (valley = low thrust, peak = high thrust)
  Bottom    — Leg stick figure (sagittal plane) with slider to scrub through the
               OCP solution; red dot tracks current foot position on both top panels.

``--save-map`` writes the top-left panel on its own instead, as a thesis
figure: thesis style, no title, no timestep cursor.

Usage:
  python thrust_heatmap.py
  python thrust_heatmap.py --solution ../task3_solution.npz
  python thrust_heatmap.py --leg Hind_Left --grid 80 --save heatmap.png
  python thrust_heatmap.py --solution ../task3_solution.npz --save-map map.pdf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from matplotlib.widgets import Slider
from scipy.interpolate import LinearNDInterpolator, griddata
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parents[1]))
from stage1_gait_optimization.hydro_model import get_spec, load_robot
from stage1_gait_optimization.hydro_model.hydrodynamics import RHO_WATER
from stage1_gait_optimization.hydro_model.robot import QuadrupedRobot

# This tool is still amph-specific: it assumes a 3-joint serial leg and builds
# a 2x2 sagittal Jacobian over (thigh, calf) with the side joint pinned.  The
# BODY2 analogue -- sweep (theta1, theta2), mask by assemblability, sum drag
# over the leg's six links -- is a separate piece of work.
SUPPORTED_ROBOTS = ("amph",)
URDF_PATH = get_spec("amph").urdf_path


def _require_supported(robot_name: str) -> None:
    if robot_name not in SUPPORTED_ROBOTS:
        raise NotImplementedError(
            f"thrust_heatmap supports {SUPPORTED_ROBOTS}, not {robot_name!r}: its "
            f"thrust model assumes a 3-joint serial leg. See the plan, stage3 section."
        )

CD_T = 1.0
CD_A = 0.1
RHO = RHO_WATER

# Link colors matching visualization.py
_LEG_COLORS = {"side": "#E8A838", "thigh": "#5CB85C", "calf": "#D9534F"}


# ---------------------------------------------------------------------------
# Thrust heatmap computation
# ---------------------------------------------------------------------------


def _leg_nv_indices(robot: QuadrupedRobot, leg: str) -> tuple[int, int, int]:
    base = robot.n_base_v
    names = robot.actuated_joint_names
    return (
        base + names.index(f"{leg}_Side_joint"),
        base + names.index(f"{leg}_Thigh_joint"),
        base + names.index(f"{leg}_Calf_joint"),
    )


def _leg_q_indices(robot: QuadrupedRobot, leg: str) -> tuple[int, int, int]:
    base = robot.n_base_q
    names = robot.actuated_joint_names
    return (
        base + names.index(f"{leg}_Side_joint"),
        base + names.index(f"{leg}_Thigh_joint"),
        base + names.index(f"{leg}_Calf_joint"),
    )


def _link_drag_x(robot: QuadrupedRobot, link_name: str, q: np.ndarray, v: np.ndarray) -> float:
    link = robot.links.get(link_name)
    if link is None or link.cylinder is None:
        return 0.0
    cyl = link.cylinder
    R = np.array(robot.data.oMf[link.frame_id].rotation)
    J = pin.computeFrameJacobian(
        robot.model, robot.data, q, link.frame_id,
        pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
    )
    # Velocity at the cylinder midpoint, not just the frame origin.
    # Frame origin may be at a joint pivot (zero linear velocity there),
    # so we add the rotational contribution: v_mid = v_origin + omega x r_offset
    v_origin = J[:3, :] @ v
    omega = J[3:, :] @ v
    r_offset = R @ cyl.center_local
    v_link = v_origin + np.cross(omega, r_offset)

    axis = R @ cyl.axis_local
    axis /= np.linalg.norm(axis) + 1e-15
    v_ax_mag = float(np.dot(v_link, axis))
    v_ax = v_ax_mag * axis
    v_tr = v_link - v_ax
    F_drag = (
        -0.5 * RHO * CD_T * cyl.cross_section_transverse * np.linalg.norm(v_tr) * v_tr
        - 0.5 * RHO * CD_A * cyl.cross_section_axial * abs(v_ax_mag) * v_ax
    )
    # F_drag is the fluid force ON the link (opposes motion).
    # For a backward-sweeping link (v_x < 0), F_drag[0] > 0 = forward thrust.
    return float(F_drag[0])


def compute_max_thrust(
    robot: QuadrupedRobot, q: np.ndarray, leg: str, n_angles: int = 72,
) -> tuple[float, np.ndarray]:
    """Best F_drag_x at this pose, over unit-norm (v_thigh, v_calf) directions.

    Sweeps the joint-velocity direction on the unit circle and returns:
      thrust:  max F_drag_x [N at unit joint speed], pose-only and bounded
      u_foot:  foot-velocity direction (du_x, du_z) induced by the optimal
               joint velocity — the direction the foot must sweep through to
               extract that max thrust.
    """
    _, thigh_nv, calf_nv = _leg_nv_indices(robot, leg)
    foot_fid = robot.foot_frame_ids[leg]

    J_foot = pin.computeFrameJacobian(
        robot.model, robot.data, q, foot_fid,
        pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
    )
    J_tc = J_foot[np.ix_([0, 2], [thigh_nv, calf_nv])]  # 2×2 sagittal

    # Cache per-link drag parameters (frame rotations / Jacobians are pose-only).
    link_cache = []
    for tname in ("Thigh", "Calf", "Foot"):
        link = robot.links.get(f"{leg}_{tname}_link")
        if link is None or link.cylinder is None:
            continue
        cyl = link.cylinder
        R = np.array(robot.data.oMf[link.frame_id].rotation)
        J = pin.computeFrameJacobian(
            robot.model, robot.data, q, link.frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        axis = R @ cyl.axis_local
        axis /= np.linalg.norm(axis) + 1e-15
        # We only need the columns acting on (v_thigh, v_calf)
        link_cache.append((
            J[:3, [thigh_nv, calf_nv]],
            J[3:, [thigh_nv, calf_nv]],
            R @ cyl.center_local, axis,
            0.5 * RHO * CD_T * cyl.cross_section_transverse,
            0.5 * RHO * CD_A * cyl.cross_section_axial,
        ))

    # Vectorised angle sweep
    angles = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)
    v_tc_all = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # (n_angles, 2)

    thrust = np.zeros(n_angles)
    for J_lin_tc, J_ang_tc, r_off, axis, k_t, k_a in link_cache:
        v_origin = v_tc_all @ J_lin_tc.T              # (n_angles, 3)
        omega = v_tc_all @ J_ang_tc.T                 # (n_angles, 3)
        v_link = v_origin + np.cross(omega, r_off)    # (n_angles, 3)
        v_ax_mag = v_link @ axis                      # (n_angles,)
        v_ax = v_ax_mag[:, None] * axis[None, :]
        v_tr = v_link - v_ax
        v_tr_norm = np.linalg.norm(v_tr, axis=1)
        thrust += -k_t * v_tr_norm * v_tr[:, 0] - k_a * np.abs(v_ax_mag) * v_ax[:, 0]

    best_idx = int(np.argmax(thrust))
    best_v_tc = v_tc_all[best_idx]
    u_foot = J_tc @ best_v_tc
    n = float(np.linalg.norm(u_foot))
    u_foot = u_foot / n if n > 1e-9 else np.array([1.0, 0.0])
    return float(thrust[best_idx]), u_foot


def build_heatmap(robot: QuadrupedRobot, leg: str, n_grid: int = 60):
    """Sweep (q_thigh, q_calf) with side=0 and return foot positions, max
    thrust capability, and the optimal sweep direction at each pose."""
    q_base = robot.neutral_config()
    _, thigh_q, calf_q = _leg_q_indices(robot, leg)
    foot_fid = robot.foot_frame_ids[leg]

    # Sweep each joint over its actual URDF travel.  The fixed box this
    # replaced (thigh +/-90 deg, calf -60..120 deg) put 55% of the samples
    # outside the joint limits and still clipped 15 deg off the calf's top end.
    lo, hi = robot.model.lowerPositionLimit, robot.model.upperPositionLimit
    thigh_vals = np.linspace(lo[thigh_q], hi[thigh_q], n_grid)
    calf_vals = np.linspace(lo[calf_q], hi[calf_q], n_grid)

    xs, zs, eff, dirs = [], [], [], []
    for qt in thigh_vals:
        for qc in calf_vals:
            q = q_base.copy()
            q[thigh_q] = qt
            q[calf_q] = qc
            robot.forward_kinematics(q)
            pos = np.array(robot.data.oMf[foot_fid].translation)
            xs.append(pos[0])
            zs.append(pos[2])
            thrust, u_opt = compute_max_thrust(robot, q, leg)
            eff.append(thrust)
            dirs.append(u_opt)

    xs, zs = np.array(xs), np.array(zs)
    eff, dirs = np.array(eff), np.array(dirs)

    # Past ~90 deg of calf the joint rectangle folds onto itself, so two
    # postures can put the foot in the same place with different link
    # orientations — and different drag.  Thrust is then multivalued in (x, z),
    # and the interpolation triangulates across both sheets, which shows up as
    # spikes near the fold.  Keep the best posture per 1 mm cell (the display
    # resolution): the map already reports a best case over sweep directions,
    # so a max over postures is the same kind of envelope.
    key = np.round(np.column_stack((xs, zs)) / 1e-3).astype(np.int64)
    order = np.lexsort((-eff, key[:, 1], key[:, 0]))
    first = np.ones(len(order), dtype=bool)
    first[1:] = np.any(key[order[1:]] != key[order[:-1]], axis=1)
    keep = order[first]
    return xs[keep], zs[keep], eff[keep], dirs[keep]


# ---------------------------------------------------------------------------
# OCP animation data
# ---------------------------------------------------------------------------


def _build_anim_data(robot: QuadrupedRobot, sol_path: Path, leg: str) -> dict:
    """Pre-compute body-frame sagittal (x, z) positions for all joints at all timesteps.

    Uses leg_centerline_positions (sagittal projection) so the chain appears planar,
    exactly matching the foot positions shown on the heatmap.  Those are swept with
    the base at neutral_config, so the frames only agree once the base rotation is
    undone too — subtracting the base position alone leaves the pitch in.
    """
    d = np.load(sol_path)
    X, nq, N = d["X"], int(d["nq"]), int(d["N"])
    T = float(d["T"])

    joint_keys = ("side", "thigh", "calf", "foot")
    data: dict[str, np.ndarray] = {k: np.zeros((N + 1, 2)) for k in joint_keys}

    for t in range(N + 1):
        q = X[:nq, t]
        robot.forward_kinematics(q)
        proj = robot.leg_centerline_positions(leg)
        # oMi[1] is the free-flyer placement, and a view into robot.data, so it
        # has to be read here rather than hoisted out of the loop.
        oMb = robot.data.oMi[1]
        for k in joint_keys:
            v = oMb.actInv(proj[k])
            data[k][t] = [v[0], v[2]]

    data["N"] = N
    data["T"] = T
    return data


# ---------------------------------------------------------------------------
# Leg stick figure helpers
# ---------------------------------------------------------------------------


def _init_leg_ax(ax: plt.Axes, anim_data: dict, leg: str, xs: np.ndarray, zs: np.ndarray):
    """Draw the initial leg stick figure and return a dict of artist handles."""
    t0_pts = {k: anim_data[k][0] for k in ("side", "thigh", "calf", "foot")}

    # Body reference: a rounded box centred on the mean side-joint x position
    side_x = float(np.mean(anim_data["side"][:, 0]))
    side_z = float(t0_pts["side"][1])
    box_w, box_h = 0.20, 0.036
    body_patch = FancyBboxPatch(
        (side_x - box_w / 2, side_z - box_h / 2), box_w, box_h,
        boxstyle="round,pad=0.005",
        linewidth=1.5, edgecolor="steelblue", facecolor="#AED6F1", zorder=2,
    )
    ax.add_patch(body_patch)
    ax.text(side_x, side_z, "body", ha="center", va="center", fontsize=8, color="steelblue", zorder=3)

    def _seg(a, b, color, lw=3.0, ls="-"):
        (h,) = ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, ls=ls,
                       solid_capstyle="round", zorder=4)
        return h

    # Three link segments with matching colors
    h_side = _seg(t0_pts["side"], t0_pts["thigh"], _LEG_COLORS["side"])
    h_thigh = _seg(t0_pts["thigh"], t0_pts["calf"], _LEG_COLORS["thigh"])
    h_calf = _seg(t0_pts["calf"], t0_pts["foot"], _LEG_COLORS["calf"], ls="--")

    # Joint markers
    jx = [t0_pts[k][0] for k in ("side", "thigh", "calf")]
    jz = [t0_pts[k][1] for k in ("side", "thigh", "calf")]
    (h_joints,) = ax.plot(jx, jz, "ko", ms=9, zorder=5)

    # Foot
    (h_foot,) = ax.plot([t0_pts["foot"][0]], [t0_pts["foot"][1]],
                         "v", color="#9B59B6", ms=14, markeredgecolor="k", zorder=6)

    # Axes formatting
    all_x = np.concatenate([anim_data[k][:, 0] for k in ("side", "thigh", "calf", "foot")])
    all_z = np.concatenate([anim_data[k][:, 1] for k in ("side", "thigh", "calf", "foot")])
    margin = 0.04
    ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
    ax.set_ylim(all_z.min() - margin, all_z.max() + margin)
    ax.set_aspect("equal")
    ax.set_xlabel("Foot  x [m]  (forward →)")
    ax.set_ylabel("Foot  z [m]  (up ↑)")
    ax.set_title(f"{leg.replace('_', ' ')} — sagittal plane view")
    ax.grid(True, alpha=0.25)

    # Trajectory ghost
    ft = anim_data["foot"]
    ax.plot(ft[:, 0], ft[:, 1], color="gray", lw=1, alpha=0.4, zorder=3, label="full trajectory")
    ax.legend(fontsize=7, loc="upper right")

    return {
        "side": h_side,
        "thigh": h_thigh,
        "calf": h_calf,
        "joints": h_joints,
        "foot": h_foot,
    }


def _update_leg(handles: dict, anim_data: dict, t: int):
    pts = {k: anim_data[k][t] for k in ("side", "thigh", "calf", "foot")}
    handles["side"].set_data([pts["side"][0], pts["thigh"][0]],
                              [pts["side"][1], pts["thigh"][1]])
    handles["thigh"].set_data([pts["thigh"][0], pts["calf"][0]],
                               [pts["thigh"][1], pts["calf"][1]])
    handles["calf"].set_data([pts["calf"][0], pts["foot"][0]],
                              [pts["calf"][1], pts["foot"][1]])
    jx = [pts[k][0] for k in ("side", "thigh", "calf")]
    jz = [pts[k][1] for k in ("side", "thigh", "calf")]
    handles["joints"].set_data(jx, jz)
    handles["foot"].set_data([pts["foot"][0]], [pts["foot"][1]])


# ---------------------------------------------------------------------------
# Main plot
# ---------------------------------------------------------------------------


def _interp_grid(xs: np.ndarray, zs: np.ndarray, eff: np.ndarray):
    """Max thrust on a regular grid, blanked where the leg cannot reach.

    griddata fills the convex hull of the samples, but a two-link leg sweeps a
    crescent: the hull bridges concavities the foot cannot reach.  Every cell
    far from an actual sampled pose is blanked, with "far" scaled by the local
    sample spacing, which varies across the map with the leg Jacobian.

    Returns ``(Xg, Zg, Eg, tree)`` — the tree is reused to place the quiver.
    """
    x_lin = np.linspace(xs.min(), xs.max(), 300)
    z_lin = np.linspace(zs.min(), zs.max(), 300)
    Xg, Zg = np.meshgrid(x_lin, z_lin)
    Eg = griddata((xs, zs), eff, (Xg, Zg), method="linear")
    pts = np.column_stack((xs, zs))
    tree = cKDTree(pts)
    spacing = tree.query(pts, k=2)[0][:, 1]
    d, idx = tree.query(np.column_stack((Xg.ravel(), Zg.ravel())))
    Eg[(d > 2.0 * spacing[idx]).reshape(Xg.shape)] = np.nan
    return Xg, Zg, Eg, tree


def _draw_quiver(ax, xs, zs, dirs, tree, n_arrows: int = 14, scale: float = 22,
                 width: float = 0.005, keep_frac: float = 0.6, **kw):
    """Optimal sweep direction at each pose, subsampled for clarity.

    One sample nearest each node of a coarse lattice, so the arrows stay evenly
    spaced in foot space.  Index striding would not: the fold reduction leaves
    the samples no longer a full square grid.
    """
    gx = np.linspace(xs.min(), xs.max(), n_arrows)
    gz = np.linspace(zs.min(), zs.max(), n_arrows)
    Gx, Gz = np.meshgrid(gx, gz)
    d_q, i_q = tree.query(np.column_stack((Gx.ravel(), Gz.ravel())))
    # Drop lattice nodes with no sample nearby, so no arrows in blank areas.
    step = max(gx[1] - gx[0], gz[1] - gz[0])
    keep = np.unique(i_q[d_q < keep_frac * step])
    return ax.quiver(
        xs[keep], zs[keep], dirs[keep, 0], dirs[keep, 1],
        color="white", edgecolor="black", linewidth=0.5,
        pivot="middle", scale=scale, width=width, zorder=6, **kw,
    )


def plot_thrust_map(
    xs: np.ndarray,
    zs: np.ndarray,
    eff: np.ndarray,
    leg: str,
    dirs: np.ndarray | None = None,
    foot_traj: np.ndarray | None = None,
    save_path: Path | None = None,
    figsize: tuple[float, float] = (5.6, 4.0),
):
    """The thrust map alone, as a figure for the thesis.

    The panel ``plot`` puts top left, without what only makes sense on screen:
    no title (the LaTeX caption carries it) and no timestep cursor.

    Drawn under the thesis style rather than this module's plain-matplotlib
    default, so it sets in the same face as the rest of the figures; that is
    scoped to this function, since the interactive figure's labels carry unicode
    arrows that usetex would choke on.
    """
    import scienceplots  # noqa: F401  registers the 'science' matplotlib style

    Xg, Zg, Eg, tree = _interp_grid(xs, zs, eff)
    with plt.style.context(["science"]), plt.rc_context({"text.usetex": True}):
        fig, ax = plt.subplots(figsize=figsize)
        cf = ax.contourf(Xg, Zg, Eg, levels=40, cmap="viridis")
        ax.contour(Xg, Zg, Eg, levels=12, colors="k", linewidths=0.4, alpha=0.4)
        fig.colorbar(cf, ax=ax, label="max thrust [N at unit joint speed]")

        handles = []
        if dirs is not None:
            # Fewer, shorter arrows than the on-screen panel: this canvas is
            # half its width, and at the original density they close over the
            # trajectory instead of framing it.
            _draw_quiver(ax, xs, zs, dirs, tree, n_arrows=10, scale=20,
                         width=0.005, keep_frac=0.45)
            handles.append(Line2D([], [], ls="none", marker=r"$\rightarrow$",
                                  color="0.35", ms=9,
                                  label="optimal sweep direction"))
        if foot_traj is not None:
            # Above the quiver (zorder 6): at the default zorder the arrows drew
            # over the trajectory, which is the thing you actually want to read.
            ax.plot(foot_traj[:, 0], foot_traj[:, 1], "k-", lw=2.0, zorder=7)
            handles.append(Line2D([], [], color="k", lw=2.0,
                                  label="OCP trajectory"))

        ax.set_xlabel(r"foot $x$ [m]")
        ax.set_ylabel(r"foot $z$ [m]")
        ax.set_aspect("equal")
        if handles:
            ax.legend(handles=handles, loc="upper right", fontsize=7,
                      framealpha=0.9)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
            print(f"Saved → {save_path}")
        else:
            plt.show()
    return fig


def plot(
    xs: np.ndarray,
    zs: np.ndarray,
    eff: np.ndarray,
    leg: str,
    dirs: np.ndarray | None = None,
    foot_traj: np.ndarray | None = None,
    anim_data: dict | None = None,
    save_path: Path | None = None,
):
    Xg, Zg, Eg, _tree = _interp_grid(xs, zs, eff)
    # Fast interpolator for slider updates
    _interp = LinearNDInterpolator(list(zip(xs, zs)), eff)

    has_anim = anim_data is not None

    # --- Figure layout ---
    if has_anim:
        fig = plt.figure(figsize=(14, 13))
        gs = GridSpec(
            2, 2,
            figure=fig,
            height_ratios=[1.1, 1.0],
            hspace=0.50,
            wspace=0.35,
            left=0.07, right=0.96, top=0.95, bottom=0.10,
        )
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1], projection="3d")
        ax3 = fig.add_subplot(gs[1, :])
    else:
        fig = plt.figure(figsize=(14, 6))
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2, projection="3d")
        ax3 = None

    # ── Top-left: 2D contour map ──────────────────────────────────────────
    cf = ax1.contourf(Xg, Zg, Eg, levels=40, cmap="viridis")
    ax1.contour(Xg, Zg, Eg, levels=12, colors="k", linewidths=0.4, alpha=0.4)
    plt.colorbar(cf, ax=ax1, label="Max thrust  [N at unit joint speed]")

    if dirs is not None:
        _draw_quiver(ax1, xs, zs, dirs, _tree, label="optimal sweep direction")

    if foot_traj is not None:
        # Above the quiver (zorder 6): at the default zorder the arrows drew
        # over the trajectory, which is the thing you actually want to read.
        ax1.plot(foot_traj[:, 0], foot_traj[:, 1], "k-", lw=2.5, alpha=0.9,
                 zorder=7, label="OCP trajectory")
        ax1.plot(foot_traj[0, 0], foot_traj[0, 1], "ko", ms=8, zorder=7)
        ax1.plot(foot_traj[-1, 0], foot_traj[-1, 1], "k^", ms=8, zorder=7)

    # Red dot (current timestep) — starts hidden
    (dot_heat,) = ax1.plot([], [], "ro", ms=11, zorder=10,
                            markeredgecolor="darkred", markeredgewidth=1.5)

    ax1.set_xlabel("Foot  x [m]  (forward →)")
    ax1.set_ylabel("Foot  z [m]  (up ↑)")
    ax1.set_title(
        f"{leg.replace('_', ' ')} — max thrust capability + optimal sweep\n"
        "(arrows: foot-velocity direction giving max forward thrust)"
    )
    ax1.set_aspect("equal")
    if foot_traj is not None or dirs is not None:
        ax1.legend(fontsize=8, loc="upper right")

    # ── Top-right: 3D topographic surface ────────────────────────────────
    # NaN straight through: masked cells become holes in the surface.  Filling
    # them with the minimum (as this did) draws a flat floor that reads as a
    # genuine low-thrust basin rather than as absent data.
    ax2.plot_surface(Xg, Zg, Eg, cmap="viridis", edgecolor="none", alpha=0.88)

    if foot_traj is not None:
        e_traj = griddata((xs, zs), eff, foot_traj, method="linear")
        ax2.plot(foot_traj[:, 0], foot_traj[:, 1], e_traj,
                 "k-", lw=2.5, alpha=0.9, label="OCP trajectory")

    (dot_3d,) = ax2.plot([], [], [], "ro", ms=9, zorder=10,
                          markeredgecolor="darkred", markeredgewidth=1.5)

    ax2.set_xlabel("Foot x [m]")
    ax2.set_ylabel("Foot z [m]")
    ax2.set_zlabel("Max thrust [N]")
    ax2.set_title("3D max-thrust topography\n(peak = pose with highest thrust capability)")
    ax2.view_init(elev=30, azim=-55)
    if foot_traj is not None:
        ax2.legend(fontsize=8)

    # ── Bottom: Leg stick figure + slider ────────────────────────────────
    if has_anim:
        N = int(anim_data["N"])
        T = float(anim_data.get("T", 1.0))

        leg_handles = _init_leg_ax(ax3, anim_data, leg, xs, zs)

        # Initialise red dot at t=0
        fx0, fz0 = anim_data["foot"][0]
        e0 = float(_interp([[fx0, fz0]])[0])
        if np.isnan(e0):
            e0 = float(np.nanmin(eff))
        dot_heat.set_data([fx0], [fz0])
        dot_3d.set_data([fx0], [fz0])
        dot_3d.set_3d_properties([e0])

        # Red dot also on leg axes (for fun, same point)
        (dot_leg,) = ax3.plot([fx0], [fz0], "ro", ms=11, zorder=10,
                               markeredgecolor="darkred", markeredgewidth=1.5)

        # Time-step label
        dt = T / N
        time_text = ax3.text(
            0.02, 0.96, f"t = 0 / {N}  ({0*dt:.2f} s)",
            transform=ax3.transAxes, fontsize=9, va="top",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
        )

        # Slider
        ax_sl = fig.add_axes([0.15, 0.03, 0.70, 0.025])
        slider = Slider(ax_sl, "Time step", 0, N, valinit=0, valstep=1, color="#5CB85C")

        def _update(val):
            t = int(slider.val)
            _update_leg(leg_handles, anim_data, t)

            fx, fz = anim_data["foot"][t]
            dot_heat.set_data([fx], [fz])
            dot_leg.set_data([fx], [fz])

            e_val = float(_interp([[fx, fz]])[0])
            if np.isnan(e_val):
                e_val = float(np.nanmin(eff))
            dot_3d.set_data([fx], [fz])
            dot_3d.set_3d_properties([e_val])

            time_text.set_text(f"t = {t} / {N}  ({t * dt:.2f} s)")
            fig.canvas.draw_idle()

        slider.on_changed(_update)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--leg", default="Front_Left",
        choices=["Front_Left", "Front_Right", "Hind_Left", "Hind_Right"],
    )
    parser.add_argument(
        "--solution", type=Path, default=None,
        help="OCP solution .npz — enables leg animation panel + slider",
    )
    parser.add_argument("--grid", type=int, default=250,
                        help="Pose-sweep resolution per joint (default 150)")
    parser.add_argument("--save", type=Path, default=None, help="Save figure to path")
    parser.add_argument(
        "--save-map", type=Path, default=None,
        help="write just the thrust map — the top-left panel, titleless — as "
             "a standalone thesis figure (.pdf); the other panels are not built",
    )
    args = parser.parse_args()

    print("Loading robot…")
    robot = load_robot("amph")
    robot.forward_kinematics(robot.neutral_config())
    robot.build_cylinders()

    print(f"Sweeping {args.grid}×{args.grid} workspace for {args.leg}…")
    xs, zs, eff, dirs = build_heatmap(robot, leg=args.leg, n_grid=args.grid)
    print(f"  Max-thrust range: {eff.min():.4f} … {eff.max():.4f} N  (per unit joint speed)")

    anim_data = None
    foot_traj = None
    if args.solution is not None:
        if args.solution.exists():
            print(f"Loading OCP solution from {args.solution}…")
            anim_data = _build_anim_data(robot, args.solution, args.leg)
            foot_traj = anim_data["foot"]
        else:
            print(f"Warning: solution not found: {args.solution}")

    if args.save_map:
        plot_thrust_map(xs, zs, eff, leg=args.leg, dirs=dirs,
                        foot_traj=foot_traj, save_path=args.save_map)
    else:
        plot(xs, zs, eff, leg=args.leg, dirs=dirs, foot_traj=foot_traj,
             anim_data=anim_data, save_path=args.save)


if __name__ == "__main__":
    main()
