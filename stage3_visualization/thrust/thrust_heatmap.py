#!/usr/bin/env python3
"""Best-case drag thrust and optimal sweep direction over a leg's workspace (amph).

For each (thigh, calf) pose, the joint-velocity direction on the unit circle
that maximises forward drag is found:

    thrust = max_{|v_joints|=1} F_drag_x    [N at unit joint speed]
    arrow  = foot-velocity direction of that optimum

A power stroke along the arrows gives maximum thrust, a recovery stroke along
them maximum anti-thrust. The interactive figure shows the 2D map, a 3D surface
and, with --solution, the leg with a time slider. --save-map writes only the
2D map as a thesis figure.

Usage:
  python stage3_visualization/thrust/thrust_heatmap.py
  python stage3_visualization/thrust/thrust_heatmap.py --solution task3_solution.npz
  python stage3_visualization/thrust/thrust_heatmap.py --leg Hind_Left --grid 80 --save heatmap.png
  python stage3_visualization/thrust/thrust_heatmap.py --solution task3_solution.npz --save-map map.pdf --half
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
from matplotlib.ticker import MaxNLocator
from matplotlib.widgets import Slider
from scipy.interpolate import LinearNDInterpolator, griddata
from scipy.spatial import cKDTree

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "stage1_gait_optimization"))
from stage3_visualization.common.drag_model import LEG_SEGMENTS, drag_force, link_drag_terms  # noqa: E402
from stage3_visualization.common.thesis_style import HALF, half_width  # noqa: E402

from stage1_gait_optimization.hydro_model import load_robot  # noqa: E402
from stage1_gait_optimization.hydro_model.robot import QuadrupedRobot  # noqa: E402

# Half-width but taller canvas for --save-map; the bottom colourbar makes the
# map height-limited (see plot_thrust_map).
HALF_MAP = (HALF[0], 3.0)

# Same link colours as hydro_model/visualization.py
_LEG_COLORS = {"side": "#E8A838", "thigh": "#5CB85C", "calf": "#D9534F"}


# --- Heatmap computation ---


def _leg_nv_indices(robot: QuadrupedRobot, leg: str) -> tuple[int, int, int]:
    """Velocity indices of a leg's side, thigh and calf joints."""
    base = robot.n_base_v
    names = robot.actuated_joint_names
    return (
        base + names.index(f"{leg}_Side_joint"),
        base + names.index(f"{leg}_Thigh_joint"),
        base + names.index(f"{leg}_Calf_joint"),
    )


def _leg_q_indices(robot: QuadrupedRobot, leg: str) -> tuple[int, int, int]:
    """Configuration indices of a leg's side, thigh and calf joints."""
    base = robot.n_base_q
    names = robot.actuated_joint_names
    return (
        base + names.index(f"{leg}_Side_joint"),
        base + names.index(f"{leg}_Thigh_joint"),
        base + names.index(f"{leg}_Calf_joint"),
    )


def compute_max_thrust(
    robot: QuadrupedRobot, q: np.ndarray, leg: str, n_angles: int = 72,
) -> tuple[float, np.ndarray]:
    """Max forward drag over unit (v_thigh, v_calf) and the corresponding foot direction.

    Returns ``(thrust, u_foot)`` with ``u_foot`` the unit sagittal foot-velocity
    direction. Call FK at ``q`` first.
    """
    _, thigh_nv, calf_nv = _leg_nv_indices(robot, leg)
    foot_fid = robot.foot_frame_ids[leg]

    J_foot = pin.computeFrameJacobian(
        robot.model, robot.data, q, foot_fid,
        pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
    )
    J_tc = J_foot[np.ix_([0, 2], [thigh_nv, calf_nv])]  # 2×2 sagittal

    # Pose-dependent terms once per link; the velocity sweep is vectorised.
    link_cache = []
    for tname in LEG_SEGMENTS:
        terms = link_drag_terms(robot, f"{leg}_{tname}_link", q)
        if terms is not None:
            link_cache.append(terms)

    angles = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)
    v_tc_all = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # (n_angles, 2)

    thrust = np.zeros(n_angles)
    for cyl, R, alpha, axis, J in link_cache:
        # Thigh and calf columns only
        v_origin = v_tc_all @ J[:3, [thigh_nv, calf_nv]].T   # (n_angles, 3)
        omega = v_tc_all @ J[3:, [thigh_nv, calf_nv]].T      # (n_angles, 3)
        thrust += drag_force(cyl, R, alpha, axis, v_origin, omega)[:, 0]

    best_idx = int(np.argmax(thrust))
    best_v_tc = v_tc_all[best_idx]
    u_foot = J_tc @ best_v_tc
    n = float(np.linalg.norm(u_foot))
    u_foot = u_foot / n if n > 1e-9 else np.array([1.0, 0.0])
    return float(thrust[best_idx]), u_foot


def build_heatmap(robot: QuadrupedRobot, leg: str, n_grid: int = 60):
    """Sweep (thigh, calf) over the URDF limits (side = 0).

    Returns foot x, foot z, max thrust and optimal direction per sample.
    """
    q_base = robot.neutral_config()
    _, thigh_q, calf_q = _leg_q_indices(robot, leg)
    foot_fid = robot.foot_frame_ids[leg]

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

    # Different postures can reach the same foot position (the workspace folds),
    # which makes the interpolation spiky. Keep the best posture per 1 mm cell.
    key = np.round(np.column_stack((xs, zs)) / 1e-3).astype(np.int64)
    order = np.lexsort((-eff, key[:, 1], key[:, 0]))
    first = np.ones(len(order), dtype=bool)
    first[1:] = np.any(key[order[1:]] != key[order[:-1]], axis=1)
    keep = order[first]
    return xs[keep], zs[keep], eff[keep], dirs[keep]


# --- Solution animation data ---


def _build_anim_data(robot: QuadrupedRobot, sol_path: Path, leg: str) -> dict:
    """Base-frame sagittal (x, z) of the leg joints and foot at every node.

    Base-frame (not world) positions, to match the heatmap computed at the neutral base.
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
        # Base placement (a view into robot.data, so read it per step)
        oMb = robot.data.oMi[1]
        for k in joint_keys:
            v = oMb.actInv(proj[k])
            data[k][t] = [v[0], v[2]]

    data["N"] = N
    data["T"] = T
    return data


# --- Leg stick figure ---


def _init_leg_ax(ax: plt.Axes, anim_data: dict, leg: str, xs: np.ndarray, zs: np.ndarray):
    """Draw the leg at node 0 and return the artist handles."""
    t0_pts = {k: anim_data[k][0] for k in ("side", "thigh", "calf", "foot")}

    # Body box centred on the mean hip position
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

    h_side = _seg(t0_pts["side"], t0_pts["thigh"], _LEG_COLORS["side"])
    h_thigh = _seg(t0_pts["thigh"], t0_pts["calf"], _LEG_COLORS["thigh"])
    h_calf = _seg(t0_pts["calf"], t0_pts["foot"], _LEG_COLORS["calf"], ls="--")

    jx = [t0_pts[k][0] for k in ("side", "thigh", "calf")]
    jz = [t0_pts[k][1] for k in ("side", "thigh", "calf")]
    (h_joints,) = ax.plot(jx, jz, "ko", ms=9, zorder=5)

    (h_foot,) = ax.plot([t0_pts["foot"][0]], [t0_pts["foot"][1]],
                         "v", color="#9B59B6", ms=14, markeredgecolor="k", zorder=6)

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

    # Full foot path
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
    """Move the leg artists to node ``t``."""
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


# --- Plotting ---


def _interp_grid(xs: np.ndarray, zs: np.ndarray, eff: np.ndarray):
    """Thrust on a regular grid, blanked away from sampled poses: ``(Xg, Zg, Eg, tree)``.

    Needed because griddata fills the convex hull, while the workspace is
    concave. "Far" scales with the local sample spacing.
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
    """Optimal sweep arrows at the samples nearest a coarse lattice (even spacing)."""
    gx = np.linspace(xs.min(), xs.max(), n_arrows)
    gz = np.linspace(zs.min(), zs.max(), n_arrows)
    Gx, Gz = np.meshgrid(gx, gz)
    d_q, i_q = tree.query(np.column_stack((Gx.ravel(), Gz.ravel())))
    # No arrows in unreachable areas
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
    """The 2D thrust map alone as a thesis figure (no title, no cursor).

    On narrow canvases (< 4 in) the colourbar goes below the map, which only
    pays off with a taller canvas such as ``HALF_MAP``. The thesis style is
    scoped to this function since the interactive labels use unicode arrows
    that usetex cannot render.
    """
    Xg, Zg, Eg, tree = _interp_grid(xs, zs, eff)
    with plt.style.context(["science"]), plt.rc_context({"text.usetex": True}):
        # Narrow: smaller fonts and a single-column legend
        narrow = figsize[0] < 4.0
        if narrow:
            half_width()
        fig, ax = plt.subplots(figsize=figsize)
        cf = ax.contourf(Xg, Zg, Eg, levels=40, cmap="viridis")
        ax.contour(Xg, Zg, Eg, levels=12, colors="k", linewidths=0.4, alpha=0.4)
        cb = fig.colorbar(cf, ax=ax, **(dict(location="bottom", pad=0.15,
                                             fraction=0.075, aspect=30)
                                        if narrow else {}))
        cb.set_label("max thrust [N at unit joint speed]")
        if narrow:
            # Fewer ticks than the 40 contour levels would give
            cb.locator = MaxNLocator(5)
            cb.update_ticks()

        handles = []
        if dirs is not None:
            # Sparser arrows than on screen
            _draw_quiver(ax, xs, zs, dirs, tree, n_arrows=10, scale=20,
                         width=0.005, keep_frac=0.45)
            handles.append(Line2D([], [], ls="none", marker=r"$\rightarrow$",
                                  color="0.35", ms=9,
                                  label="optimal sweep direction"))
        if foot_traj is not None:
            # Trajectory above the arrows (zorder 6)
            ax.plot(foot_traj[:, 0], foot_traj[:, 1], "k-", lw=2.0, zorder=7)
            handles.append(Line2D([], [], color="k", lw=2.0,
                                  label="OCP trajectory"))

        ax.set_xlabel(r"foot $x$ [m]")
        ax.set_ylabel(r"foot $z$ [m]")
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        if handles:
            # Legend above the axes; any corner covers the map for some leg.
            ax.legend(handles=handles, loc="lower center",
                      bbox_to_anchor=(0.5, 1.0),
                      ncol=1 if narrow else len(handles), fontsize=7,
                      frameon=False, borderaxespad=0.4, handlelength=1.6,
                      columnspacing=1.4, labelspacing=0.3)
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
    """Interactive figure: 2D map, 3D surface and (with ``anim_data``) the leg with a slider."""
    Xg, Zg, Eg, _tree = _interp_grid(xs, zs, eff)
    # Interpolator for slider updates
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

    # --- 2D map (top left) ---
    cf = ax1.contourf(Xg, Zg, Eg, levels=40, cmap="viridis")
    ax1.contour(Xg, Zg, Eg, levels=12, colors="k", linewidths=0.4, alpha=0.4)
    plt.colorbar(cf, ax=ax1, label="Max thrust  [N at unit joint speed]")

    if dirs is not None:
        _draw_quiver(ax1, xs, zs, dirs, _tree, label="optimal sweep direction")

    if foot_traj is not None:
        # Trajectory above the arrows (zorder 6)
        ax1.plot(foot_traj[:, 0], foot_traj[:, 1], "k-", lw=2.5, alpha=0.9,
                 zorder=7, label="OCP trajectory")
        ax1.plot(foot_traj[0, 0], foot_traj[0, 1], "ko", ms=8, zorder=7)
        ax1.plot(foot_traj[-1, 0], foot_traj[-1, 1], "k^", ms=8, zorder=7)

    # Current-node marker (empty until the slider is set up)
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
        # Legend below the axes (see plot_thrust_map; the title occupies the top)
        ax1.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.14),
                   ncol=2, frameon=False)

    # --- 3D surface (top right); NaN cells stay holes ---
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

    # --- Leg stick figure and slider (bottom) ---
    if has_anim:
        N = int(anim_data["N"])
        T = float(anim_data.get("T", 1.0))

        leg_handles = _init_leg_ax(ax3, anim_data, leg, xs, zs)

        fx0, fz0 = anim_data["foot"][0]
        e0 = float(_interp([[fx0, fz0]])[0])
        if np.isnan(e0):
            e0 = float(np.nanmin(eff))
        dot_heat.set_data([fx0], [fz0])
        dot_3d.set_data([fx0], [fz0])
        dot_3d.set_3d_properties([e0])

        (dot_leg,) = ax3.plot([fx0], [fz0], "ro", ms=11, zorder=10,
                               markeredgecolor="darkred", markeredgewidth=1.5)

        dt = T / N
        time_text = ax3.text(
            0.02, 0.96, f"t = 0 / {N}  ({0*dt:.2f} s)",
            transform=ax3.transAxes, fontsize=9, va="top",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
        )

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
                        help="pose-sweep resolution per joint (default: %(default)s)")
    parser.add_argument("--half", action="store_true",
                        help=f"render --save-map at half text width "
                             f"({HALF_MAP[0]} x {HALF_MAP[1]} in) instead of the "
                             f"default")
    parser.add_argument("--figsize", type=float, nargs=2, default=None,
                        metavar=("W", "H"),
                        help="explicit --save-map canvas in inches; overrides --half")
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
        figsize = (tuple(args.figsize) if args.figsize
                   else HALF_MAP if args.half else (5.6, 4.0))
        plot_thrust_map(xs, zs, eff, leg=args.leg, dirs=dirs,
                        foot_traj=foot_traj, save_path=args.save_map,
                        figsize=figsize)
    else:
        plot(xs, zs, eff, leg=args.leg, dirs=dirs, foot_traj=foot_traj,
             anim_data=anim_data, save_path=args.save)


if __name__ == "__main__":
    main()
