#!/usr/bin/env python3
"""Compare a Gazebo/SPH replay (rosbag) with the OCP solution it replayed.

Resamples joint states and base pose onto the OCP time grid, prints RMSEs and
writes three figures: joint tracking, base tracking and a per-joint RMSE bar
chart. The robot is taken from the solution file.

Usage:
  python stage2_sim_validation/gazebo_replay/validate_sim.py --bag sim_log.bag --ocp task3_solution.npz --start 1.49
  python stage2_sim_validation/gazebo_replay/validate_sim.py --out figs --no-show
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.interpolate import interp1d

sys.path.insert(0, str(Path(__file__).parents[2]))
sys.path.insert(0, str(Path(__file__).parents[2] / "stage3_visualization" / "common"))
from stage1_gait_optimization.hydro_model import get_spec  # noqa: E402
from thesis_style import LEG_COLORS, PALETTE, tex  # noqa: E402  also activates the plot style

# --- Joint naming, bound by use_robot() from the solution file ---
SPEC = None
LEG_NAMES = JOINT_TYPES = OCP_JOINT_NAMES = None
N_PER_LEG = None


def use_robot(name: str) -> None:
    """Bind the module's naming tables to a registered robot."""
    global SPEC, LEG_NAMES, JOINT_TYPES, N_PER_LEG, OCP_JOINT_NAMES
    SPEC = get_spec(name)
    LEG_NAMES = list(SPEC.leg_names)
    JOINT_TYPES = list(SPEC.leg_joint_labels)
    N_PER_LEG = len(JOINT_TYPES)
    OCP_JOINT_NAMES = list(SPEC.actuated_joint_names)


C_OCP, C_SIM = PALETTE[0], PALETTE[2]  # model solid, simulation dashed

DEG = r"$^\circ$"   # usetex has no degree glyph in the text font


# --- Data loading ---

def load_ocp(path: str):
    """Return ``(q_joints, t_ocp, T, xyz_ocp, vx_ocp)`` and bind the module to the file's robot.

    q_joints : (n_joints, N+1) joint angles [rad]
    xyz_ocp  : (3, N+1) base position; vx_ocp: forward speed
    """
    d = np.load(path)
    # Files without a robot field are amph solutions.
    use_robot(str(d["robot"]) if "robot" in d.files else "amph")
    X = d["X"]
    T = float(d["T"])
    N = int(d["N"])
    nq = int(d["nq"])
    q_joints = X[7:nq, :]  # skip base [x,y,z, qx,qy,qz,qw]
    xyz_ocp = X[0:3, :]    # base position (world frame)
    t_ocp = np.linspace(0.0, T, N + 1)
    vx_ocp = np.gradient(X[0, :], t_ocp)
    return q_joints, t_ocp, T, xyz_ocp, vx_ocp


def read_bag_joint_states(bag_path: str):
    """Read ``/<robot>/joint_states``.

    Returns ``times`` (M,) relative to the first message and ``positions``
    (n_joints, M) in OCP joint order [rad].
    """
    import rosbag

    times = []
    pos_list = []
    ocp_indices = None  # bag index for each OCP joint

    with rosbag.Bag(bag_path) as bag:
        for _, msg, t in bag.read_messages(topics=[f"/{SPEC.ros}/joint_states"]):
            if ocp_indices is None:
                bag_names = list(msg.name)
                ocp_indices = []
                for ocp_name in OCP_JOINT_NAMES:
                    if ocp_name not in bag_names:
                        raise ValueError(
                            f"Joint {ocp_name!r} not found in bag.\n"
                            f"  Bag joints: {bag_names}"
                        )
                    ocp_indices.append(bag_names.index(ocp_name))
            pos = np.array(msg.position)
            pos_list.append(pos[ocp_indices])
            times.append(t.to_sec())

    times = np.array(times)
    positions = np.stack(pos_list, axis=1)  # (n_joints, M)
    times -= times[0]
    return times, positions


def read_bag_model_states(bag_path: str, model_name: str = None):
    """Read the robot's base pose from ``/gazebo/model_states``.

    ``model_name`` defaults to the robot's Gazebo model name. Returns ``times``
    (M,) relative to the first message, world-frame ``xyz`` (3, M) [m] and
    forward velocity ``vx_lin`` (M,) [m/s].
    """
    if model_name is None:
        model_name = SPEC.ros
    import rosbag

    times, xs, ys, zs, vxs = [], [], [], [], []
    model_idx = None

    with rosbag.Bag(bag_path) as bag:
        for _, msg, t in bag.read_messages(topics=["/gazebo/model_states"]):
            if model_idx is None:
                if model_name not in msg.name:
                    raise ValueError(
                        f"Model {model_name!r} not in model_states. "
                        f"Available: {list(msg.name)}"
                    )
                model_idx = list(msg.name).index(model_name)
            p = msg.pose[model_idx].position
            v = msg.twist[model_idx].linear
            xs.append(p.x); ys.append(p.y); zs.append(p.z)
            vxs.append(v.x)
            times.append(t.to_sec())

    times = np.array(times)
    xyz = np.array([xs, ys, zs])  # (3, M)
    vx_lin = np.array(vxs)        # (M,)
    times -= times[0]
    return times, xyz, vx_lin


# --- Alignment ---

def align(q_ocp, t_ocp, q_sim, t_sim, start_time: float):
    """Resample sim joint data onto the OCP grid; ``start_time`` is the sim time of OCP t=0."""
    t_query = t_ocp + start_time
    if t_query[-1] > t_sim[-1]:
        print(
            f"  WARNING: OCP window end ({t_query[-1]:.3f}s) exceeds bag end "
            f"({t_sim[-1]:.3f}s) — extrapolating last sample."
        )
    aligned = np.zeros_like(q_ocp)
    for i in range(q_ocp.shape[0]):
        f = interp1d(
            t_sim, q_sim[i, :], kind="linear",
            bounds_error=False, fill_value=(q_sim[i, 0], q_sim[i, -1]),
        )
        aligned[i, :] = f(t_query)
    return aligned


def align_base(t_ocp, xyz_sim, vx_sim, t_sim, start_time: float):
    """Resample base position (3, N+1) and forward speed (N+1,) onto the OCP grid."""
    t_query = t_ocp + start_time
    xyz_aligned = np.zeros((3, len(t_ocp)))
    for i in range(3):
        f = interp1d(
            t_sim, xyz_sim[i, :], kind="linear",
            bounds_error=False, fill_value=(xyz_sim[i, 0], xyz_sim[i, -1]),
        )
        xyz_aligned[i, :] = f(t_query)
    f_vx = interp1d(
        t_sim, vx_sim, kind="linear",
        bounds_error=False, fill_value=(vx_sim[0], vx_sim[-1]),
    )
    vx_aligned = f_vx(t_query)
    return xyz_aligned, vx_aligned


def compute_rmse(q_ocp, q_sim):
    """Per-joint RMSE [rad]."""
    return np.sqrt(np.mean((q_ocp - q_sim) ** 2, axis=1))


def compute_base_rmse(xyz_ocp, vx_ocp, xyz_sim, vx_sim):
    """RMSE of [Δx, Δy, Δz, v_x]; positions relative to t=0 since the origins differ."""
    xyz_o = xyz_ocp - xyz_ocp[:, [0]]
    xyz_s = xyz_sim - xyz_sim[:, [0]]
    return np.array([np.sqrt(np.mean((o - s) ** 2))
                     for o, s in zip([*xyz_o, vx_ocp], [*xyz_s, vx_sim])])


# --- Plotting ---

def _overlay(ax, t, y_ocp, y_sim):
    """Model and simulation curves with the gap shaded."""
    ax.fill_between(t, y_ocp, y_sim, color=C_SIM, alpha=0.16, lw=0, zorder=1)
    ax.plot(t, y_ocp, color=C_OCP, lw=1.3, zorder=3)
    ax.plot(t, y_sim, color=C_SIM, lw=1.1, ls=(0, (4, 1.6)), zorder=4)
    ax.set_xlim(t[0], t[-1])
    ax.grid(alpha=0.3, lw=0.4)


def _headroom(ax, frac: float = 0.22):
    """Extend the y range to make room for the RMSE badge (once per shared axis)."""
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + frac * (hi - lo))


def _corner(ax, text):
    """RMSE badge in the top-right corner."""
    ax.annotate(text, xy=(0.975, 0.94), xycoords="axes fraction",
                ha="right", va="top", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.75",
                          lw=0.5, alpha=0.9))


def _overlay_legend(fig):
    handles = [
        Line2D([], [], color=C_OCP, lw=1.3, label="reduced-order model"),
        Line2D([], [], color=C_SIM, lw=1.1, ls=(0, (4, 1.6)),
               label="SPH--Gazebo reference"),
        Patch(facecolor=C_SIM, alpha=0.16, label="tracking error"),
    ]
    # "outside" lets constrained layout reserve space for the legend.
    fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=8)


def plot_joint_tracking(q_ocp, q_sim, t_ocp, rmse_all):
    """Grid of joint tracking panels: legs as rows, joints as columns (shared y per column)."""
    n_legs = len(LEG_NAMES)
    fig, axes = plt.subplots(
        n_legs, N_PER_LEG, sharex=True, sharey="col", squeeze=False,
        figsize=(2.45 * N_PER_LEG + 0.7, 1.55 * n_legs + 0.9),
        layout="constrained",
    )

    for i, leg in enumerate(LEG_NAMES):
        for j, jtype in enumerate(JOINT_TYPES):
            k = i * N_PER_LEG + j
            ax = axes[i, j]
            _overlay(ax, t_ocp, np.degrees(q_ocp[k, :]), np.degrees(q_sim[k, :]))
            _corner(ax, rf"RMSE {np.degrees(rmse_all[k]):.2f}{DEG}")
            if i == 0:
                ax.set_title(tex(jtype))
            if j == 0:
                ax.set_ylabel(tex(leg.replace("_", " ")) + "\n" + rf"$q$ [{DEG}]")
            if i == n_legs - 1:
                ax.set_xlabel(r"time $t$ [s]")

    for j in range(N_PER_LEG):   # once per shared y axis
        _headroom(axes[0, j])

    _overlay_legend(fig)
    return fig


def plot_base(xyz_ocp, vx_ocp, xyz_sim, vx_sim, t_ocp):
    """Base displacement and forward speed with an error row below each panel."""
    xyz_o = xyz_ocp - xyz_ocp[:, [0]]
    xyz_s = xyz_sim - xyz_sim[:, [0]]

    titles = [r"$\Delta x$ (forward)", r"$\Delta y$ (lateral)",
              r"$\Delta z$ (heave)", r"forward velocity $v_x$"]
    units = ["m", "m", "m", "m/s"]
    ocp_data = [xyz_o[0], xyz_o[1], xyz_o[2], vx_ocp]
    sim_data = [xyz_s[0], xyz_s[1], xyz_s[2], vx_sim]

    rmse_base = compute_base_rmse(xyz_ocp, vx_ocp, xyz_sim, vx_sim)

    fig, axes = plt.subplots(2, 4, sharex=True, squeeze=False,
                             gridspec_kw=dict(height_ratios=[2.0, 1.0]),
                             figsize=(10.0, 4.4), layout="constrained")

    for col, (name, unit, q_o, q_s) in enumerate(
        zip(titles, units, ocp_data, sim_data)
    ):
        ax0, ax1 = axes[0, col], axes[1, col]
        _overlay(ax0, t_ocp, q_o, q_s)
        ax0.set_title(name)
        ax0.set_ylabel(f"[{unit}]")
        _headroom(ax0)
        _corner(ax0, rf"RMSE {rmse_base[col]:.4f} {unit}")

        err = q_o - q_s
        ax1.axhline(0, color="0.4", lw=0.5, ls=":")
        ax1.fill_between(t_ocp, err, color=C_SIM, alpha=0.16, lw=0)
        ax1.plot(t_ocp, err, color=C_SIM, lw=1.0)
        ax1.set_xlim(t_ocp[0], t_ocp[-1])
        ax1.grid(alpha=0.3, lw=0.4)
        ax1.set_ylabel(f"error [{unit}]")
        ax1.set_xlabel(r"time $t$ [s]")

    _overlay_legend(fig)
    return fig


def plot_rmse(rmse_all):
    """All actuated joints on one axis, grouped and coloured by leg."""
    rmse_deg = np.degrees(rmse_all)
    # 0.8 bar widths between leg groups
    x = np.array([i * (N_PER_LEG + 0.8) + j
                  for i in range(len(LEG_NAMES)) for j in range(N_PER_LEG)])
    colors = [LEG_COLORS[i % len(LEG_COLORS)]
              for i in range(len(LEG_NAMES)) for _ in range(N_PER_LEG)]

    fig, ax = plt.subplots(figsize=(7.0, 3.0), layout="constrained")
    bars = ax.bar(x, rmse_deg, width=0.85, color=colors, edgecolor="k", lw=0.4)
    ax.axhline(rmse_deg.mean(), color="0.35", lw=0.9, ls="--", zorder=3)

    for bar, v in zip(bars, rmse_deg):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02 * rmse_deg.max(),
                f"{v:.2f}", ha="center", va="bottom", fontsize=6)

    ax.set_xticks(x)
    ax.set_xticklabels([tex(jt) for _ in LEG_NAMES for jt in JOINT_TYPES],
                       rotation=45, ha="right", fontsize=7)
    ax.set_ylabel(rf"RMSE [{DEG}]")
    ax.set_ylim(0, rmse_deg.max() * 1.22)
    ax.grid(axis="y", alpha=0.3, lw=0.4)
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)

    leg_handles = [Patch(facecolor=LEG_COLORS[i % len(LEG_COLORS)],
                         edgecolor="k", lw=0.4,
                         label=tex(leg.replace("_", " ")))
                   for i, leg in enumerate(LEG_NAMES)]
    mean_handle = Line2D([], [], color="0.35", lw=0.9, ls="--",
                         label=rf"all-joint mean ({rmse_deg.mean():.2f}{DEG})")
    fig.legend(handles=leg_handles + [mean_handle], loc="outside right upper",
               fontsize=7.5)

    return fig


def show_tabbed(figures: list, titles: list):
    """Show figures as tabs in one Tk window."""
    import tkinter as tk
    from tkinter import ttk

    from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                                   NavigationToolbar2Tk)

    root = tk.Tk()
    root.title("Sim Validation")
    root.attributes("-zoomed", True)  # start maximised (Linux)

    notebook = ttk.Notebook(root)
    notebook.pack(fill=tk.BOTH, expand=True)

    for fig, title in zip(figures, titles):
        frame = ttk.Frame(notebook)
        notebook.add(frame, text=title)

        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()

        toolbar = NavigationToolbar2Tk(canvas, frame)
        toolbar.update()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    root.mainloop()


def main():
    parser = argparse.ArgumentParser(
        description="Validate Gazebo sim bag against OCP solution.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--bag", default="/home/ws/sim_log.bag")
    parser.add_argument("--ocp", default="/home/ws/task3_solution.npz")
    parser.add_argument(
        "--start", type=float, default=0.0,
        help="Sim time [s] corresponding to OCP t=0.",
    )
    parser.add_argument("--out", default=".", help="Output directory.")
    parser.add_argument("--format", default="pdf",
                        help="Figure format; pdf keeps the text vector for LaTeX.")
    parser.add_argument("--no-show", action="store_true",
                        help="Write the figures without opening the viewer.")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading OCP: {args.ocp}")
    q_ocp, t_ocp, T, xyz_ocp, vx_ocp = load_ocp(args.ocp)
    print(f"  {q_ocp.shape[1]} nodes, T={T:.3f}s, {q_ocp.shape[0]} joints")

    print(f"Reading bag: {args.bag}")
    t_js, q_sim = read_bag_joint_states(args.bag)
    t_ms, xyz_sim_raw, vx_sim_raw = read_bag_model_states(args.bag)
    print(f"  joint_states:  {q_sim.shape[1]} messages over {t_js[-1]:.3f}s")
    print(f"  model_states:  {xyz_sim_raw.shape[1]} messages over {t_ms[-1]:.3f}s")

    print(f"Aligning  (sim start offset = {args.start:.3f}s) …")
    q_sim_aligned = align(q_ocp, t_ocp, q_sim, t_js, args.start)
    xyz_sim_aligned, vx_sim_aligned = align_base(t_ocp, xyz_sim_raw, vx_sim_raw, t_ms, args.start)

    rmse = compute_rmse(q_ocp, q_sim_aligned)

    print("\nRMSE per joint:")
    print(f"  {'Joint':<38} RMSE [rad]   RMSE [°]")
    print(f"  {'-'*60}")
    for i, name in enumerate(OCP_JOINT_NAMES):
        print(f"  {name:<38} {rmse[i]:.4f} rad   {np.degrees(rmse[i]):.3f}°")
    print(f"\n  Overall mean RMSE: {np.degrees(rmse.mean()):.3f}°")

    # Normalise base RMSEs by the simulated (reference) cycle distance d, and
    # the speed RMSE by d/T.
    rmse_base = compute_base_rmse(xyz_ocp, vx_ocp, xyz_sim_aligned, vx_sim_aligned)
    d_cycle = float(np.linalg.norm(xyz_sim_aligned[:, -1] - xyz_sim_aligned[:, 0]))
    d_model = float(np.linalg.norm(xyz_ocp[:, -1] - xyz_ocp[:, 0]))
    scales = [d_cycle, d_cycle, d_cycle, d_cycle / T]

    print(f"\nBase RMSE over t=0…{t_ocp[-1]:.3f}s "
          f"(SPH cycle distance d={d_cycle:.4f} m, d/T={d_cycle / T:.4f} m/s; "
          f"model d={d_model:.4f} m, d/T={d_model / T:.4f} m/s):")
    print(f"  {'':6}  {'RMSE':>12}  {'Normalised':>11}")
    for i, (label, unit) in enumerate(
        zip(["Δx", "Δy", "Δz", "v_x"], ["m", "m", "m", "m/s"])
    ):
        norm = (f"{100 * rmse_base[i] / scales[i]:.2f}%"
                if scales[i] > 1e-9 else "n/a")
        print(f"  {label:<6}  {rmse_base[i]:8.4f} {unit:<3}  {norm:>11}")

    print("\nGenerating figures …")
    figs = {
        "joint_tracking": plot_joint_tracking(q_ocp, q_sim_aligned, t_ocp, rmse),
        "base": plot_base(xyz_ocp, vx_ocp, xyz_sim_aligned, vx_sim_aligned, t_ocp),
        "rmse": plot_rmse(rmse),
    }
    for tag, fig in figs.items():
        path = out_dir / f"validation_{tag}.{args.format.lstrip('.')}"
        # Extra padding: the tight bbox under-measures usetex text.
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.12)
        print(f"  Saved {path}")

    if not args.no_show:
        show_tabbed(list(figs.values()),
                    ["Joint tracking", "Base state", "RMSE"])
    print("\nDone.")


if __name__ == "__main__":
    main()
