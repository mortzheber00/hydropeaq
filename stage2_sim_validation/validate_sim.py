#!/usr/bin/env python3
"""
Validate Gazebo fluid simulation against OCP solution.

Reads joint states from a ROS1 .bag file, aligns to the OCP time grid,
and produces per-leg validation figures (3-panel layout per joint):

  Row 0: Joint angle — OCP vs Sim overlay
  Row 1: Tracking error over time  (q_ocp - q_sim)
  Row 2: RMSE bar chart for all 12 joints  (this leg highlighted)

Usage:
    python3 validate_sim.py [--bag PATH] [--ocp PATH] [--start T] [--out DIR]

    --bag    path to .bag file            (default: /home/ws/sim_log.bag)
    --ocp    path to OCP .npz file        (default: /home/ws/task3_solution.npz)
    --start  sim time [s] for OCP t=0     (default: 0.0 = bag start)
    --out    output directory for PNGs    (default: .)
"""

import argparse
import tkinter as tk
from tkinter import ttk
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from scipy.interpolate import interp1d

# ── Canonical joint ordering (matches OCP state vector X[7:19, :]) ──────────
LEG_NAMES = ["Front_Left", "Front_Right", "Hind_Left", "Hind_Right"]
JOINT_TYPES = ["Side", "Thigh", "Calf"]

OCP_JOINT_NAMES = [
    f"{leg}_{jtype}_joint"
    for leg in LEG_NAMES
    for jtype in JOINT_TYPES
]

# Short labels for bar chart: FL_Si, FL_Th, FL_Ca, FR_Si, …
_LEG_SHORT = {"Front_Left": "FL", "Front_Right": "FR", "Hind_Left": "HL", "Hind_Right": "HR"}
_JTYPE_SHORT = {"Side": "Si", "Thigh": "Th", "Calf": "Ca"}
BAR_LABELS = [
    f"{_LEG_SHORT[leg]}_{_JTYPE_SHORT[jt]}"
    for leg in LEG_NAMES
    for jt in JOINT_TYPES
]

LEG_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red"]


# ── Data loading ─────────────────────────────────────────────────────────────

def load_ocp(path: str):
    """Return (q_joints, t_ocp, T).

    q_joints : (12, N+1)  joint angles in OCP canonical order [rad]
    t_ocp    : (N+1,)     time vector [s]
    T        : float      cycle period [s]
    """
    d = np.load(path)
    X = d["X"]
    T = float(d["T"])
    N = int(d["N"])
    nq = int(d["nq"])
    q_joints = X[7:nq, :]  # skip base [x,y,z, qx,qy,qz,qw]
    xyz_ocp = X[0:3, :]    # base position (world frame)
    t_ocp = np.linspace(0.0, T, N + 1)
    # Forward speed: numerical derivative of x position (world frame)
    vx_ocp = np.gradient(X[0, :], t_ocp)
    return q_joints, t_ocp, T, xyz_ocp, vx_ocp


def read_bag_joint_states(bag_path: str):
    """Read /amph/joint_states from a ROS1 .bag file.

    Returns:
        times     : (M,)    timestamps [s] relative to first message
        positions : (12, M) joint positions in OCP canonical order [rad]
    """
    import rosbag

    times = []
    pos_list = []
    ocp_indices = None  # mapping: bag column index → OCP canonical index

    with rosbag.Bag(bag_path) as bag:
        for _, msg, t in bag.read_messages(topics=["/amph/joint_states"]):
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
    positions = np.stack(pos_list, axis=1)  # (12, M)
    times -= times[0]
    return times, positions


def read_bag_model_states(bag_path: str, model_name: str = "amph"):
    """Read /gazebo/model_states from a ROS1 .bag file.

    Returns:
        times   : (M,)   timestamps [s] relative to first message
        xyz     : (3, M) world-frame position [m]
        vx_lin  : (M,)   world-frame forward (x) velocity [m/s]
    """
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


# ── Alignment ────────────────────────────────────────────────────────────────

def align(q_ocp, t_ocp, q_sim, t_sim, start_time: float):
    """Interpolate sim data to the OCP time grid.

    start_time : sim timestamp [s] corresponding to OCP t=0.
    Returns q_sim_aligned : (12, N+1) [rad]
    """
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
    """Interpolate base position and forward speed to the OCP time grid.

    Returns:
        xyz_aligned : (3, N+1) [m]
        vx_aligned  : (N+1,)  [m/s]
    """
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
    """Per-joint RMSE. Returns (12,) [rad]."""
    return np.sqrt(np.mean((q_ocp - q_sim) ** 2, axis=1))


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_leg(leg_idx: int, q_ocp, q_sim, t_ocp, rmse_all, out_dir: Path):
    """Generate the 3-panel validation figure for one leg.

    Layout per joint column:
      row 0 — angle overlay (OCP vs Sim)
      row 1 — tracking error (q_ocp - q_sim)
      row 2 — RMSE bar chart for all 12 joints (this leg highlighted, full-width)
    """
    leg = LEG_NAMES[leg_idx]
    ji = leg_idx * 3  # first joint index for this leg

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(
        3, 3,
        height_ratios=[2.2, 1.2, 1.8],
        hspace=0.50,
        wspace=0.35,
    )

    for col, jtype in enumerate(JOINT_TYPES):
        j = ji + col
        q_o = np.degrees(q_ocp[j, :])
        q_s = np.degrees(q_sim[j, :])
        err = q_o - q_s

        # ── Row 0: angle overlay ─────────────────────────────────────────
        ax0 = fig.add_subplot(gs[0, col])
        ax0.plot(t_ocp, q_o, color="tab:blue", lw=1.8, label="OCP")
        ax0.plot(t_ocp, q_s, color="tab:orange", lw=1.5, ls="--", label="Sim")
        ax0.set_title(f"{jtype} joint", fontsize=10)
        if col == 0:
            ax0.set_ylabel(f"{leg.replace('_', ' ')}\nangle [°]", fontsize=9)
        else:
            ax0.set_ylabel("angle [°]", fontsize=9)
        ax0.legend(fontsize=8, loc="best")
        ax0.grid(True, alpha=0.3)
        ax0.tick_params(labelbottom=False)

        # ── Row 1: tracking error ────────────────────────────────────────
        ax1 = fig.add_subplot(gs[1, col], sharex=ax0)
        ax1.plot(t_ocp, err, color="tab:red", lw=1.5)
        ax1.axhline(0, color="k", lw=0.7, ls=":")
        ax1.fill_between(t_ocp, err, alpha=0.18, color="tab:red")
        ax1.set_ylabel("error [°]", fontsize=9)
        ax1.set_xlabel("time [s]", fontsize=9)
        ax1.grid(True, alpha=0.3)

        rmse_deg = np.degrees(rmse_all[j])
        ax1.annotate(
            f"RMSE = {rmse_deg:.2f}°",
            xy=(0.98, 0.92), xycoords="axes fraction",
            ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", lw=0.7),
        )

    # ── Row 2: RMSE bar chart for all 12 joints ──────────────────────────
    ax2 = fig.add_subplot(gs[2, :])

    bar_colors = [
        LEG_COLORS[i // 3] if (i // 3) == leg_idx else "lightgrey"
        for i in range(12)
    ]
    rmse_deg_all = np.degrees(rmse_all)
    bars = ax2.bar(
        range(12), rmse_deg_all,
        color=bar_colors, edgecolor="k", linewidth=0.5,
    )
    ax2.set_xticks(range(12))
    ax2.set_xticklabels(BAR_LABELS, rotation=45, ha="right", fontsize=9)
    ax2.set_ylabel("RMSE [°]", fontsize=9)
    ax2.set_title(
        f"RMSE — all joints  (mean = {rmse_deg_all.mean():.2f}°, "
        f"highlighted = {leg.replace('_', ' ')})",
        fontsize=10,
    )
    ax2.grid(True, axis="y", alpha=0.3)
    for bar, v in zip(bars, rmse_deg_all):
        ax2.text(
            bar.get_x() + bar.get_width() / 2, v + 0.01 * rmse_deg_all.max(),
            f"{v:.2f}", ha="center", va="bottom", fontsize=7,
        )

    fig.suptitle(
        f"Sim Validation — {leg.replace('_', ' ')} Leg",
        fontsize=13, fontweight="bold",
    )

    out_path = out_dir / f"validation_{leg.lower()}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved {out_path}")
    return fig


def plot_base(xyz_ocp, vx_ocp, xyz_sim, vx_sim, t_ocp, out_dir: Path):
    """3-panel validation figure for base position and forward speed.

    4 columns: x, y, z (position [m]), vx (forward speed [m/s])
      row 0 — overlay (OCP vs Sim), both shifted to start at 0
      row 1 — tracking error
      row 2 — RMSE bar chart for all 4 variables (full width)
    """
    # Shift both to zero at t=0 so we compare relative displacement / speed shape
    xyz_o = xyz_ocp - xyz_ocp[:, [0]]
    xyz_s = xyz_sim - xyz_sim[:, [0]]

    labels = ["x position", "y position", "z position", "forward speed (vx)"]
    units  = ["m", "m", "m", "m/s"]
    ocp_data = [xyz_o[0], xyz_o[1], xyz_o[2], vx_ocp]
    sim_data = [xyz_s[0], xyz_s[1], xyz_s[2], vx_sim]

    rmse_base = np.array([
        np.sqrt(np.mean((o - s) ** 2))
        for o, s in zip(ocp_data, sim_data)
    ])

    fig = plt.figure(figsize=(18, 9))
    gs = fig.add_gridspec(
        3, 4,
        height_ratios=[2.2, 1.2, 1.8],
        hspace=0.50,
        wspace=0.35,
    )

    for col, (label, unit, q_o, q_s) in enumerate(
        zip(labels, units, ocp_data, sim_data)
    ):
        err = q_o - q_s

        # ── Row 0: overlay ───────────────────────────────────────────────
        ax0 = fig.add_subplot(gs[0, col])
        ax0.plot(t_ocp, q_o, color="tab:blue", lw=1.8, label="OCP")
        ax0.plot(t_ocp, q_s, color="tab:orange", lw=1.5, ls="--", label="Sim")
        ax0.set_title(label, fontsize=10)
        ax0.set_ylabel(f"Δ [{unit}]" if col < 3 else f"[{unit}]", fontsize=9)
        ax0.legend(fontsize=8, loc="best")
        ax0.grid(True, alpha=0.3)
        ax0.tick_params(labelbottom=False)

        # ── Row 1: tracking error ────────────────────────────────────────
        ax1 = fig.add_subplot(gs[1, col], sharex=ax0)
        ax1.plot(t_ocp, err, color="tab:red", lw=1.5)
        ax1.axhline(0, color="k", lw=0.7, ls=":")
        ax1.fill_between(t_ocp, err, alpha=0.18, color="tab:red")
        ax1.set_ylabel(f"error [{unit}]", fontsize=9)
        ax1.set_xlabel("time [s]", fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.annotate(
            f"RMSE = {rmse_base[col]:.4f} {unit}",
            xy=(0.98, 0.92), xycoords="axes fraction",
            ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", lw=0.7),
        )

    # ── Row 2: RMSE bar chart ────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[2, :])
    bar_colors = ["tab:blue", "tab:blue", "tab:blue", "tab:purple"]
    bar_short = ["x pos", "y pos", "z pos", "vx"]
    bars = ax2.bar(
        range(4), rmse_base,
        color=bar_colors, edgecolor="k", linewidth=0.5,
    )
    ax2.set_xticks(range(4))
    ax2.set_xticklabels(bar_short, fontsize=10)
    ax2.set_ylabel("RMSE", fontsize=9)
    ax2.set_title("RMSE — base state variables", fontsize=10)
    ax2.grid(True, axis="y", alpha=0.3)
    for bar, v, unit in zip(bars, rmse_base, units):
        ax2.text(
            bar.get_x() + bar.get_width() / 2, v + 0.005 * rmse_base.max(),
            f"{v:.4f} {unit}", ha="center", va="bottom", fontsize=9,
        )

    fig.suptitle("Sim Validation — Base Position & Forward Speed", fontsize=13, fontweight="bold")

    out_path = out_dir / "validation_base.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved {out_path}")
    return fig


def show_tabbed(figures: list, titles: list):
    """Display a list of matplotlib figures as tabs in a single Tk window."""
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


# ── Main ─────────────────────────────────────────────────────────────────────

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
    parser.add_argument("--out", default=".", help="Output directory for PNGs.")
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

    print("\nGenerating figures …")
    figs = [
        plot_leg(leg_idx, q_ocp, q_sim_aligned, t_ocp, rmse, out_dir)
        for leg_idx in range(4)
    ]
    figs.append(plot_base(xyz_ocp, vx_ocp, xyz_sim_aligned, vx_sim_aligned, t_ocp, out_dir))

    titles = [leg.replace("_", " ") for leg in LEG_NAMES] + ["Base State"]
    show_tabbed(figs, titles)
    print("\nDone.")


if __name__ == "__main__":
    main()
