#!/usr/bin/env python3
"""
Validate the Gazebo fluid simulation against the OCP solution it was replayed
from, and write the comparison as thesis-ready vector figures.

Reads joint states and base pose from a ROS1 .bag file, resamples them onto the
OCP time grid, and produces three figures:

  1. Joint tracking  — one panel per (leg, joint); OCP and simulation overlaid,
     with the area between them shaded as the tracking error and the per-joint
     RMSE quoted in the corner.
  2. Base tracking   — the same comparison for base x/y/z and forward speed,
     with an explicit error trace underneath.
  3. RMSE summary    — every actuated joint on one axis, grouped and coloured by
     leg, against the all-joint mean.

The overlay is the comparison itself, so the error is drawn *between* the two
curves rather than in a separate row: sign, timing and magnitude are then read
off the same panel as the trajectories.  The RMSE bar chart used to be repeated
once per leg with a different group highlighted; it is one figure now.

Usage:
    python3 validate_sim.py [--bag PATH] [--ocp PATH] [--start T] [--out DIR]
                            [--format EXT] [--no-show]

    --bag     path to .bag file            (default: /home/ws/sim_log.bag)
    --ocp     path to OCP .npz file        (default: /home/ws/task3_solution.npz)
    --start   sim time [s] for OCP t=0     (default: 0.0 = bag start)
    --out     output directory             (default: .)
    --format  figure format                (default: pdf — vector, for LaTeX)
    --no-show skip the interactive viewer  (for headless figure generation)

The robot is read from the solution file, so the same command works for either
one; the bag has to be the run of that same solution.
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

sys.path.insert(0, str(Path(__file__).parents[1]))
sys.path.insert(0, str(Path(__file__).parents[1] / "stage3_visualization"))
from stage1_gait_optimization.hydro_model import get_spec  # noqa: E402
from thesis_style import PALETTE, tex  # noqa: E402  also activates the plot style

# ── Canonical ordering, taken from the robot's spec ─────────────────────────
# The plots below lay one column out per actuated joint of a leg, so this
# module works for any robot whose legs share a joint count.
#
# Which robot that is comes from the solution file, which records it -- the
# same source replay_trajectory.py reads.  It used to be a module constant
# here, so validating BODY2 meant editing the file, and forgetting to edit it
# back made the next amph run compare against the wrong joint names.
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


# The comparison is binary, so it gets the two ends of the shared palette:
# reference solid, measurement dashed on top of it.
C_OCP, C_SIM = PALETTE[0], PALETTE[2]
# Legs keep the colours plot_solution.py gives them in the animation, in the
# same leg order, so a leg reads the same here as in the swimming figures.  The
# shared palette is not usable for this: its fourth entry is a pink no other
# figure draws with.
LEG_COLORS = ("tab:blue", "tab:orange", "tab:green", "tab:red")

DEG = r"$^\circ$"   # usetex has no degree glyph in the text font


# ── Data loading ─────────────────────────────────────────────────────────────

def load_ocp(path: str):
    """Return (q_joints, t_ocp, T).

    q_joints : (12, N+1)  joint angles in OCP canonical order [rad]
    t_ocp    : (N+1,)     time vector [s]
    T        : float      cycle period [s]

    Also binds the module to the robot the file names, so everything below
    reads the right joints out of the bag.
    """
    d = np.load(path)
    # v1 files predate the field; they are all amph, which is what they were.
    use_robot(str(d["robot"]) if "robot" in d.files else "amph")
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
    """Read the robot's joint_states topic from a ROS1 .bag file.

    Returns:
        times     : (M,)    timestamps [s] relative to first message
        positions : (12, M) joint positions in OCP canonical order [rad]
    """
    import rosbag

    times = []
    pos_list = []
    ocp_indices = None  # mapping: bag column index → OCP canonical index

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
    positions = np.stack(pos_list, axis=1)  # (12, M)
    times -= times[0]
    return times, positions


def read_bag_model_states(bag_path: str, model_name: str = None):
    """``model_name`` defaults to the robot's Gazebo model name."""
    if model_name is None:
        model_name = SPEC.ros
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
    # Three rows because xyz is a position, not because a leg has three joints.
    # This loop ran over N_PER_LEG, which is 3 for amph by coincidence and 2 for
    # BODY2 -- there it left the z row at zero.
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


def compute_base_rmse(xyz_ocp, vx_ocp, xyz_sim, vx_sim):
    """RMSE of the four base panels: (4,) = [Δx, Δy, Δz [m], v_x [m/s]].

    Positions are referenced to t=0 first, exactly as the figure draws them:
    the OCP frame and the Gazebo spawn pose share no origin, so only the
    displacement is comparable.
    """
    xyz_o = xyz_ocp - xyz_ocp[:, [0]]
    xyz_s = xyz_sim - xyz_sim[:, [0]]
    return np.array([np.sqrt(np.mean((o - s) ** 2))
                     for o, s in zip([*xyz_o, vx_ocp], [*xyz_s, vx_sim])])


# ── Plotting ─────────────────────────────────────────────────────────────────

def _overlay(ax, t, y_ocp, y_sim):
    """Reference / measurement overlay with the gap between them shaded."""
    ax.fill_between(t, y_ocp, y_sim, color=C_SIM, alpha=0.16, lw=0, zorder=1)
    ax.plot(t, y_ocp, color=C_OCP, lw=1.3, zorder=3)
    ax.plot(t, y_sim, color=C_SIM, lw=1.1, ls=(0, (4, 1.6)), zorder=4)
    ax.set_xlim(t[0], t[-1])
    ax.grid(alpha=0.3, lw=0.4)


def _headroom(ax, frac: float = 0.22):
    """Open a band at the top of the panel for the RMSE badge to sit in.

    Call once per set of shared axes: on a shared-y column every call would
    expand the same limits again.
    """
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + frac * (hi - lo))


def _corner(ax, text):
    """RMSE badge, top-right, clear of the curves."""
    ax.annotate(text, xy=(0.975, 0.94), xycoords="axes fraction",
                ha="right", va="top", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.75",
                          lw=0.5, alpha=0.9))


def _overlay_legend(fig):
    handles = [
        Line2D([], [], color=C_OCP, lw=1.3, label="OCP solution"),
        Line2D([], [], color=C_SIM, lw=1.1, ls=(0, (4, 1.6)),
               label="Gazebo simulation"),
        Patch(facecolor=C_SIM, alpha=0.16, label="tracking error"),
    ]
    # "outside" placement is what constrained layout reserves room for, so the
    # legend never has to be nudged by hand.
    fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=8)


def plot_joint_tracking(q_ocp, q_sim, t_ocp, rmse_all):
    """Every actuated joint on one grid: legs down the rows, joints across.

    Columns share a y axis, so the same joint type is directly comparable
    between legs — the thing the figure is there to show.
    """
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

    for j in range(N_PER_LEG):   # columns share a y axis: expand each once
        _headroom(axes[0, j])

    _overlay_legend(fig)
    return fig


def plot_base(xyz_ocp, vx_ocp, xyz_sim, vx_sim, t_ocp):
    """Base position and forward speed: overlay on top, error trace below.

    Position is drawn as displacement from ``t=0`` for both, so the panels
    compare the motion the gait produces rather than where the model happened to
    be spawned.  The base gets its own error row because its drift is the
    headline result, and it is small enough to disappear inside the overlay.
    """
    xyz_o = xyz_ocp - xyz_ocp[:, [0]]
    xyz_s = xyz_sim - xyz_sim[:, [0]]

    titles = [r"$\Delta x$ (forward)", r"$\Delta y$ (lateral)",
              r"$\Delta z$ (heave)", r"forward speed $v_x$"]
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
    # Gap of 0.8 bar widths between legs, so the groups read as groups without
    # needing a separator line.
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
    """Display a list of matplotlib figures as tabs in a single Tk window."""
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

    # The four base panels' RMSE, on the scale of the motion that produced it:
    # the distance the simulation covers in one cycle.  SPH is the reference, so
    # it is the denominator -- the same convention the relative errors used.
    # That makes the errors comparable across the panels and between runs of
    # different stroke sizes.  Speed is normalised by the matching mean speed,
    # d/T, so its ratio is on the same scale as the position ones.
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
        # pad_inches above the default: the tight bbox under-measures usetex text.
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.12)
        print(f"  Saved {path}")

    if not args.no_show:
        show_tabbed(list(figs.values()),
                    ["Joint tracking", "Base state", "RMSE"])
    print("\nDone.")


if __name__ == "__main__":
    main()
