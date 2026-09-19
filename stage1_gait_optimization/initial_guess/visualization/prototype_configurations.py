#!/usr/bin/env python3
"""Stick figures of the firmware (prototype) swim gait over one cycle.

Draws the front and hind leg coloured by phase with the foot path, plus a
per-leg gait-timing diagram. Gait parameters are resolved the same way as in
initial_guess/firmware.py. With --save, the timing diagram goes to
``<name>_timing.<ext>``.

Usage:
  python stage1_gait_optimization/initial_guess/visualization/prototype_configurations.py
  python stage1_gait_optimization/initial_guess/visualization/prototype_configurations.py --save proto.pdf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401  registers the 'science' matplotlib style

plt.style.use(["science"])
plt.rcParams["text.usetex"] = True

sys.path.insert(0, str(Path(__file__).parents[2]))
sys.path.insert(0, str(Path(__file__).parents[3] / "stage3_visualization" / "common"))
from thesis_style import PALETTE
from hydro_model import SymbolicDynamics, get_spec, load_robot
from initial_guess import firmware

ROBOT = "amph"   # registered robot name; see hydro_model/robots/


# --- Gait parameters (resolved as in firmware.py) ---
_D = {**firmware._DEFAULT_GAIT, **get_spec(ROBOT).firmware_gait}
R_REC, R_STR, R_POW, R_LIFT = _D["ratio_recovery"], _D["ratio_strike"], _D["ratio_power"], _D["ratio_lift"]
STROKE_LEN = _D["stroke_len"]
STAND_H, DEPTH_SURF, DEPTH_DEEP = _D["stand_h"], _D["depth_surface"], _D["depth_deep"]
CENTER_X_FRONT, CENTER_X_REAR = _D["center_x_front"], _D["center_x_rear"]
DIAG_OFFSET = -R_REC / 2.0  # builder default: FR/HL lag by half a recovery phase

# Deeper = more negative dz from trim
DZ_SURF = -(DEPTH_SURF - STAND_H)
DZ_DEEP = -(DEPTH_DEEP - STAND_H)

RATIOS = [R_REC, R_STR, R_POW, R_LIFT]
BOUNDS = np.concatenate([[0.0], np.cumsum(RATIOS)])  # phase edges in [0, 1]
PHASES = ["recovery", "strike", "power", "lift"]
PHASE_COLORS = {
    "recovery": PALETTE[1],  # green
    "strike":   PALETTE[3],  # pink
    "power":    PALETTE[2],  # red
    "lift":     PALETTE[0],  # blue
}

LEG_LABELS = ["FL", "FR", "HL", "HR"]  # firmware leg order
PHASE_OFFSETS = [0.0, DIAG_OFFSET, DIAG_OFFSET, 0.0]
# (title, leg name, leg index, stroke centre x)
DISPLAY_LEGS = [
    ("Front leg", "Front_Left", 0, CENTER_X_FRONT),
    ("Hind leg", "Hind_Left", 2, CENTER_X_REAR),
]

N_TOTAL_FRAMES = 24  # stick figures per cycle


def firmware_dxz(t_norm: float, cx: float) -> np.ndarray:
    """(Δx, Δz) foot offset from trim at cycle fraction t_norm ∈ [0, 1)."""
    dx, dz = firmware._firmware_foot_target(
        t_norm, 1.0, R_REC, R_STR, R_POW,
        cx + STROKE_LEN, cx - STROKE_LEN, DZ_SURF, DZ_DEEP,
    )
    return np.array([dx, dz])


def phase_frames(n_total: int = N_TOTAL_FRAMES):
    """``(t, phase_idx, is_waypoint)`` over one cycle, in time order.

    Frames per phase scale with its duration; each phase's first frame is its waypoint.
    """
    frames = []
    for pi in range(4):
        s, e = BOUNDS[pi], BOUNDS[pi + 1]
        cnt = max(2, round(n_total * RATIOS[pi]))
        for j, t in enumerate(np.linspace(s, e, cnt, endpoint=False)):
            frames.append((t, pi, j == 0))
    return frames


def leg_cycle_skeletons(robot, q_trim, p_ref, leg_idx, leg_name, cx, ts):
    """Hip-relative (x, z) of [hip, thigh, calf, foot] at each time in ``ts`` (firmware IK)."""
    js = 7 + leg_idx * 3
    q_ctx = q_trim.copy()
    out = []
    for t in ts:
        target = np.array([p_ref[0], p_ref[1], p_ref[2]]) + np.array(
            [firmware_dxz(t, cx)[0], 0.0, firmware_dxz(t, cx)[1]]
        )
        q_sol = firmware._solve_leg_ik(robot, q_ctx, leg_idx, target, q_trim)
        q_ctx[js : js + 3] = q_sol
        robot.forward_kinematics(q_ctx)
        pos = robot.leg_centerline_positions(leg_name)
        pts = np.array([pos["side"], pos["thigh"], pos["calf"], pos["foot"]]) - pos["side"]
        out.append(pts[:, [0, 2]])
    return out


def _wrap_segs(start: float, length: float):
    """broken_barh segments of a bar wrapped around [0, 1)."""
    start %= 1.0
    end = start + length
    if end <= 1.0:
        return [(start, length)]
    return [(start, 1.0 - start), (0.0, end - 1.0)]


def draw_phase_timing(ax, offsets, labels) -> None:
    """Per-leg 4-phase timing bars over one cycle (cf. paddling-gait fig. 5)."""
    n = len(labels)
    for i, off in enumerate(offsets):
        y = n - 1 - i  # leg 0 drawn on top
        start = (-off) % 1.0  # start of this leg's recovery
        for ratio, ph in zip(RATIOS, PHASES):
            for s, length in _wrap_segs(start, ratio):
                ax.broken_barh([(s, length)], (y - 0.4, 0.8),
                               facecolors=PHASE_COLORS[ph], edgecolors="0.3", lw=0.6)
            start += ratio
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels[::-1])
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks(np.linspace(0.0, 1.0, 5))
    ax.set_xlabel("cycle fraction")
    ax.set_title("Prototype gait timing: 4 phases per leg")
    handles = [mpatches.Patch(color=PHASE_COLORS[ph], label=ph) for ph in PHASES]
    ax.legend(handles=handles, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, -0.28), fontsize=8, frameon=False)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--save", type=Path, default=None)
    args = parser.parse_args()

    print("Loading robot and dynamics…")
    robot = load_robot(ROBOT)
    dyn = SymbolicDynamics(robot)
    q_trim = dyn.find_trim_state()
    robot.forward_kinematics(q_trim)
    trim_feet = robot.foot_positions()

    frames = phase_frames()
    ts = [t for t, _, _ in frames]

    print("Building strokes (firmware IK)…")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    all_pts = []
    for ax, (title, leg_name, leg_idx, cx) in zip(axes, DISPLAY_LEGS):
        p_ref = trim_feet[leg_name]

        # Hip-relative foot loop
        robot.forward_kinematics(q_trim)
        pos0 = robot.leg_centerline_positions(leg_name)
        offset_xz = (pos0["foot"] - pos0["side"])[[0, 2]]
        tl = np.linspace(0.0, 1.0, 200, endpoint=False)
        loop = np.array([offset_xz + firmware_dxz(t, cx) for t in tl])
        loop = np.vstack([loop, loop[0]])  # close the cycle
        ax.plot(loop[:, 0], loop[:, 1], "-", color="0.4", lw=1.2, zorder=1,
                label="firmware foot path")
        all_pts.append(loop)

        for (t, pi, key), pts in zip(frames, leg_cycle_skeletons(
            robot, q_trim, p_ref, leg_idx, leg_name, cx, ts
        )):
            all_pts.append(pts)
            color = PHASE_COLORS[PHASES[pi]]
            alpha = 1.0 if key else 0.28
            lw = 2.8 if key else 1.5
            ax.plot(pts[:, 0], pts[:, 1], "-", color=color, lw=lw, alpha=alpha,
                    marker="o", ms=3, zorder=3 if key else 2)
            ax.plot(pts[-1, 0], pts[-1, 1], "o", color=color,
                    ms=9 if key else 4, alpha=alpha, zorder=3 if key else 2)
            if key:
                ax.annotate(PHASES[pi], pts[-1], textcoords="offset points",
                            xytext=(6, 4), fontsize=8, color=color)

        ax.plot(0, 0, "ks", ms=7, zorder=4)  # hip
        ax.set_title(title)
        ax.set_xlabel(r"$x$ [m]")
        ax.set_ylabel(r"$z$ [m]")
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)

    stacked = np.vstack(all_pts)
    (x0, z0), (x1, z1) = stacked.min(0), stacked.max(0)
    mx, mz = 0.05 * (x1 - x0), 0.05 * (z1 - z0)
    for ax in axes:
        ax.set_xlim(x0 - mx, x1 + mx)
        ax.set_ylim(z0 - mz, z1 + mz)
    axes[0].legend(loc="lower left", fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.95))

    fig_timing, ax_timing = plt.subplots(figsize=(9, 3))
    draw_phase_timing(ax_timing, PHASE_OFFSETS, LEG_LABELS)
    fig_timing.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved → {args.save}")
        timing_path = args.save.with_name(f"{args.save.stem}_timing{args.save.suffix}")
        fig_timing.savefig(timing_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {timing_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
