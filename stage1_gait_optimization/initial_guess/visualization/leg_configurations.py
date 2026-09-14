#!/usr/bin/env python3
"""
Leg-configuration stick figures at the power/recovery stroke keyframes.

Each stroke is defined by three paper-angle keyframes (initial / midpoint / end)
in ``initial_guess/paper.py`` (``_PADDLE_KEYFRAMES_DEG``); intermediate key
positions are filled in there automatically.  This script visualises those
keyframes as leg stick figures and overlays the actual Fourier-fitted foot path
that ``paper_fourier_trajectory`` produces from them — so retuning the gait in
paper.py is reflected directly here.

Front-leg joints map from the paper angles directly; hind-leg joints are solved
by IK so the hind foot traces the same hip-relative path (as in paper.py).
Left/right legs are mirror-symmetric, so only the front and hind leg are shown.

Usage:
  python leg_configurations.py
  python leg_configurations.py --gait TLPG50 --save legs.pdf
  python leg_configurations.py --front-only

``--save`` writes a vector PDF (plus a ``*_timing.pdf`` for the gait-timing
diagram); the format follows the extension you give.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401  registers the 'science' matplotlib style

# Professional thesis style with real LaTeX text rendering (Computer Modern).
plt.style.use(["science"])
plt.rcParams["text.usetex"] = True

sys.path.insert(0, str(Path(__file__).parents[2]))  # stage1_gait_optimization/
sys.path.insert(0, str(Path(__file__).parents[3] / "stage3_visualization" / "common"))
from thesis_style import PALETTE
from hydro_model import load_robot
from initial_guess.paper import (
    GAITS,
    _CALF_OFFSET_DEG,
    _LSPG_OFFSETS,
    _THIGH_OFFSET_DEG,
    _TLPG_OFFSETS,
    _grid_seed,
    _ik_leg,
    _leg_foot_xz,
    _stroke_keypoints,
    paper_fourier_trajectory,
)

ROBOT = "amph"   # registered robot name; see hydro_model/robots/


N_PER_PHASE = 6  # stick figures drawn per stroke (display density only)

PHASES = ["power", "recovery"]
LEG_TYPES = ["Front", "Hind"]  # plotted; left/right are mirror-symmetric
LEG_LABELS = ["FL", "FR", "HL", "HR"]  # phase-offset order (paper.py offsets)

# One colour per keyframe (init / mid / end); intermediates blend between them.
KEY_COLORS = [PALETTE[1], PALETTE[0], PALETTE[2]]  # green → blue → red
_KEY_FR = [0.0, 0.5, 1.0]
_KEY_RGB = np.array([mcolors.to_rgb(c) for c in KEY_COLORS])  # (3, 3)


def frame_color(f: float) -> tuple[float, float, float]:
    """Colour for a frame at stroke fraction f, blended through KEY_COLORS."""
    return tuple(np.interp(f, _KEY_FR, _KEY_RGB[:, j]) for j in range(3))


def stroke_paper_angles(phase: str, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample a stroke at n+1 fractions in [0, 1] (endpoints included).

    Returns (fractions, theta1, theta2) from paper.py's keyframes, including the
    stroke end (fraction 1.0) so the init/mid/end keyframes are all shown.
    """
    fr = np.linspace(0.0, 1.0, n + 1)
    return fr, _stroke_keypoints("theta1", phase, fr), _stroke_keypoints("theta2", phase, fr)


def leg_skeleton_xz(robot, leg: str, q_thigh: float, q_calf: float) -> np.ndarray:
    """Hip-relative (x, z) of [hip, thigh-joint, calf-joint, foot]."""
    q = robot.neutral_config()
    base = robot.n_base_q
    names = robot.actuated_joint_names
    q[base + names.index(f"{leg}_Thigh_joint")] = q_thigh
    q[base + names.index(f"{leg}_Calf_joint")] = q_calf
    robot.forward_kinematics(q)
    pos = robot.leg_centerline_positions(leg)
    pts = np.array([pos["side"], pos["thigh"], pos["calf"], pos["foot"]]) - pos["side"]
    return pts[:, [0, 2]]


def stroke_configs(robot, leg_type: str, phase: str):
    """(fractions, [(thigh, calf) rad]) for a leg type over one stroke.

    Front joints come straight from the paper angles; hind joints are solved by
    IK so the hind foot matches the front foot path (continuation-seeded).
    """
    fr, t1, t2 = stroke_paper_angles(phase, N_PER_PHASE)
    front_thigh = np.radians(t1 + _THIGH_OFFSET_DEG)
    front_calf = np.radians(_CALF_OFFSET_DEG - t2)

    if leg_type == "Front":
        return fr, list(zip(front_thigh, front_calf))

    leg = "Hind_Left"
    front_xz = [_leg_foot_xz(robot, "Front_Left", a, b) for a, b in zip(front_thigh, front_calf)]
    seed = _grid_seed(robot, leg, front_xz[0])
    configs = []
    for xz in front_xz:
        seed = _ik_leg(robot, leg, xz, seed)
        configs.append((seed[0], seed[1]))
    return fr, configs


def fourier_foot_path(robot, theta1_fn, theta2_fn, phase: str, pp: float, n: int = 160):
    """Hip-relative (x, z) foot path traced by the fitted Fourier trajectory.

    Computed from the front leg; the hind foot follows the same hip-relative
    path by construction, so the same curve applies to both rows.
    """
    t0, t1 = (0.0, pp) if phase == "power" else (pp, 1.0)
    t = np.linspace(t0, t1, n)
    thigh = np.radians(theta1_fn(t) + _THIGH_OFFSET_DEG)
    calf = np.radians(_CALF_OFFSET_DEG - theta2_fn(t))
    return np.array([leg_skeleton_xz(robot, "Front_Left", a, b)[-1] for a, b in zip(thigh, calf)])


def draw_gait_timing(ax, pp: float, offsets: np.ndarray, labels: list[str]) -> None:
    """Per-leg power/recovery phase bars over one cycle (paper fig. 5(g)-(i)).

    Each leg's power phase starts at its phase offset and lasts ``pp`` (the
    power-phase fraction), wrapping around the cycle; the rest is recovery.
    """
    n = len(labels)
    for i, off in enumerate(offsets):
        y = n - 1 - i  # leg 0 drawn on top
        ax.broken_barh([(0.0, 1.0)], (y - 0.4, 0.8),
                       facecolors="#D9EAF3", edgecolors="0.6", lw=0.8)
        start = off % 1.0
        end = start + pp
        segs = [(start, pp)] if end <= 1.0 else [(start, 1.0 - start), (0.0, end - 1.0)]
        ax.broken_barh(segs, (y - 0.4, 0.8),
                       facecolors=PALETTE[0], edgecolors="0.3", lw=0.8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels[::-1])
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks(np.linspace(0.0, 1.0, 5))
    ax.set_xlabel("cycle fraction")
    ax.set_title("Gait timing: power (blue) vs recovery (light) per leg")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--gait", default="LSPG33", choices=list(GAITS))
    parser.add_argument("--save", type=Path, default=None)
    parser.add_argument("--front-only", action="store_true",
                        help="draw only the front-leg row (hind is identical) and "
                             "title it 'Front/Hind leg'")
    args = parser.parse_args()

    print("Loading robot…")
    robot = load_robot(ROBOT)

    pp = GAITS[args.gait]
    theta1_fn, theta2_fn = paper_fourier_trajectory(pp)
    foot_paths = {ph: fourier_foot_path(robot, theta1_fn, theta2_fn, ph, pp) for ph in PHASES}

    leg_types = ["Front"] if args.front_only else LEG_TYPES
    print("Building strokes (hind via IK)…" if not args.front_only else "Building strokes…")
    fig, axes = plt.subplots(len(leg_types), 2, figsize=(11, 5 * len(leg_types)),
                             squeeze=False)
    all_pts = [p for p in foot_paths.values()]  # for a shared, comparable range
    for row, leg_type in enumerate(leg_types):
        leg = f"{leg_type}_Left"
        for col, phase in enumerate(PHASES):
            ax = axes[row, col]
            fr, configs = stroke_configs(robot, leg_type, phase)

            path = foot_paths[phase]
            ax.plot(path[:, 0], path[:, 1], "-", color="0.4", lw=1.3, zorder=1,
                    label="Fourier-fitted foot path")

            for f, (q_thigh, q_calf) in zip(fr, configs):
                key = np.isclose(f, [0.0, 0.5, 1.0]).any()  # init / mid / end
                pts = leg_skeleton_xz(robot, leg, q_thigh, q_calf)
                all_pts.append(pts)
                color = frame_color(f)
                alpha = 1.0 if key else 0.3
                lw = 2.8 if key else 1.6
                ax.plot(pts[:, 0], pts[:, 1], "-", color=color, lw=lw, alpha=alpha,
                        marker="o", ms=3, zorder=3 if key else 2)
                ax.plot(pts[-1, 0], pts[-1, 1], "o", color=color,
                        ms=9 if key else 5, alpha=alpha, zorder=3 if key else 2)
                if key:
                    label = ["init", "mid", "end"][int(round(f * 2))]
                    ax.annotate(label, pts[-1], textcoords="offset points",
                                xytext=(6, 4), fontsize=8, color=color)

            ax.plot(0, 0, "ks", ms=7, zorder=4)  # hip
            title_leg = "Front/Hind" if args.front_only else leg_type
            ax.set_title(f"{title_leg} leg, {phase} stroke")
            ax.set_xlabel(r"$x$ [m]")
            ax.set_ylabel(r"$z$ [m]")
            ax.set_aspect("equal")
            ax.grid(alpha=0.3)

    # Shared limits so leg sizes/positions are comparable across all panels.
    stacked = np.vstack(all_pts)
    (x0, z0), (x1, z1) = stacked.min(0), stacked.max(0)
    mx, mz = 0.05 * (x1 - x0), 0.05 * (z1 - z0)
    for ax in axes.flat:
        ax.set_xlim(x0 - mx, x1 + mx)
        ax.set_ylim(z0 - mz, z1 + mz)
    axes[0, 0].legend(loc="lower left", fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.96))

    # Gait timing (power/recovery per leg) as a separate figure.
    fig_timing, ax_timing = plt.subplots(figsize=(9, 3))
    offsets = _TLPG_OFFSETS if args.gait == "TLPG50" else _LSPG_OFFSETS
    draw_gait_timing(ax_timing, pp, offsets, LEG_LABELS)
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
