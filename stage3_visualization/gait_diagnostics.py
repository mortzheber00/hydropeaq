#!/usr/bin/env python3
"""
Gait time-asymmetry diagnostic.

Answers the question: "if the OCP gait passes both strokes through the same
heatmap region, where does its net forward thrust come from?"

For a chosen leg, evaluates the SAME drag model used by thrust_heatmap.py
against the OCP's actual joint state (q(t), v(t)) — not a unit probe — and
plots three stacked panels over one gait cycle:

  1. Foot velocity         v_foot_x(t)  (signed) and |v_foot|(t)
  2. Instantaneous thrust  F_drag_x(t)  on the three leg links combined
  3. Cumulative impulse    ∫ F_drag_x dt  (final value = net thrust per cycle)

Power-stroke timesteps (v_foot_x < 0) are shaded green so the asymmetry
between the two halves of the cycle is visually obvious.

Usage:
  python gait_diagnostics.py
  python gait_diagnostics.py --solution ../task3_solution.npz --leg Hind_Left
  python gait_diagnostics.py --legs all
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin
import scienceplots  # noqa: F401  registers the 'science' matplotlib style

# Professional thesis style with real LaTeX text rendering (Computer Modern).
plt.style.use(["science"])
plt.rcParams["text.usetex"] = True

sys.path.insert(0, str(Path(__file__).parents[1]))
from stage1_gait_optimization.hydro_model.robot import QuadrupedRobot
from thrust_heatmap import URDF_PATH, _link_drag_x


def compute_traces(robot: QuadrupedRobot, leg: str, X: np.ndarray, nq: int):
    foot_fid = robot.foot_frame_ids[leg]
    N1 = X.shape[1]
    F_drag_x = np.zeros(N1)
    v_foot_x = np.zeros(N1)
    v_foot_mag = np.zeros(N1)
    for t in range(N1):
        q = X[:nq, t]
        v = X[nq : nq + robot.nv, t]
        robot.forward_kinematics(q)
        F_drag_x[t] = sum(
            _link_drag_x(robot, f"{leg}_{name}_link", q, v)
            for name in ("Thigh", "Calf", "Foot")
        )
        J = pin.computeFrameJacobian(
            robot.model, robot.data, q, foot_fid,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        v_foot = J[:3, :] @ v
        v_foot_x[t] = v_foot[0]
        v_foot_mag[t] = float(np.linalg.norm(v_foot))
    return v_foot_x, v_foot_mag, F_drag_x


def _shade_power(ax, t_arr, v_foot_x):
    """Shade contiguous intervals where v_foot_x < 0 (power stroke)."""
    in_power = v_foot_x < 0
    start = None
    for i, p in enumerate(in_power):
        if p and start is None:
            start = t_arr[i]
        elif not p and start is not None:
            ax.axvspan(start, t_arr[i], alpha=0.10, color="green", zorder=0)
            start = None
    if start is not None:
        ax.axvspan(start, t_arr[-1], alpha=0.10, color="green", zorder=0)


def plot_diagnostic(
    t_arr: np.ndarray,
    v_foot_x: np.ndarray,
    v_foot_mag: np.ndarray,
    F_drag_x: np.ndarray,
    leg: str,
    save: Path | None = None,
):
    dt = t_arr[1] - t_arr[0]
    impulse = np.cumsum(F_drag_x) * dt
    net = impulse[-1]

    # Split power/recovery impulse contributions
    power_mask = v_foot_x < 0
    I_power = (F_drag_x[power_mask] * dt).sum()
    I_recovery = (F_drag_x[~power_mask] * dt).sum()

    # Mean speed in each phase
    s_power = v_foot_mag[power_mask].mean() if power_mask.any() else 0.0
    s_recovery = v_foot_mag[~power_mask].mean() if (~power_mask).any() else 0.0

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    # ── Panel 1: foot velocities ─────────────────────────────────────────────
    ax = axes[0]
    _shade_power(ax, t_arr, v_foot_x)
    ax.plot(t_arr, v_foot_x, "C0-", lw=2, label=r"$v_{\mathrm{foot},x}$ (signed)")
    ax.plot(t_arr, v_foot_mag, "C1--", lw=1.5, label=r"$|v_{\mathrm{foot}}|$ (magnitude)")
    ax.axhline(0, color="k", lw=0.5, alpha=0.5)
    ax.set_ylabel("Foot velocity [m/s]")
    ax.set_title(
        leg.replace("_", " ") + ": gait time-asymmetry diagnostic\n"
        r"green shading = power stroke ($v_{\mathrm{foot},x} < 0$);  "
        r"$\langle |v_{\mathrm{foot}}| \rangle$  "
        f"power: {s_power:.3f} m/s,  recovery: {s_recovery:.3f} m/s"
    )
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    # ── Panel 2: instantaneous drag-thrust ──────────────────────────────────
    ax = axes[1]
    _shade_power(ax, t_arr, v_foot_x)
    ax.plot(t_arr, F_drag_x, "C2-", lw=2)
    ax.fill_between(t_arr, 0, F_drag_x, where=F_drag_x > 0,
                     alpha=0.35, color="forestgreen", label=r"thrust ($+x$)")
    ax.fill_between(t_arr, 0, F_drag_x, where=F_drag_x < 0,
                     alpha=0.35, color="crimson", label=r"anti-thrust ($-x$)")
    ax.axhline(0, color="k", lw=0.5, alpha=0.5)
    ax.set_ylabel(r"$F_{\mathrm{drag},x}$ on leg [N]")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    # ── Panel 3: cumulative impulse ──────────────────────────────────────────
    ax = axes[2]
    _shade_power(ax, t_arr, v_foot_x)
    ax.plot(t_arr, impulse, "C3-", lw=2)
    ax.axhline(0, color="k", lw=0.5, alpha=0.5)
    ax.set_ylabel(r"$\int F_{\mathrm{drag},x}\,\mathrm{d}t$  [N$\cdot$s]")
    ax.set_xlabel("Time [s]")
    ax.set_title(
        rf"Power-stroke impulse: {I_power:+.4f} N$\cdot$s;  "
        rf"recovery-stroke impulse: {I_recovery:+.4f} N$\cdot$s;  "
        rf"net per cycle: {net:+.4f} N$\cdot$s"
    )
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
        print(f"Saved → {save}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--solution", type=Path,
        default=Path(__file__).parent.parent / "task3_solution.npz",
    )
    parser.add_argument(
        "--leg", default="Front_Left",
        choices=["Front_Left", "Front_Right", "Hind_Left", "Hind_Right", "all"],
    )
    parser.add_argument("--save", type=Path, default=None)
    args = parser.parse_args()

    print("Loading robot…")
    robot = QuadrupedRobot(URDF_PATH)
    robot.forward_kinematics(robot.neutral_config())
    robot.build_cylinders()

    print(f"Loading OCP solution from {args.solution}…")
    d = np.load(args.solution)
    X = d["X"]
    nq = int(d["nq"])
    N = int(d["N"])
    T = float(d["T"])
    t_arr = np.linspace(0.0, T, N + 1)

    legs = ["Front_Left", "Front_Right", "Hind_Left", "Hind_Right"] if args.leg == "all" else [args.leg]
    for leg in legs:
        v_foot_x, v_foot_mag, F_drag_x = compute_traces(robot, leg, X, nq)
        if args.save is None:
            save = None
        elif len(legs) == 1:
            save = args.save
        else:
            save = args.save.with_name(f"{args.save.stem}_{leg}{args.save.suffix}")
        plot_diagnostic(t_arr, v_foot_x, v_foot_mag, F_drag_x, leg=leg, save=save)


if __name__ == "__main__":
    main()
