#!/usr/bin/env python3
"""
Gait time-asymmetry diagnostic.

Answers the question: "if the OCP gait passes both strokes through the same
heatmap region, where does its net forward thrust come from?"

For a chosen leg, evaluates the SAME drag model used by thrust_heatmap.py
against the OCP's actual joint state (q(t), v(t)) — not a unit probe — and
plots three standalone figures over one gait cycle, sharing a time axis:

  velocity  Foot velocity         v_foot_x(t)  (signed) and |v_foot|(t), both
                                  relative to the hull and in the base frame
  thrust    Instantaneous thrust  F_drag_x(t)  on the three leg links combined
  impulse   Cumulative impulse    ∫ F_drag_x dt  (final value = net per cycle)

Power-stroke timesteps (v_foot_x < 0) are shaded green so the asymmetry
between the two halves of the cycle is visually obvious.  ``--save x.pdf``
writes ``x_velocity.pdf``, ``x_thrust.pdf`` and ``x_impulse.pdf`` (with the leg
name folded in under ``--leg all``); the power/recovery speeds and impulses go
to stdout.

Usage:
  python gait_diagnostics.py
  python gait_diagnostics.py --solution ../task3_solution.npz --leg Hind_Left
  python gait_diagnostics.py --leg all --save diag.pdf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin
import scienceplots  # noqa: F401  registers the 'science' matplotlib style
from matplotlib.patches import Patch

# Professional thesis style with real LaTeX text rendering (Computer Modern).
plt.style.use(["science"])
plt.rcParams["text.usetex"] = True

sys.path.insert(0, str(Path(__file__).parents[1]))
from thrust_heatmap import _link_drag_x, _require_supported

from stage1_gait_optimization.hydro_model import load_robot
from stage1_gait_optimization.hydro_model.robot import QuadrupedRobot
from stage1_gait_optimization.hydro_model.trajectory import load_solution


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
        # Foot velocity relative to the hull, in the base frame: zero the base
        # twist so only joint motion contributes, then rotate out of the world
        # axes.  The world-frame velocity would fold in the body's own 0.15 m/s
        # of forward travel, which shortens the power window by up to 18 points
        # of the cycle and answers a different question -- "is this foot pushing
        # water backwards" rather than "is this leg sweeping backwards".  The
        # kinematic definition is the one plot_solution_legs.py already uses,
        # and this is what its docstring has always claimed the two share.
        v_rel = np.asarray(v, dtype=float).copy()
        v_rel[:6] = 0.0
        v_foot = np.array(robot.data.oMi[1].rotation).T @ (J[:3, :] @ v_rel)
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


def stroke_stats(t_arr, v_foot_x, v_foot_mag, F_drag_x):
    """Cumulative impulse plus the power/recovery split, printed by ``main``.

    The split used to be figure text; it is reported on stdout now that the
    figures carry no titles.
    """
    dt = t_arr[1] - t_arr[0]
    impulse = np.cumsum(F_drag_x) * dt
    power = v_foot_x < 0
    return {
        "impulse": impulse,
        "net": impulse[-1],
        "I_power": (F_drag_x[power] * dt).sum(),
        "I_recovery": (F_drag_x[~power] * dt).sum(),
        "s_power": v_foot_mag[power].mean() if power.any() else 0.0,
        "s_recovery": v_foot_mag[~power].mean() if (~power).any() else 0.0,
    }


# Green shading means the same thing in all three figures, and is the only
# encoding that no line or fill in them explains.
_POWER_PATCH = Patch(facecolor="green", alpha=0.10,
                     label=r"power stroke ($v_{\mathrm{foot},x} < 0$)")


def plot_diagnostic(
    t_arr: np.ndarray,
    v_foot_x: np.ndarray,
    v_foot_mag: np.ndarray,
    F_drag_x: np.ndarray,
    impulse: np.ndarray,
):
    """The three diagnostics as standalone figures, keyed by name.

    Separate rather than stacked so each can stand on its own in the text; they
    keep the shared time axis, which is all the stack really bought.
    """
    figs = {}

    def panel(name):
        fig, ax = plt.subplots(figsize=(8.8, 3.4))
        _shade_power(ax, t_arr, v_foot_x)
        ax.axhline(0, color="k", lw=0.5, alpha=0.5)
        ax.set_xlim(t_arr[0], t_arr[-1])
        ax.set_xlabel("Time [s]")
        ax.grid(alpha=0.3)
        figs[name] = fig
        return ax

    # ── Foot velocities ──────────────────────────────────────────────────────
    ax = panel("velocity")
    ax.plot(t_arr, v_foot_x, "C0-", lw=2, label=r"$v_{\mathrm{foot},x}$ (signed)")
    ax.plot(t_arr, v_foot_mag, "C1--", lw=1.5, label=r"$|v_{\mathrm{foot}}|$ (magnitude)")
    ax.set_ylabel("Foot velocity [m/s]")
    ax.legend(handles=ax.get_legend_handles_labels()[0] + [_POWER_PATCH],
              loc="upper right")

    # ── Instantaneous drag-thrust ────────────────────────────────────────────
    ax = panel("thrust")
    ax.plot(t_arr, F_drag_x, "C2-", lw=2)
    ax.fill_between(t_arr, 0, F_drag_x, where=F_drag_x > 0,
                     alpha=0.35, color="forestgreen", label=r"thrust ($+x$)")
    ax.fill_between(t_arr, 0, F_drag_x, where=F_drag_x < 0,
                     alpha=0.35, color="crimson", label=r"anti-thrust ($-x$)")
    ax.set_ylabel(r"$F_{\mathrm{drag},x}$ on leg [N]")
    ax.legend(handles=ax.get_legend_handles_labels()[0] + [_POWER_PATCH],
              loc="upper right")

    # ── Cumulative impulse ───────────────────────────────────────────────────
    ax = panel("impulse")
    ax.plot(t_arr, impulse, "C3-", lw=2)
    ax.set_ylabel(r"$\int F_{\mathrm{drag},x}\,\mathrm{d}t$  [N$\cdot$s]")
    ax.legend(handles=[_POWER_PATCH], loc="upper right")

    for fig in figs.values():
        fig.tight_layout()
    return figs


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

    print(f"Loading OCP solution from {args.solution}…")
    d = load_solution(str(args.solution))
    _require_supported(d["robot"])
    X, nq, N, T = d["X"], d["nq"], d["N"], d["T"]

    print("Loading robot…")
    robot = load_robot(d["robot"], q=None)
    t_arr = np.linspace(0.0, T, N + 1)

    legs = ["Front_Left", "Front_Right", "Hind_Left", "Hind_Right"] if args.leg == "all" else [args.leg]
    for leg in legs:
        v_foot_x, v_foot_mag, F_drag_x = compute_traces(robot, leg, X, nq)
        st = stroke_stats(t_arr, v_foot_x, v_foot_mag, F_drag_x)
        print(f"{leg}:  <|v_foot|> power {st['s_power']:.3f} m/s, "
              f"recovery {st['s_recovery']:.3f} m/s;  impulse power "
              f"{st['I_power']:+.4f}, recovery {st['I_recovery']:+.4f}, "
              f"net {st['net']:+.4f} N·s")
        figs = plot_diagnostic(t_arr, v_foot_x, v_foot_mag, F_drag_x, st["impulse"])
        if args.save is None:
            continue
        stem = args.save.stem + (f"_{leg}" if len(legs) > 1 else "")
        for name, fig in figs.items():
            path = args.save.with_name(f"{stem}_{name}{args.save.suffix}")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Saved → {path}")
    if args.save is None:
        plt.show()


if __name__ == "__main__":
    main()
