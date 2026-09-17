#!/usr/bin/env python3
"""Per-leg drag diagnostics over one gait cycle: where does the net thrust come from?

Evaluates the OCP drag model on the solved states and writes four figures per
leg (``<name>_{velocity,thrust,impulse,area}``):
  velocity  hull-relative foot velocity v_foot_x and |v_foot| (base frame)
  thrust    forward drag force on the leg's links
  impulse   cumulative impulse (final value = net per cycle)
  area      wetted projected area A = alpha (2rL|sin θ| + πr²|cos θ|) of the
            link cylinders against their world-frame flow, and A|v|²
Power stroke (v_foot_x < 0) is shaded green. Power/recovery averages are
printed. --overlay (with --leg all) adds ``<name>_impulse_all`` with all legs
over cycle phase. Works for amph and BODY2 (use the robot's leg names).

Usage:
  python stage3_visualization/thrust/gait_diagnostics.py
  python stage3_visualization/thrust/gait_diagnostics.py --solution task3_solution.npz --leg Hind_Left
  python stage3_visualization/thrust/gait_diagnostics.py --leg all --overlay --save diag.pdf
  python stage3_visualization/thrust/gait_diagnostics.py --solution body2.npz --leg BL --save diag.pdf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin
from matplotlib.patches import Patch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "stage1_gait_optimization"))
from stage3_visualization.common.collocation import (  # noqa: E402
    cumulative_integral,
    power_spans,
    quadrature_weights,
    solution_states,
)
from stage3_visualization.common.drag_model import (  # noqa: E402
    leg_drag_x,
    leg_links,
    link_drag_terms,
)
from stage3_visualization.common.thesis_style import (  # noqa: E402
    HALF,
    LEGEND_ROW_IN,
    LEG_COLORS,
    PALETTE,
    TEXT_WIDTH_IN,
    full_width,
    half_width,
)

from stage1_gait_optimization.hydro_model import load_robot  # noqa: E402
from stage1_gait_optimization.hydro_model.robot import QuadrupedRobot  # noqa: E402
from stage1_gait_optimization.hydro_model.trajectory import (  # noqa: E402
    expand_to_tree,
    load_solution,
)

full_width()


def _presented_area(robot: QuadrupedRobot, link_name: str, q, v):
    """Wetted projected area ``A`` of one link cylinder and ``A|v|^2``.

    Uses the velocity at the cylinder midpoint (the frame origin may barely
    move for a pivoting link) and scales by the submersion ratio.
    """
    terms = link_drag_terms(robot, link_name, q)
    if terms is None:
        return 0.0, 0.0
    cyl, R, alpha, axis, J = terms
    v_c = J[:3, :] @ v + np.cross(J[3:, :] @ v, R @ cyl.center_local)
    speed = float(np.linalg.norm(v_c))
    cos_t = abs(float((v_c / (speed + 1e-12)) @ axis))
    sin_t = np.sqrt(max(1.0 - cos_t ** 2, 0.0))
    area = alpha * (cyl.cross_section_transverse * sin_t
                    + cyl.cross_section_axial * cos_t)
    return area, area * speed ** 2


def compute_traces(robot: QuadrupedRobot, leg: str, Xc_tree: np.ndarray):
    """``(v_foot_x, v_foot_mag, F_drag_x, area, area_v2)`` per collocation sample.

    ``Xc_tree`` must already be expanded to tree coordinates.
    """
    foot_fid = robot.foot_frame_ids[leg]
    foot_offset = robot.foot_offsets[leg]
    links = leg_links(robot, leg)
    nq, nv = robot.nq, robot.nv
    N1 = Xc_tree.shape[1]
    F_drag_x = np.zeros(N1)
    v_foot_x = np.zeros(N1)
    v_foot_mag = np.zeros(N1)
    area = np.zeros(N1)
    area_v2 = np.zeros(N1)
    for t in range(N1):
        q = Xc_tree[:nq, t]
        v = Xc_tree[nq : nq + nv, t]
        robot.forward_kinematics(q)
        F_drag_x[t] = leg_drag_x(robot, leg, q, v)
        per_link = [_presented_area(robot, name, q, v) for name in links]
        area[t] = sum(a for a, _ in per_link)
        area_v2[t] = sum(a for _, a in per_link)
        J = pin.computeFrameJacobian(
            robot.model, robot.data, q, foot_fid,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        # Hull-relative foot velocity in the base frame (base twist zeroed), as in
        # plot_solution_legs.py. The Jacobian is shifted to the foot offset
        # (non-zero for BODY2's blade tip).
        v_rel = np.asarray(v, dtype=float).copy()
        v_rel[:6] = 0.0
        r_tip = np.array(robot.data.oMf[foot_fid].rotation) @ foot_offset
        v_foot = np.array(robot.data.oMi[1].rotation).T @ (
            J[:3, :] @ v_rel + np.cross(J[3:, :] @ v_rel, r_tip))
        v_foot_x[t] = v_foot[0]
        v_foot_mag[t] = float(np.linalg.norm(v_foot))
    return v_foot_x, v_foot_mag, F_drag_x, area, area_v2


def _shade_power(ax, t_arr, v_foot_x):
    """Shade contiguous intervals where v_foot_x < 0 (power stroke)."""
    in_power = v_foot_x < 0
    start = None
    for i, p in enumerate(in_power):
        if p and start is None:
            start = t_arr[i]
        elif not p and start is not None:
            ax.axvspan(start, t_arr[i], alpha=0.10, color=PALETTE[1], zorder=0)
            start = None
    if start is not None:
        ax.axvspan(start, t_arr[-1], alpha=0.10, color=PALETTE[1], zorder=0)


def stroke_stats(w, N, T, v_foot_x, v_foot_mag, F_drag_x, area, area_v2):
    """Cumulative impulse and Radau-weighted power/recovery statistics.

    Area ratios > 1 mean the power stroke presents more area.
    """
    impulse = cumulative_integral(F_drag_x, T, N)
    power = v_foot_x < 0

    def split(a):
        return (np.average(a[power], weights=w[power]) if power.any() else 0.0,
                np.average(a[~power], weights=w[~power]) if (~power).any() else 0.0)

    A_pow, A_rec = split(area)
    Av_pow, Av_rec = split(area_v2)
    return {
        "impulse": impulse,
        "net": impulse[-1],
        "I_power": (F_drag_x * w)[power].sum(),
        "I_recovery": (F_drag_x * w)[~power].sum(),
        "s_power": split(v_foot_mag)[0],
        "s_recovery": split(v_foot_mag)[1],
        "A_power": A_pow, "A_recovery": A_rec,
        "A_ratio": A_pow / A_rec if A_rec else np.inf,
        "Av2_ratio": Av_pow / Av_rec if Av_rec else np.inf,
    }


# Legend entry for the power-stroke shading
_POWER_PATCH = Patch(facecolor=PALETTE[1], alpha=0.10,
                     label=r"power stroke ($v_{\mathrm{foot},x} < 0$)")


def plot_impulse_overlay(phase, per_leg, legs):
    """Cumulative impulse of all legs vs cycle phase, with a per-leg power-stroke ribbon.

    ``per_leg`` maps leg name to ``(impulse, v_foot_x)``. A ribbon is used
    instead of shading because the legs' power windows differ.
    """
    colours = dict(zip(legs, LEG_COLORS))
    # tight_layout fails for this layout; use constrained layout.
    fig, (ax, ax_r) = plt.subplots(
        2, 1, figsize=HALF, sharex=True, layout="constrained",
        gridspec_kw={"height_ratios": [5, 1]})

    ax.axhline(0, color="k", lw=0.5, alpha=0.5)
    for leg in legs:
        ax.plot(phase, per_leg[leg][0], lw=1.3, color=colours[leg],
                label=leg.replace("_", " "))
    ax.set_ylabel(r"impulse [N$\cdot$s]")
    ax.grid(alpha=0.3)

    for row, leg in enumerate(legs):
        spans = power_spans(phase, per_leg[leg][1])
        # Split spans that wrap past 1
        drawn = []
        for start, width in spans:
            drawn.append((start, min(width, 1.0 - start)))
            if start + width > 1.0:
                drawn.append((0.0, start + width - 1.0))
        ax_r.broken_barh(drawn, (row + 0.15, 0.7),
                         facecolor=colours[leg], edgecolor="none")
    ax_r.set_ylim(len(legs), 0)
    ax_r.set_yticks([])
    ax_r.set_xlim(0.0, 1.0)
    ax_r.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax_r.set_xlabel("cycle phase")
    ax_r.grid(False)
    ax_r.set_ylabel("power", rotation=0, ha="right", va="center", fontsize=7)
    ax_r.tick_params(labelsize=7)

    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
              frameon=False, handlelength=1.4, columnspacing=1.2,
              borderaxespad=0.2)
    # Extra height for the legend (legend_row does not fit this two-axes layout)
    w_in, h_in = fig.get_size_inches()
    fig.set_size_inches(w_in, h_in + 2 * LEGEND_ROW_IN)
    return fig


def plot_diagnostic(
    t_arr: np.ndarray,
    v_foot_x: np.ndarray,
    v_foot_mag: np.ndarray,
    F_drag_x: np.ndarray,
    impulse: np.ndarray,
    area: np.ndarray,
    area_v2: np.ndarray,
):
    """The four diagnostic figures, keyed by name."""
    figs = {}

    def panel(name):
        fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 2.6))
        _shade_power(ax, t_arr, v_foot_x)
        ax.axhline(0, color="k", lw=0.5, alpha=0.5)
        ax.set_xlim(t_arr[0], t_arr[-1])
        ax.set_xlabel("Time [s]")
        ax.grid(alpha=0.3)
        figs[name] = fig
        return ax

    # --- Foot velocity ---
    ax = panel("velocity")
    ax.plot(t_arr, v_foot_x, "C0-", lw=2, label=r"$v_{\mathrm{foot},x}$ (signed)")
    ax.plot(t_arr, v_foot_mag, "C1--", lw=1.5, label=r"$|v_{\mathrm{foot}}|$ (magnitude)")
    ax.set_ylabel("Foot velocity [m/s]")
    ax.legend(handles=ax.get_legend_handles_labels()[0] + [_POWER_PATCH],
              loc="upper right")

    # --- Drag thrust ---
    ax = panel("thrust")
    ax.plot(t_arr, F_drag_x, "C4-", lw=2)
    ax.fill_between(t_arr, 0, F_drag_x, where=F_drag_x > 0,
                     alpha=0.35, color=PALETTE[1], label=r"thrust ($+x$)")
    ax.fill_between(t_arr, 0, F_drag_x, where=F_drag_x < 0,
                     alpha=0.35, color=PALETTE[2], label=r"anti-thrust ($-x$)")
    ax.set_ylabel(r"$F_{\mathrm{drag},x}$ on leg [N]")
    ax.legend(handles=ax.get_legend_handles_labels()[0] + [_POWER_PATCH],
              loc="upper right")

    # --- Cumulative impulse ---
    ax = panel("impulse")
    ax.plot(t_arr, impulse, "C3-", lw=2)
    ax.set_ylabel(r"$\int F_{\mathrm{drag},x}\,\mathrm{d}t$  [N$\cdot$s]")
    ax.legend(handles=[_POWER_PATCH], loc="upper right")

    # --- Presented area (twin axes, different units) ---
    ax = panel("area")
    ax.plot(t_arr, area * 1e4, "C0-", lw=2, label=r"$A$ (wetted, projected)")
    ax.set_ylabel(r"presented area $A$ [cm$^2$]", color="C0")
    ax.tick_params(axis="y", labelcolor="C0")
    ax2 = ax.twinx()
    # C3 rather than green C1, which would blend with the shading
    ax2.plot(t_arr, area_v2 * 1e4, "C3--", lw=1.5,
             label=r"$A\,|v|^2$ (drag-relevant)")
    ax2.set_ylabel(r"$A\,|v|^{2}$ [cm$^2$m$^2$s$^{-2}$]", color="C3")
    ax2.tick_params(axis="y", labelcolor="C3")
    ax2.grid(False)
    ax.legend(handles=(ax.get_legend_handles_labels()[0]
                       + ax2.get_legend_handles_labels()[0] + [_POWER_PATCH]),
              loc="upper right")

    for fig in figs.values():
        fig.tight_layout()
    return figs


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--solution", type=Path,
                        default=_ROOT / "task3_solution.npz")
    # Leg names depend on the robot in the solution; default is its first leg.
    parser.add_argument("--leg", default=None)
    parser.add_argument("--save", type=Path, default=None)
    parser.add_argument(
        "--overlay", action="store_true",
        help="with --leg all, also write one half-width figure carrying all "
             "four legs' cumulative impulse on a shared cycle-phase axis")
    args = parser.parse_args()
    if args.overlay and args.leg != "all":
        raise SystemExit("--overlay needs --leg all: it is the four-leg figure.")

    print(f"Loading OCP solution from {args.solution}…")
    d = load_solution(str(args.solution))
    nq, N, T = d["nq"], d["N"], d["T"]

    print("Loading robot…")
    robot = load_robot(d["robot"], q=None)
    all_legs = list(robot.spec.leg_names)
    leg_arg = args.leg if args.leg is not None else all_legs[0]
    if leg_arg != "all" and leg_arg not in all_legs:
        raise SystemExit(
            f"{d['robot']} has legs {all_legs}, not {leg_arg!r}.")

    Xc_leg, phase, _ = solution_states(robot, d, args.solution.name)
    # FK and Jacobians need tree coordinates.
    Xc_tree = expand_to_tree(robot, Xc_leg, nq)
    t_arr = phase * T
    w, _ = quadrature_weights(N, T)

    legs = all_legs if leg_arg == "all" else [leg_arg]
    totals = {"I_power": 0.0, "I_recovery": 0.0, "net": 0.0}
    overlay = {}
    for leg in legs:
        v_foot_x, v_foot_mag, F_drag_x, area, area_v2 = compute_traces(
            robot, leg, Xc_tree)
        st = stroke_stats(w, N, T, v_foot_x, v_foot_mag, F_drag_x, area, area_v2)
        print(f"{leg}:  <|v_foot|> power {st['s_power']:.3f} m/s, "
              f"recovery {st['s_recovery']:.3f} m/s;  impulse power "
              f"{st['I_power']:+.4f}, recovery {st['I_recovery']:+.4f}, "
              f"net {st['net']:+.4f} N·s")
        print(f"{' ' * len(leg)}   <A> power {st['A_power'] * 1e4:.1f} cm², "
              f"recovery {st['A_recovery'] * 1e4:.1f} cm² "
              f"(ratio {st['A_ratio']:.2f});  A|v|² ratio {st['Av2_ratio']:.2f}")
        for k in totals:
            totals[k] += st[k]
        overlay[leg] = (st["impulse"], v_foot_x)
        figs = plot_diagnostic(t_arr, v_foot_x, v_foot_mag, F_drag_x,
                               st["impulse"], area, area_v2)
        if args.save is None:
            continue
        stem = args.save.stem + (f"_{leg}" if len(legs) > 1 else "")
        for name, fig in figs.items():
            path = args.save.with_name(f"{stem}_{name}{args.save.suffix}")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Saved → {path}")

    if len(legs) > 1:
        print(f"{'all four':12s}   impulse power {totals['I_power']:+.4f}, "
              f"recovery {totals['I_recovery']:+.4f}, "
              f"net {totals['net']:+.4f} N·s")

    if args.overlay:
        # Scope the half-width rcParams to this figure
        with plt.rc_context():
            half_width()
            fig = plot_impulse_overlay(phase, overlay, legs)
        if args.save is not None:
            path = args.save.with_name(
                f"{args.save.stem}_impulse_all{args.save.suffix}")
            fig.savefig(path, dpi=300)
            print(f"Saved → {path}")

    if args.save is None:
        plt.show()


if __name__ == "__main__":
    main()
