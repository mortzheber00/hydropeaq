#!/usr/bin/env python3
"""
Gait time-asymmetry diagnostic.

Answers the question: "if the OCP gait passes both strokes through the same
heatmap region, where does its net forward thrust come from?"

For a chosen leg, evaluates ``drag_model.py`` — the same model the OCP solved,
and the one ``thrust_heatmap.py`` maps — against the OCP's actual joint state
(q(t), v(t)) rather than a unit probe, and plots four standalone figures over
one gait cycle, sharing a time axis:

  velocity  Foot velocity         v_foot_x(t)  (signed) and |v_foot|(t), both
                                  relative to the hull and in the base frame
  thrust    Instantaneous thrust  F_drag_x(t)  on the leg's links combined
  impulse   Cumulative impulse    ∫ F_drag_x dt  (final value = net per cycle)
  area      Presented area        A(t), and the drag-relevant A|v|^2(t)

``area`` is the one that tests "the recovery stroke presents a smaller area to
the flow than the power stroke", which nothing else here measures.  For each
link it is the exact silhouette of the cylinder the drag model uses, projected
on the plane normal to the flow it sees:

    A = alpha * (2 r L |sin θ| + π r² |cos θ|),   cos θ = v̂ · â

— broadside gives the rectangle, end-on gives the end cap, and ``alpha`` is the
submerged fraction, so a link out of the water presents nothing.  The direction
v̂ is the **world-frame** velocity of the cylinder's midpoint, because that is
the flow the link actually meets (still water, hull travelling at ~0.15 m/s), and
it is the same velocity the thrust panel's drag is built from.  The green
shading stays the *kinematic*, hull-relative power stroke, as everywhere else
here; the two differ by the hull's own travel, which is the point the note in
``compute_traces`` makes.

The area alone is the quantity the hypothesis names; A|v|² is what the force
follows, and it is far more asymmetric because the foot moves faster on the
power stroke.  The cycle-mean ratio power/recovery is printed for both — that
ratio is the number the claim stands or falls on.

Sampling and quadrature are ``collocation.py``'s, so the net impulse below is
``plot_thrust_budget.py``'s cycle mean times T.  The grid nodes this used to
sample were the wrong place for the integral rather than for the state: drag
depends only on (q, v), which are genuine at a node, but a rectangle sum over
N+1 of them counts the periodic endpoint twice, which biased the per-leg
impulses by about 7%.

Power-stroke timesteps (v_foot_x < 0) are shaded green so the asymmetry
between the two halves of the cycle is visually obvious.  ``--save x.pdf``
writes ``x_velocity.pdf``, ``x_thrust.pdf``, ``x_impulse.pdf`` and
``x_area.pdf`` (with the leg name folded in under ``--leg all``); the
power/recovery speeds, impulses and areas go to stdout, and under ``--leg all``
a row summing the four legs follows them.

``--overlay`` adds a fifth figure, ``x_impulse_all.pdf``: all four legs'
cumulative impulse on one half-width axes against **cycle phase**, so it can be
read beside ``plot_thrust_attribution``'s phase-based traces.  Its power strokes
go in a four-row ribbon under the axes rather than as shading — see
``plot_impulse_overlay`` for why the shading cannot survive the merge.

Both robots are handled.  ``--leg`` takes the loaded robot's own leg names —
amph's ``Front_Left``-style names, BODY2's ``FL``/``FR``/``BL``/``BR`` — and
defaults to its first leg.  Nothing here assumes a serial leg: the links a
leg's drag and area are summed over come from ``drag_model.leg_links``, so
BODY2's six-link closed chain is summed over all six, and the states are
expanded from the solution's reduced coordinates onto the tree once, in
``main``, because FK and the frame Jacobians are tree-level.

Usage:
  python gait_diagnostics.py
  python gait_diagnostics.py --solution ../task3_solution.npz --leg Hind_Left
  python gait_diagnostics.py --leg all --save diag.pdf
  python gait_diagnostics.py --leg all --overlay --save diag.pdf
  python gait_diagnostics.py --solution body2.npz --leg BL --save diag.pdf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin
from matplotlib.patches import Patch

# resolve() first: run as "python gait_diagnostics.py" from this directory,
# __file__ is relative and parents[1] does not exist, which is what the usage
# line above asks for.
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
    """``(A, A|v|^2)`` for one link — its wetted silhouette against the flow.

    The projection is exact for a cylinder: the side wall contributes
    ``2 r L |sin θ|`` and the two end caps ``π r² |cos θ|``, with θ between the
    flow and the cylinder axis, so an end-on link still presents its cap rather
    than vanishing.  Scaled by the submersion ratio, which is what makes a leg
    lifted clear of the water contribute nothing.

    Velocity is taken at the cylinder's midpoint (``center_local`` off the link
    frame), not at the frame origin: a thigh pivoting about a nearly stationary
    hip has almost no origin velocity while sweeping a large area.  The epsilon
    on the norm only matters where the midpoint is instantaneously at rest, and
    there the direction is arbitrary but the area stays bounded between the cap
    and the broadside value — which is less misleading than a hole in the trace.
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
    """The five traces, one value per collocation sample of ``Xc_tree``.

    States are the tree's, not the solution's: a closed-chain robot solves in
    reduced coordinates, and every call below — FK, the frame Jacobians, the
    drag model — is a tree-level one.  ``main`` does that expansion once.
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
        # Foot velocity relative to the hull, in the base frame: zero the base
        # twist so only joint motion contributes, then rotate out of the world
        # axes.  The world-frame velocity would fold in the body's own 0.15 m/s
        # of forward travel, which shortens the power window by up to 18 points
        # of the cycle and answers a different question -- "is this foot pushing
        # water backwards" rather than "is this leg sweeping backwards".  The
        # kinematic definition is the one plot_solution_legs.py already uses,
        # and this is what its docstring has always claimed the two share.
        #
        # The foot point is the frame origin on amph but the blade tip on
        # BODY2, 66 mm down the last link, so the Jacobian is carried out to
        # ``foot_offsets`` the same way ``_presented_area`` carries it to a
        # cylinder midpoint.  The offset is zero wherever it does not apply.
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
    """Cumulative impulse plus the power/recovery split, printed by ``main``.

    The split used to be figure text; it is reported on stdout now that the
    figures carry no titles.  The two area ratios are the ones the "recovery
    presents a smaller area" claim is read off: above 1 means the power stroke
    presents more.

    Every sum and every mean carries the Radau weight ``w`` of its sample.  The
    collocation points are not equally spaced, so an unweighted mean over a half
    of the cycle silently reweights it — on this solution that alone moved the
    leg impulses by several percent, which is the size of the effects the panel
    is used to argue about.
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


# Green shading means the same thing in all three figures, and is the only
# encoding that no line or fill in them explains.
_POWER_PATCH = Patch(facecolor=PALETTE[1], alpha=0.10,
                     label=r"power stroke ($v_{\mathrm{foot},x} < 0$)")


def plot_impulse_overlay(phase, per_leg, legs):
    """All four legs' cumulative impulse on one axes, over a power-stroke ribbon.

    ``per_leg`` maps leg name to ``(impulse, v_foot_x)``.  The x-axis is cycle
    phase rather than time so the panel lines up with ``plot_thrust_attribution``
    's traces figure, which is phase-based; the four curves share one cycle, so
    a seconds axis would only restate T.

    The single-leg panels shade their power stroke with ``axvspan``, which cannot
    survive the merge: four legs have four different power windows, and the hind
    pair's are fragmented — the stroke grazes ``v_foot_x = 0`` two or three extra
    times per cycle — so overlaid shading would wash the axes grey and say
    nothing about which leg is which.  A four-row ribbon under the axes says it
    per leg instead, in the same colour as that leg's curve.

    Spans come from ``collocation.power_spans``, not from a scan over samples:
    the collocation points are unevenly spaced, so a crossing has to be
    interpolated in phase to land where the stroke actually turns.
    """
    colours = dict(zip(legs, LEG_COLORS))
    # Constrained rather than tight layout: on a canvas this small, with a 5:1
    # height ratio and a legend outside the axes, tight_layout declares the axes
    # incompatible and leaves the y-label hanging 0.1 in off the left edge.
    fig, (ax, ax_r) = plt.subplots(
        2, 1, figsize=HALF, sharex=True, layout="constrained",
        gridspec_kw={"height_ratios": [5, 1]})

    ax.axhline(0, color="k", lw=0.5, alpha=0.5)
    for leg in legs:
        ax.plot(phase, per_leg[leg][0], lw=1.3, color=colours[leg],
                label=leg.replace("_", " "))
    # Short label: the full integral expression is wider than a 2.94 in canvas
    # can spare, and the standalone impulse panel already carries it in full.
    ax.set_ylabel(r"impulse [N$\cdot$s]")
    ax.grid(alpha=0.3)

    for row, leg in enumerate(legs):
        spans = power_spans(phase, per_leg[leg][1])
        # Spans may wrap past 1; draw the tail at the front so the ribbon reads
        # as one cycle rather than running off the axis.
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
    # Buy the legend rows on the canvas.  legend_row is not used: it pins a
    # single axes south, which on this two-row grid is the wrong axes.
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
    """The four diagnostics as standalone figures, keyed by name.

    Separate rather than stacked so each can stand on its own in the text; they
    keep the shared time axis, which is all the stack really bought.
    """
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

    # ── Foot velocities ──────────────────────────────────────────────────────
    ax = panel("velocity")
    ax.plot(t_arr, v_foot_x, "C0-", lw=2, label=r"$v_{\mathrm{foot},x}$ (signed)")
    ax.plot(t_arr, v_foot_mag, "C1--", lw=1.5, label=r"$|v_{\mathrm{foot}}|$ (magnitude)")
    ax.set_ylabel("Foot velocity [m/s]")
    ax.legend(handles=ax.get_legend_handles_labels()[0] + [_POWER_PATCH],
              loc="upper right")

    # ── Instantaneous drag-thrust ────────────────────────────────────────────
    ax = panel("thrust")
    # C4, not C2: C2 is the red of the anti-thrust fill.
    ax.plot(t_arr, F_drag_x, "C4-", lw=2)
    ax.fill_between(t_arr, 0, F_drag_x, where=F_drag_x > 0,
                     alpha=0.35, color=PALETTE[1], label=r"thrust ($+x$)")
    ax.fill_between(t_arr, 0, F_drag_x, where=F_drag_x < 0,
                     alpha=0.35, color=PALETTE[2], label=r"anti-thrust ($-x$)")
    ax.set_ylabel(r"$F_{\mathrm{drag},x}$ on leg [N]")
    ax.legend(handles=ax.get_legend_handles_labels()[0] + [_POWER_PATCH],
              loc="upper right")

    # ── Cumulative impulse ───────────────────────────────────────────────────
    ax = panel("impulse")
    ax.plot(t_arr, impulse, "C3-", lw=2)
    ax.set_ylabel(r"$\int F_{\mathrm{drag},x}\,\mathrm{d}t$  [N$\cdot$s]")
    ax.legend(handles=[_POWER_PATCH], loc="upper right")

    # ── Presented area ───────────────────────────────────────────────────────
    # Two axes rather than one normalised pair: both quantities are absolute and
    # in different units, and the area in cm^2 is the number the hypothesis is
    # about, so it should be readable off the axis rather than as a ratio.
    ax = panel("area")
    ax.plot(t_arr, area * 1e4, "C0-", lw=2, label=r"$A$ (wetted, projected)")
    ax.set_ylabel(r"presented area $A$ [cm$^2$]", color="C0")
    ax.tick_params(axis="y", labelcolor="C0")
    ax2 = ax.twinx()
    # C3, not the C1 the velocity panel uses for its second curve: C1 is green
    # in this style and would sit on top of the green power-stroke shading.
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
    # No choices=: the leg names are the robot's, and which robot this is only
    # becomes known when the solution loads.  Default is its first leg, which
    # on amph is the Front_Left this always defaulted to.
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
    # Reduced coordinates for a closed-chain robot; the identity for a serial
    # one, whose array is returned untouched.
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
        # half_width() is an rcParams update, so it is scoped: the standalone
        # panels above are full-width figures and must keep their own type.
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
