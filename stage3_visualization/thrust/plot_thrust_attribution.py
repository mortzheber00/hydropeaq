#!/usr/bin/env python3
"""Per-link attribution of the world-frame forward force (drag and added mass).

``*_traces``      forward force per segment type (summed over legs) vs phase
``*_means``       cycle-mean drag and added-mass force per link
``*_submersion``  submersion ratio of each link over the cycle

Per-link terms are built exactly like ``SymbolicHydrodynamicModel.build``
(strip-integrated drag, J^T projection), so they sum to the whole-robot totals
of plot_thrust_budget.py; this is checked.

Usage:
  python stage3_visualization/thrust/plot_thrust_attribution.py
  python stage3_visualization/thrust/plot_thrust_attribution.py --solution task3_solution.npz --save attr.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin
import pinocchio.casadi as cpin

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "stage1_gait_optimization"))
from stage3_visualization.common.collocation import D_COLLOC, cycle_mean, solution_states  # noqa: E402
from stage3_visualization.common.thesis_style import (  # noqa: E402
    PALETTE,
    TEXT_WIDTH_IN,
    full_width,
    legend_row,
)

from stage1_gait_optimization.hydro_model import SymbolicDynamics, load_robot  # noqa: E402
from stage1_gait_optimization.hydro_model.hydrodynamics import (  # noqa: E402
    SymbolicHydrodynamicModel,
)
from stage1_gait_optimization.hydro_model.trajectory import load_solution  # noqa: E402

# Allowed mismatch between summed per-link forces and the totals (round-off only)
SUM_TOL = 1e-9

full_width()

# Segment types in kinematic order; links are named "<Leg>_<Segment>_link".
SEGMENTS = ("Side", "Thigh", "Calf", "Foot")
SEG_C = dict(zip(SEGMENTS, (PALETTE[3], PALETTE[4], PALETTE[1], PALETTE[0])))
SEG_PLURAL = {"Side": "sides", "Thigh": "thighs", "Calf": "calves",
              "Foot": "feet"}
HULL_C = PALETTE[2]


def build_per_link_forces(robot, dyn):
    """``{link: ca.Function(q, v, a) -> ([tau_drag, tau_added], alpha)}``.

    Same terms as ``SymbolicHydrodynamicModel.build``, kept per link.
    """
    q, v, a = dyn.q, dyn.v, dyn.a
    hyd = SymbolicHydrodynamicModel(
        robot=robot, cmodel=dyn.cmodel, cdata=dyn.cdata,
        q_sym=q, v_sym=v, a_sym=a, nv=dyn.nv,
        rho=dyn.rho, Cd_transverse=dyn.Cd_t, Cd_axial=dyn.Cd_a,
        Cd_lin_transverse=dyn.Cd_lin_t, Cd_lin_axial=dyn.Cd_lin_a,
        Ca_transverse=dyn.Ca_t, Ca_axial=dyn.Ca_a,
        z_surface=dyn.z_surface, v_linear_threshold=dyn.v_linear_threshold,
        leg_thrust_scale=dyn.leg_thrust_scale,
    )
    cpin.forwardKinematics(dyn.cmodel, dyn.cdata, q, v, a)
    cpin.updateFramePlacements(dyn.cmodel, dyn.cdata)

    out = {}
    for link in robot.links.values():
        cyl = link.cylinder
        if cyl is None:
            continue
        fid = link.frame_id
        R_sym = dyn.cdata.oMf[fid].rotation
        J_full = cpin.computeFrameJacobian(
            dyn.cmodel, dyn.cdata, q, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        axis_sym = R_sym @ ca.SX(cyl.axis_local)
        axis_sym = axis_sym / (ca.norm_2(axis_sym) + 1e-15)
        alpha = hyd.submersion_ratio(cyl, axis_sym, dyn.cdata.oMf[fid], R_sym)

        scale = 1.0 if link.name == robot.spec.base_link else dyn.leg_thrust_scale
        tau_d = J_full.T @ (scale * hyd.drag_wrench(cyl, alpha, axis_sym,
                                                    J_full, R_sym))
        J_local = cpin.computeFrameJacobian(
            dyn.cmodel, dyn.cdata, q, fid, pin.ReferenceFrame.LOCAL)
        tau_a = hyd.added_mass_force(cyl, alpha, fid, J_local)

        out[link.name] = ca.Function(f"link_{link.name}", [q, v, a],
                                     [ca.horzcat(tau_d, tau_a), alpha])
    return out


def evaluate(dyn, per_link, Xc_leg, U, N, nq, d=D_COLLOC):
    """World-x force per link and collocation point: ``(drag, added, alpha, totals)``.

    ``drag``, ``added`` and ``alpha`` map link name to ``(N*d,)`` arrays;
    ``totals`` holds the whole-robot drag and added-mass forces.
    """
    n_col = N * d
    names = list(per_link)
    drag = {n: np.zeros(n_col) for n in names}
    added = {n: np.zeros(n_col) for n in names}
    alpha = {n: np.zeros(n_col) for n in names}
    tot_d, tot_a = np.zeros(n_col), np.zeros(n_col)

    for col in range(n_col):
        k = col // d                     # U is piecewise constant per interval
        q, v = Xc_leg[:nq, col], Xc_leg[nq:, col]
        R = pin.Quaternion(*np.roll(q[3:7], 1)).matrix()
        tau_full = np.concatenate([np.zeros(6), U[:, k]])
        a = np.asarray(dyn.eval_reduced_forward_dynamics(
            q[:7], q[7:nq], v, tau_full)).flatten()

        for n in names:
            cols, al = per_link[n](q, v, a)
            cols = np.array(cols)
            drag[n][col] = (R @ cols[:3, 0])[0]
            added[n][col] = (R @ cols[:3, 1])[0]
            alpha[n][col] = float(al)

        tot_d[col] = (R @ np.array(dyn.f_tau_drag(q, v)).flatten()[:3])[0]
        tot_a[col] = (R @ np.array(dyn.f_tau_added(q, v, a)).flatten()[:3])[0]

    return drag, added, alpha, {"drag": tot_d, "added": tot_a}


def _split(name, base_link):
    """``(leg, segment)`` for a link, or ``(None, None)`` for the hull."""
    if name == base_link:
        return None, None
    for seg in SEGMENTS:
        if name.endswith(f"_{seg}_link"):
            return name[: -len(f"_{seg}_link")], seg
    return None, None


def plot_traces(drag, added, robot, phase):
    """Forward force per segment type over the cycle."""
    base = robot.spec.base_link
    total = {n: drag[n] + added[n] for n in drag}

    by_seg = {s: np.zeros(len(phase)) for s in SEGMENTS}
    for n, arr in total.items():
        _, seg = _split(n, base)
        if seg:
            by_seg[seg] += arr

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 2.7))
    # The last sample (phase 1) equals phase 0; prepend it to close the gap.
    ph = np.concatenate([[0.0], phase])
    wrap = lambda a: np.concatenate([a[-1:], a])           # noqa: E731

    ax.axhline(0.0, color="0.75", lw=0.5, zorder=1)

    ax.plot(ph, wrap(total[base]), color=HULL_C, lw=1.3, zorder=3,
            label="hull")
    for seg in SEGMENTS:
        ax.plot(ph, wrap(by_seg[seg]), color=SEG_C[seg], lw=1.0, zorder=3,
                label=SEG_PLURAL[seg])

    ax.set_xlabel("cycle phase")
    ax.set_ylabel(r"forward force $F_x^{\mathrm{world}}$ [N]")
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.grid(alpha=0.3)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=5,
              handlelength=1.6, frameon=False, columnspacing=1.3,
              borderaxespad=0.2, fontsize=8)
    legend_row(fig, ax)
    fig.tight_layout(pad=0.3)
    return fig


def plot_means(drag, added, robot, N):
    """Cycle-mean drag and added-mass force per link (same rows as the submersion figure)."""
    rows = _row_order(robot, drag)

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 4.4))
    y = np.arange(len(rows))
    d = np.array([cycle_mean(drag[n], N) for n, _, _ in rows])
    a = np.array([cycle_mean(added[n], N) for n, _, _ in rows])

    h = 0.38
    ax.barh(y - h / 2, d, height=h, color=PALETTE[2], zorder=3, label="drag")
    ax.barh(y + h / 2, a, height=h, color=PALETTE[0], zorder=3,
            label=r"added mass $M_A a + C_A v$")

    ax.set_yticks(y)
    ax.set_yticklabels(
        ["hull" if lg is None else f"{lg.replace('_', ' ')} {seg.lower()}"
         for _, seg, lg in rows], fontsize=7)
    ax.invert_yaxis()
    ax.axvline(0.0, color="0.4", lw=0.7, zorder=4)
    # Separate the hull row from the leg links
    ax.axhline(0.5, color="0.6", lw=0.6, zorder=4)
    ax.set_xlabel(r"cycle-mean forward force $\bar F_x^{\mathrm{world}}$ [N]")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
              frameon=False, fontsize=8, borderaxespad=0.2, handlelength=1.4)
    legend_row(fig, ax)
    fig.tight_layout(pad=0.3)
    return fig


def _row_order(robot, present):
    """Link rows in drawing order: hull, then segment-major, leg within it."""
    base = robot.spec.base_link
    rows = [(base, "hull", None)]
    for seg in SEGMENTS:
        for lg in robot.spec.leg_names:
            rows.append((f"{lg}_{seg}_link", seg, lg))
    return [r for r in rows if r[0] in present]


def plot_submersion(alpha, robot):
    """Submersion ratio per link over the cycle.

    Needed to read the attribution, since the robot swims at the surface.
    Collocation points are drawn as equal-width cells (small phase distortion).
    """
    rows = _row_order(robot, alpha)
    M = np.array([alpha[n] for n, _, _ in rows])

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 3.1))
    im = ax.imshow(M, aspect="auto", origin="upper", cmap="Blues",
                   vmin=0.0, vmax=1.0,
                   extent=(0.0, 1.0, len(rows) - 0.5, -0.5),
                   interpolation="nearest")
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(
        ["hull" if lg is None else f"{lg.replace('_', ' ')} {seg.lower()}"
         for _, seg, lg in rows], fontsize=6.5)
    ax.set_xlabel("cycle phase")
    ax.axhline(0.5, color="0.3", lw=0.8)
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label(r"submersion ratio $\alpha$ [-]", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.tight_layout(pad=0.3)
    return fig


def report(drag, added, alpha, totals, robot, T, N):
    """Print per-link means and RMS, the sum check and gross thrust/resistance."""
    base = robot.spec.base_link
    # RMS as well as mean: added mass is large but reactive, so it averages out.
    print(f"  {'link':<26s}{'drag mean':>10s}{'drag rms':>10s}{'added mean':>11s}"
          f"{'added rms':>10s}{'net [N]':>9s}{'mean alpha':>11s}")
    rms = lambda a: float(np.sqrt((a ** 2).mean()))        # noqa: E731
    order = sorted(drag, key=lambda n: -abs(cycle_mean(drag[n] + added[n], N)))
    for n in order:
        net = cycle_mean(drag[n] + added[n], N)
        print(f"  {n:<26s}{cycle_mean(drag[n], N):>10.4f}{rms(drag[n]):>10.4f}"
              f"{cycle_mean(added[n], N):>11.4f}{rms(added[n]):>10.4f}"
              f"{net:>9.4f}{cycle_mean(alpha[n], N):>11.4f}")

    sd = sum(cycle_mean(v, N) for v in drag.values())
    sa = sum(cycle_mean(v, N) for v in added.values())
    print(f"\n  {'SUM of links':<26s}{sd:>10.4f}{sa:>11.4f}"
          f"{sd + sa:>10.4f}")
    print(f"  {'whole-robot total':<26s}{cycle_mean(totals['drag'], N):>10.4f}"
          f"{cycle_mean(totals['added'], N):>11.4f}"
          f"{cycle_mean(totals['drag'], N) + cycle_mean(totals['added'], N):>10.4f}")
    # Report gross thrust and resistance separately; shares of the small net are meaningless.
    nets = {n: cycle_mean(drag[n] + added[n], N) for n in drag}
    thrust = sum(x for x in nets.values() if x > 0)
    resist = sum(x for x in nets.values() if x < 0)
    print(f"\n  {'gross thrust (links > 0)':<26s}{thrust:>10.4f} N")
    print(f"  {'gross resistance (< 0)':<26s}{resist:>10.4f} N")
    print(f"  {'hull share of resistance':<26s}"
          f"{nets[base] / resist * 100:>9.1f}%")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--solution", type=Path, default=_ROOT / "task3_solution.npz")
    ap.add_argument("--save", type=Path, default=None,
                    help="output stem; each figure gets its own suffixed file")
    args = ap.parse_args()

    meta = load_solution(args.solution)
    robot = load_robot(meta["robot"])
    dyn = SymbolicDynamics(robot)
    U, T, N, nq = meta["U"], meta["T"], meta["N"], meta["nq"]
    Xc_leg, phase, _ = solution_states(robot, meta, args.solution.name)

    per_link = build_per_link_forces(robot, dyn)
    drag, added, alpha, totals = evaluate(dyn, per_link, Xc_leg, U, N, nq)

    print(f"{args.solution.name}:")
    print(f"  (sampled at {N * D_COLLOC} collocation points, Radau weights)")
    report(drag, added, alpha, totals, robot, T, N)

    # Per-link parts must sum to the totals.
    for tag, parts in (("drag", drag), ("added", added)):
        err = np.abs(sum(parts.values()) - totals[tag]).max()
        if err > SUM_TOL:
            raise SystemExit(
                f"\nper-link {tag} forces sum to within {err:.3e} N of the "
                f"whole-robot total, above {SUM_TOL:.0e} — the attribution is "
                f"not of the same model the OCP solved.")

    figs = {"traces": plot_traces(drag, added, robot, phase),
            "means": plot_means(drag, added, robot, N),
            "submersion": plot_submersion(alpha, robot)}
    if args.save:
        for tag, fig in figs.items():
            path = args.save.with_name(f"{args.save.stem}_{tag}{args.save.suffix}")
            fig.savefig(path, dpi=300)
            print(f"Saved → {path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
