#!/usr/bin/env python3
"""Forward-simulate the solved gait with both hind legs parked.

Uses the prescribed-joint rollout from simulate_ocp.py: the front legs follow the
solved stroke, the hind legs are held at a fixed pose, and only the base is
integrated. Compares forward travel, heave, attitude drift, hind submersion and
drag impulses for the nominal gait and three park poses (extended, tucked,
home). The hind legs cannot be lifted out of the water at any reachable pose,
so this tests "held still", not "removed".

This perturbs the solved gait; it does not compare optimised gaits (that would
need a re-solve with the hind joints pinned). A second figure
(``<name>_poses``) shows the park poses on the side view of the left legs.

Usage:
  python stage3_visualization/thrust/hind_parked_rollout.py --report
  python stage3_visualization/thrust/hind_parked_rollout.py --cycles 4 --save hind_parked.pdf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "stage1_gait_optimization"))
sys.path.insert(0, str(_ROOT / "stage2_sim_validation" / "hydro_calibration"))

from simulate_ocp import rollout  # noqa: E402

from stage1_gait_optimization.hydro_model import SymbolicDynamics, load_robot  # noqa: E402
from stage1_gait_optimization.hydro_model.trajectory import load_solution  # noqa: E402

from stage3_visualization.common.drag_model import leg_drag_x  # noqa: E402
from stage3_visualization.common.thesis_style import (PALETTE, TEXT_WIDTH_IN, LEGEND_ROW_IN, full_width,  # noqa: E402
                          legend_row)

# Thesis nominal (same as nominal_metrics.py)
_DEFAULT_SOLUTION = (_ROOT / "experiment_results" / "mlruns" / "2" /
                     "69943d2cffc94ca5b78271a7df24c546" / "artifacts" /
                     "TLPG50_v0p180_T1p400.npz")

HIND = ("Hind_Left", "Hind_Right")

# Side drawn in the side view (left and right are symmetric)
LEFT = ("Front_Left", "Hind_Left")

# Park poses (thigh, calf) [deg], same for left and right:
#   extended  thigh at upper stop, calf pointing aft
#   tucked    highest pose with the foot behind the hip
#   home      URDF zero pose
POSES = {
    "extended": (75.06, 37.64),
    "tucked": (-6.35, -43.54),
    "home": (0.0, 0.0),
}

# Several cycles show whether the attitude drifts over time.
N_CYCLES = 4

full_width()


def _joint_index(robot, leg: str, which: str) -> int:
    """Index of a leg's side/thigh/calf joint in the actuated coordinates."""
    legs = list(robot.spec.leg_names)
    per = robot.n_actuated // len(legs)
    return per * legs.index(leg) + {"side": 0, "thigh": 1, "calf": 2}[which]


def park(robot, X: np.ndarray, nq: int, thigh_deg: float, calf_deg: float) -> np.ndarray:
    """Copy of ``X`` with both hind legs at a constant pose and zero joint rate.

    Joint angles are rows ``7 + j``, joint rates rows ``nq + 6 + j``.
    """
    out = X.copy()
    for leg in HIND:
        for which, val in (("side", 0.0),
                           ("thigh", np.radians(thigh_deg)),
                           ("calf", np.radians(calf_deg))):
            j = _joint_index(robot, leg, which)
            out[7 + j, :] = val
            out[nq + 6 + j, :] = 0.0
    return out


def tile_cycles(X: np.ndarray, n_cycles: int) -> np.ndarray:
    """Repeat a periodic reference over ``n_cycles`` periods.

    The base rows of the repeats are stale but unused (only column 0 is the
    initial condition).
    """
    if n_cycles == 1:
        return X
    return np.concatenate([X[:, :-1]] * (n_cycles - 1) + [X], axis=1)


def base_traces(X: np.ndarray, nq: int):
    """World position, roll/pitch/yaw [deg] and world-frame velocity (as plot_base_motion.base_pose)."""
    pos, quat = X[0:3, :], X[3:7, :]
    rpy = np.empty_like(pos)
    vel_w = np.empty_like(pos)
    for k in range(X.shape[1]):
        R = pin.Quaternion(*np.roll(quat[:, k], 1)).matrix()
        rpy[:, k] = np.degrees(pin.rpy.matrixToRpy(R))
        vel_w[:, k] = R @ X[nq:nq + 3, k]
    return pos, rpy, vel_w


def base_stats(X: np.ndarray, T: float, nq: int) -> dict:
    """Speed, heave, attitude range and drift of a base trajectory."""
    pos, rpy, vel_w = base_traces(X, nq)
    # Horizontal path length vs net x advance separates less thrust from heading loss.
    path = float(np.linalg.norm(np.diff(pos[:2, :], axis=1), axis=0).sum())
    return {
        "v_x": (pos[0, -1] - pos[0, 0]) / T,
        "v_path": path / T,
        "vx_min": vel_w[0].min(), "vx_max": vel_w[0].max(),
        "z_mean": pos[2].mean(), "heave": np.ptp(pos[2]) * 1e3,
        "pk_roll": np.ptp(rpy[0]), "pk_pitch": np.ptp(rpy[1]), "pk_yaw": np.ptp(rpy[2]),
        "d_y": (pos[1, -1] - pos[1, 0]) * 1e3,
        "d_z": (pos[2, -1] - pos[2, 0]) * 1e3,
        "d_roll": rpy[0, -1] - rpy[0, 0],
        "d_pitch": rpy[1, -1] - rpy[1, 0],
        "d_yaw": rpy[2, -1] - rpy[2, 0],
    }


def hind_submersion(robot, X: np.ndarray, nq: int) -> float:
    """Mean submerged fraction of the hind-leg cylinders (hard-clamped submersion ratio)."""
    alphas = []
    for name, link in robot.links.items():
        if not name.startswith(HIND) or link.cylinder is None:
            continue
        cyl = link.cylinder
        for k in range(X.shape[1]):
            robot.forward_kinematics(X[:nq, k])
            oMf = robot.data.oMf[link.frame_id]
            axis = oMf.rotation @ cyl.axis_local
            axis = axis / np.linalg.norm(axis)
            z_c = (oMf.translation + oMf.rotation @ cyl.center_local)[2]
            dz = (0.5 * cyl.length * abs(axis[2])
                  + cyl.radius * np.sqrt(max(1.0 - axis[2] ** 2, 0.0)))
            alphas.append(np.clip((dz - z_c) / (2 * dz + 1e-6), 0.0, 1.0))
    return float(np.mean(alphas))


def drag_impulse(robot, X: np.ndarray, T: float, nq: int) -> dict:
    """Forward drag impulse per leg [N s], trapezoid rule on the grid nodes.

    The rollout has no collocation states, so absolute front-leg values are
    slightly biased; compare cases rather than quoting them.
    """
    dt = T / (X.shape[1] - 1)
    out = {}
    for leg in robot.spec.leg_names:
        f = np.empty(X.shape[1])
        for k in range(X.shape[1]):
            q = X[:nq, k]
            v = np.asarray(X[nq:nq + robot.nv, k], dtype=float).flatten()
            robot.forward_kinematics(q)
            f[k] = leg_drag_x(robot, leg, q, v)
        out[leg] = float(np.trapz(f, dx=dt))
    return out


def run_cases(robot, dyn, X, T, N, nq, n_cycles: int) -> dict:
    """``{label: (X_sim, stats)}`` for the nominal reference and each park pose."""
    T_t, N_t = T * n_cycles, N * n_cycles
    refs = {"nominal": X}
    refs.update({name: park(robot, X, nq, *deg) for name, deg in POSES.items()})

    out = {}
    for label, ref in refs.items():
        X_sim = rollout(tile_cycles(ref, n_cycles), dyn, T_t, N_t)
        s = base_stats(X_sim, T_t, nq)
        s["submersion"] = hind_submersion(robot, X_sim, nq)
        s["impulse"] = drag_impulse(robot, X_sim, T_t, nq)
        out[label] = (X_sim, s)
    return out


def report(cases: dict, n_cycles: int, T: float) -> None:
    """Print base statistics, drift and drag impulses per case."""
    print(f"\n===== {n_cycles} cycle(s), T_total = {T * n_cycles:.3f} s =====")
    print(f"  {'case':<10}{'v_x':>8}{'v_path':>8}{'vx min':>8}{'vx max':>8}"
          f"{'z [mm]':>8}{'heave':>8}{'roll':>7}{'pitch':>7}{'yaw':>7}{'subm.':>8}")
    for label, (_, s) in cases.items():
        print(f"  {label:<10}{s['v_x']:8.4f}{s['v_path']:8.4f}"
              f"{s['vx_min']:8.3f}{s['vx_max']:8.3f}{s['z_mean']*1e3:8.1f}"
              f"{s['heave']:8.1f}{s['pk_roll']:7.1f}{s['pk_pitch']:7.1f}"
              f"{s['pk_yaw']:7.1f}{s['submersion']*100:7.1f}%")

    print(f"\n  drift over the run (0 = returns to its start)")
    print(f"  {'case':<10}{'dy [mm]':>10}{'dz [mm]':>10}"
          f"{'d roll':>9}{'d pitch':>9}{'d yaw':>9}")
    for label, (_, s) in cases.items():
        print(f"  {label:<10}{s['d_y']:10.2f}{s['d_z']:10.2f}"
              f"{s['d_roll']:9.2f}{s['d_pitch']:9.2f}{s['d_yaw']:9.2f}")

    print(f"\n  forward drag impulse [N s]")
    print(f"  {'case':<10}{'FL':>8}{'FR':>8}{'HL':>8}{'HR':>8}"
          f"{'front':>8}{'hind':>8}{'all':>8}")
    for label, (_, s) in cases.items():
        i = s["impulse"]
        fr = i["Front_Left"] + i["Front_Right"]
        hd = i["Hind_Left"] + i["Hind_Right"]
        print(f"  {label:<10}{i['Front_Left']:+8.2f}{i['Front_Right']:+8.2f}"
              f"{i['Hind_Left']:+8.2f}{i['Hind_Right']:+8.2f}"
              f"{fr:+8.2f}{hd:+8.2f}{fr + hd:+8.2f}")


def plot_cases(cases: dict, T: float, nq: int, n_cycles: int):
    """Forward travel, yaw and roll against cycle phase, nominal against parked."""
    fig, (ax_x, ax_yaw, ax_roll) = plt.subplots(
        3, 1, figsize=(TEXT_WIDTH_IN, 4.2 + LEGEND_ROW_IN), sharex=True)

    colours = {"nominal": "0.15", "extended": PALETTE[0],
               "tucked": PALETTE[1], "home": PALETTE[2]}
    labels = {"nominal": "all four stroking", "extended": "hind parked, extended",
              "tucked": "hind parked, tucked", "home": "hind parked, home"}

    for label, (X_sim, _) in cases.items():
        pos, rpy, _ = base_traces(X_sim, nq)
        phase = np.linspace(0, n_cycles, X_sim.shape[1])
        kw = dict(color=colours[label], lw=1.2 if label == "nominal" else 1.0,
                  ls="-" if label == "nominal" else (0, (4, 1.5)))
        ax_x.plot(phase, (pos[0] - pos[0, 0]) * 1e3, label=labels[label], **kw)
        ax_yaw.plot(phase, rpy[2] - rpy[2, 0], **kw)
        ax_roll.plot(phase, rpy[0] - rpy[0, 0], **kw)

    for ax in (ax_x, ax_yaw, ax_roll):
        ax.minorticks_on()
        ax.grid(which="major", alpha=0.3)
        ax.grid(which="minor", alpha=0.12, lw=0.5)
        for c in range(1, n_cycles):
            ax.axvline(c, color="0.6", lw=0.5, ls="--")
    ax_yaw.axhline(0.0, color="0.75", lw=0.5)
    ax_roll.axhline(0.0, color="0.75", lw=0.5)

    ax_x.set_ylabel("forward travel\n[mm]")
    ax_yaw.set_ylabel("yaw drift\n[deg]")
    ax_roll.set_ylabel("roll drift\n[deg]")
    ax_roll.set_xlabel("cycle")
    ax_roll.set_xlim(0, n_cycles)
    ax_x.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0),
                columnspacing=1.4, handlelength=2.0, frameon=False,
                borderaxespad=0.2)
    fig.tight_layout(pad=0.3)
    fig.subplots_adjust(hspace=0.25)
    return fig


def leg_polyline(robot, leg: str, theta: np.ndarray) -> np.ndarray:
    """Base-frame ``(x, z)`` of a leg's joint chain from hip to foot (base at neutral)."""
    q = robot.neutral_config()
    q[robot.n_base_q:] = robot.coord_map.expand_numeric(theta)
    robot.forward_kinematics(q)
    segs = np.array(robot.leg_skeleton(leg))[:, :, [0, 2]]
    return np.vstack([segs[:, 0, :], segs[-1:, 1, :]])


def hull_outline(robot, n: int = 40) -> np.ndarray:
    """Closed base-frame ``(x, z)`` outline of the hull cylinder used by the model."""
    cyl = robot.links[robot.spec.base_link].cylinder
    cx, cz = cyl.center_local[0], cyl.center_local[2]
    half, r = 0.5 * cyl.length, cyl.radius
    a = np.linspace(-np.pi / 2, np.pi / 2, n)
    nose = np.column_stack([cx + half + r * np.cos(a), cz + r * np.sin(a)])
    tail = np.column_stack([cx - half - r * np.cos(a), cz - r * np.sin(a)])
    return np.vstack([nose, tail, nose[:1]])


def surface_lines(X: np.ndarray, x_ends: np.ndarray, y_leg: float) -> np.ndarray:
    """Base-frame height of the free surface at ``x_ends`` (a line per node).

    Solves ``t_z + R[2, :] . p_base = 0`` at fixed ``y``.
    """
    zs = []
    for k in range(X.shape[1]):
        R = pin.Quaternion(*np.roll(X[3:7, k], 1)).matrix()
        zs.append(-(X[2, k] + R[2, 0] * x_ends + R[2, 1] * y_leg) / R[2, 2])
    return np.array(zs)


def plot_side_view(robot, X_nom: np.ndarray, n_nodes: int, n_frames: int = 10):
    """Side view of the left legs' solved stroke with the park poses.

    Drawn in the base frame, rotated so the mean free surface is horizontal at z = 0.
    """
    theta = X_nom[7:7 + robot.n_actuated, :n_nodes]
    poly = {leg: np.array([leg_polyline(robot, leg, theta[:, k])
                           for k in range(n_nodes)]) for leg in LEFT}
    # Skip the last node (same as the first)
    idx = np.unique(np.linspace(0, n_nodes - 2, n_frames).round().astype(int))

    parked = {}
    for name, (thigh, calf) in POSES.items():
        th = np.zeros(robot.n_actuated)
        th[_joint_index(robot, HIND[0], "thigh")] = np.radians(thigh)
        th[_joint_index(robot, HIND[0], "calf")] = np.radians(calf)
        parked[name] = leg_polyline(robot, HIND[0], th)

    hull = hull_outline(robot)
    pts = np.vstack([hull] + [p.reshape(-1, 2) for p in poly.values()]
                    + list(parked.values()))
    x_ends = np.array([pts[:, 0].min(), pts[:, 0].max()])
    y_leg = float(np.array(robot.leg_skeleton(LEFT[0]))[0, 0, 1])
    lines = surface_lines(X_nom[:, :n_nodes], x_ends, y_leg)

    # Rigid transform that maps the mean surface line onto z = 0
    mean = lines.mean(0)
    phi = np.arctan2(mean[1] - mean[0], x_ends[1] - x_ends[0])
    c = mean[0] - np.tan(phi) * x_ends[0]
    M = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])

    def level(p):
        return (np.asarray(p) - [0.0, c]) @ M

    hull, parked = level(hull), {n: level(p) for n, p in parked.items()}
    poly = {leg: level(p) for leg, p in poly.items()}

    pts = np.vstack([hull] + [p.reshape(-1, 2) for p in poly.values()]
                    + list(parked.values()))
    (x0, z0), (x1, z1) = pts.min(0), pts.max(0)
    mx, mz = 0.10 * (x1 - x0), 0.10 * (z1 - z0)

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 3.0))

    # Band of surface lines over the cycle, extended across the panel
    ends = np.array([level(np.column_stack([x_ends, z])) for z in lines])
    x_ax = np.array([x0 - mx, x1 + mx])
    slope = ((ends[:, 1, 1] - ends[:, 0, 1])
             / (ends[:, 1, 0] - ends[:, 0, 0]))[:, None]
    band = ends[:, 0, 1][:, None] + slope * (x_ax[None, :] - ends[:, 0, 0][:, None])
    ax.fill_between(x_ax, band.min(0), band.max(0), color=PALETTE[0],
                    alpha=0.10, lw=0, zorder=0)
    ax.axhline(0.0, color=PALETTE[0], lw=0.8, alpha=0.7, zorder=0)

    ax.fill(hull[:, 0], hull[:, 1], facecolor="0.90", edgecolor="0.35",
            lw=0.9, zorder=2)

    for leg in LEFT:
        ax.plot(poly[leg][:, -1, 0], poly[leg][:, -1, 1], "-", color="0.55",
                lw=0.9, zorder=3)
        for k in idx:
            ax.plot(poly[leg][k, :, 0], poly[leg][k, :, 1], "-", color="0.15",
                    lw=0.7, alpha=0.35, marker="o", ms=1.6, zorder=3)

    for name, p in parked.items():
        colour = {"extended": PALETTE[0], "tucked": PALETTE[1],
                  "home": PALETTE[2]}[name]
        ax.plot(p[:, 0], p[:, 1], "-", color=colour, lw=1.7, marker="o",
                ms=2.6, zorder=5)
        ax.plot(p[-1, 0], p[-1, 1], "o", color=colour, ms=4.5, zorder=5)

    handles = [
        plt.Line2D([], [], color="0.15", lw=0.7, alpha=0.35, marker="o", ms=1.6,
                   label="solved stroke"),
        plt.Line2D([], [], color="0.55", lw=0.9, label="foot path"),
    ] + [plt.Line2D([], [], color=col, lw=1.7, label=f"hind parked, {n}")
         for n, col in (("extended", PALETTE[0]), ("tucked", PALETTE[1]),
                        ("home", PALETTE[2]))]
    handles.append(plt.Line2D([], [], color=PALETTE[0], lw=0.8, alpha=0.7,
                              label="free surface"))
    ax.legend(handles=handles, ncol=3, loc="lower center",
              bbox_to_anchor=(0.5, 1.0), columnspacing=1.4, handlelength=2.0,
              frameon=False, borderaxespad=0.2)

    ax.set_xlabel(r"$x$ along the hull [m]")
    ax.set_ylabel("height above the\nmean surface [m]")
    ax.set_xlim(x0 - mx, x1 + mx)
    ax.set_ylim(z0 - mz, z1 + mz)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    legend_row(fig, ax, rows=2)
    fig.tight_layout(pad=0.3)
    return fig


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--solution", type=Path, default=_DEFAULT_SOLUTION)
    ap.add_argument("--cycles", type=int, default=N_CYCLES)
    ap.add_argument("--report", action="store_true",
                    help="print the one-cycle table as well as the multi-cycle one")
    ap.add_argument("--save", type=Path, default=None)
    args = ap.parse_args()

    d = load_solution(args.solution)
    robot = load_robot(d["robot"])
    X, T, N, nq = d["X"], d["T"], d["N"], d["nq"]
    dyn = SymbolicDynamics(robot)

    print(f"{args.solution.name}: robot={d['robot']}  T={T:.4f}s  N={N}  nq={nq}")
    s_ocp = base_stats(X, T, nq)
    print(f"  OCP reference (not a rollout): v_x={s_ocp['v_x']:.4f} m/s  "
          f"heave={s_ocp['heave']:.1f} mm  "
          f"roll/pitch/yaw={s_ocp['pk_roll']:.1f}/{s_ocp['pk_pitch']:.1f}/"
          f"{s_ocp['pk_yaw']:.1f} deg")

    if args.report:
        report(run_cases(robot, dyn, X, T, N, nq, 1), 1, T)

    cases = run_cases(robot, dyn, X, T, N, nq, args.cycles)
    report(cases, args.cycles, T)

    fig = plot_cases(cases, T, nq, args.cycles)
    fig_poses = plot_side_view(robot, cases["nominal"][0], N + 1)
    if args.save:
        fig.savefig(args.save, dpi=300)
        print(f"\nSaved → {args.save}")
        poses_path = args.save.with_name(f"{args.save.stem}_poses{args.save.suffix}")
        fig_poses.savefig(poses_path, dpi=300)
        print(f"Saved → {poses_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
