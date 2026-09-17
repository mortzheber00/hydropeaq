#!/usr/bin/env python3
"""Foot paths of the hydro sensitivity sweep, one panel per coefficient.

Each panel shows the nominal hip-relative foot path and the +-25 % solutions,
with markers every 1/8 cycle to expose timing changes. Symmetry is off in these
solves, so the legs differ; the console prints the deviation of every leg.

Usage:
  python stage2_sim_validation/sensitivity_and_robustness/plot_hydro_gaits.py --sweep 20260828_081024
  python stage2_sim_validation/sensitivity_and_robustness/plot_hydro_gaits.py --sweep TAG --leg Hind_Left --save gaits.pdf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "stage1_gait_optimization"))
sys.path.insert(0, str(REPO_ROOT / "stage3_visualization" / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hydro_model import load_robot                                # noqa: E402
from initial_guess.paper import _leg_foot_xz                      # noqa: E402
from plot_hydro_sensitivity import COLOR_DOWN, COLOR_UP, LABELS, fetch  # noqa: E402

PHASE_MARKS = 8          # markers per cycle
COLOR_NOMINAL = "0.25"


def foot_path(sol, robot, leg: str) -> np.ndarray:
    """Hip-relative (x, z) foot positions over the cycle, ``(N+1, 2)``."""
    X = sol["X"]
    names = robot.actuated_joint_names
    i = names.index(f"{leg}_Thigh_joint")
    j = names.index(f"{leg}_Calf_joint")
    return np.array([_leg_foot_xz(robot, leg, X[7 + i, k], X[7 + j, k])
                     for k in range(X.shape[1])])


def path_rmse(path: np.ndarray, nominal: np.ndarray) -> float:
    """RMS foot-position deviation from nominal at matching phases [m]."""
    return float(np.sqrt((np.linalg.norm(path - nominal, axis=1) ** 2).mean()))


def _corner(ax, text):
    """Badge in the top-right corner (same style as validate_sim.py)."""
    ax.annotate(text, xy=(0.975, 0.94), xycoords="axes fraction",
                ha="right", va="top", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.75",
                          lw=0.5, alpha=0.9))


def build_figure(paths, order, leg):
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.3))  # sized for the ~1.8:1 data aspect
    every = max(1, (len(paths[(order[0], -0.25)]) - 1) // PHASE_MARKS)

    # Shared limits so the panels have the same scale
    allpts = np.vstack([p for p in paths.values()] + [paths["nominal"]]) * 100
    pad = 0.08 * max(np.ptp(allpts[:, 0]), np.ptp(allpts[:, 1]))
    xlim = (allpts[:, 0].min() - pad, allpts[:, 0].max() + pad)
    # Extra headroom for the badge
    ylim = (allpts[:, 1].min() - pad,
            allpts[:, 1].max() + pad + 0.30 * np.ptp(allpts[:, 1]))

    nom = paths["nominal"] * 100
    for ax, coef in zip(axes.ravel(), order):
        ax.plot(nom[:, 0], nom[:, 1], "-", lw=1.6, color=COLOR_NOMINAL, zorder=2)
        ax.plot(nom[::every, 0], nom[::every, 1], ".", ms=4.5,
                color=COLOR_NOMINAL, zorder=4)
        for factor, colour in ((-0.25, COLOR_DOWN), (0.25, COLOR_UP)):
            p = paths[(coef, factor)] * 100
            ax.plot(p[:, 0], p[:, 1], "-", lw=1.3, color=colour, zorder=3)
            ax.plot(p[::every, 0], p[::every, 1], ".", ms=4.5, color=colour,
                    zorder=4)
        # Path deviation per direction (COT is shown in the tornado instead)
        e_lo = 1000 * path_rmse(paths[(coef, -0.25)], paths["nominal"])
        e_hi = 1000 * path_rmse(paths[(coef, 0.25)], paths["nominal"])
        ax.set_title(LABELS.get(coef, coef), fontsize=10)
        _corner(ax, f"rmse vs nominal\n"
                    rf"$-25\%$: {e_lo:.1f} mm" "\n"
                    rf"$+25\%$: {e_hi:.1f} mm")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.set_axisbelow(True)

    for ax in axes[-1, :]:
        ax.set_xlabel(r"hip-relative $x$ [cm]")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"hip-relative $z$ [cm]")
    axes[0, 0].legend(handles=[
        plt.Line2D([], [], color=COLOR_NOMINAL, lw=1.6, label="nominal"),
        plt.Line2D([], [], color=COLOR_DOWN, lw=1.3, label=r"$-25\%$"),
        plt.Line2D([], [], color=COLOR_UP, lw=1.3, label=r"$+25\%$"),
    ], loc="upper left", fontsize=7.5, framealpha=0.92, handlelength=1.4)
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep", required=True, help="sweep_tag to plot")
    parser.add_argument("--robot", default="amph", help="registered robot name")
    parser.add_argument("--leg", default="Front_Left",
                        help="leg drawn in the panels; the console reports all")
    parser.add_argument("--save", type=Path, default=None,
                        help="write the figure here; a .pdf also writes a "
                             "300 dpi .png beside it")
    args = parser.parse_args()

    robot = load_robot(args.robot)
    if args.leg not in robot.spec.leg_names:
        raise SystemExit(f"unknown leg {args.leg!r}; "
                         f"{args.robot} has {list(robot.spec.leg_names)}")
    (base, params), data = fetch(args.sweep)
    cot0 = base["cot"]
    swing = {c: sum(abs(100 * (data[c][f]["cot"] / cot0 - 1)) for f in (-0.25, 0.25))
             for c in data}
    order = sorted(data, key=lambda c: swing[c], reverse=True)

    paths = {"nominal": foot_path(base["sol"], robot, args.leg)}
    for coef in order:
        for factor in (-0.25, 0.25):
            paths[(coef, factor)] = foot_path(data[coef][factor]["sol"],
                                              robot, args.leg)

    size = float(np.linalg.norm(np.ptp(paths["nominal"], axis=0)))
    print(f"sweep {args.sweep}: N={params.get('N')}, panels show {args.leg}, "
          f"nominal path {np.ptp(paths['nominal'], axis=0)[0] * 100:.1f} x "
          f"{np.ptp(paths['nominal'], axis=0)[1] * 100:.1f} cm")
    legs = robot.spec.leg_names
    nom_all = {l: foot_path(base["sol"], robot, l) for l in legs}
    print("\nfoot-path rmse from nominal [mm], per leg — the panels show one, "
          "quote the range")
    print(f"{'case':<14s}" + "".join(f"{l:>14s}" for l in legs))
    for coef in order:
        for factor in (-0.25, 0.25):
            row = []
            for l in legs:
                f = foot_path(data[coef][factor]["sol"], robot, l)
                row.append(1000 * path_rmse(f, nom_all[l]))
            print(f"{coef}{factor:+.0%}".ljust(14)
                  + "".join(f"{v:>14.2f}" for v in row))

    def shape(f):
        d = np.linalg.norm(np.diff(f, axis=0), axis=1)
        x, z = f[:, 0], f[:, 1]
        area = abs(0.5 * np.sum(x[:-1] * z[1:] - x[1:] * z[:-1]))
        return float(d.sum()), float(area), float(np.ptp(x)), float(np.ptp(z))

    p0, a0, rx0, rz0 = shape(paths["nominal"])
    print(f"\n{args.leg} stroke shape, change from nominal "
          f"(reach {rx0 * 100:.1f} cm, depth {rz0 * 100:.1f} cm, "
          f"perimeter {p0 * 100:.1f} cm)")
    print(f"{'case':<14s}{'reach':>10s}{'depth':>10s}{'perimeter':>12s}{'area':>10s}")
    for coef in order:
        for factor in (-0.25, 0.25):
            pe, ar, rx, rz = shape(paths[(coef, factor)])
            print(f"{coef}{factor:+.0%}".ljust(14)
                  + f"{100 * (rx / rx0 - 1):>+9.1f}%{100 * (rz / rz0 - 1):>+9.1f}%"
                    f"{100 * (pe / p0 - 1):>+11.1f}%{100 * (ar / a0 - 1):>+9.1f}%")

    fig = build_figure(paths, order, args.leg)
    if args.save:
        # Extra padding: the tight bbox under-measures usetex text.
        fig.savefig(args.save, dpi=300, bbox_inches="tight", pad_inches=0.15)
        print(f"\nSaved → {args.save}")
        if args.save.suffix == ".pdf":
            png = args.save.with_suffix(".png")
            fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.15)
            print(f"Saved → {png}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
