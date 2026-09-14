#!/usr/bin/env python3
"""
Sensitivity check on the six hydrodynamic coefficients: how much does the fit
loss change when each one is moved across its plausible range?

Each coefficient in turn is scanned over its search range from
``sweep_hydro_params.BOUNDS`` while the other five are held at their fitted
values, and the sweep's own loss J is recorded.  Plotted as J/J*, a curve that
stays on 1 means the recorded trajectory cannot tell where in its range that
coefficient sits -- so holding it at a literature value instead of fitting it
costs the fit nothing, which is the argument for fixing it.

The ranges differ per coefficient because their physical plausible values do;
the x axis is scaled to each one's own range so the curves stay comparable.

Usage:
    python3 hydro_fit_sensitivity.py [--ocp PATH...] [--bag PATH...]
                                     [--start T...] [--theta NAME=VALUE...]
                                     [--fixed NAME...] [--n-profile INT]
                                     [--out DIR] [--format EXT]

    --theta   override a coefficient of the evaluation point; the default is
              the fitted set in hydro_model/hydro_params.py
    --fixed   coefficients held out of the fit, marked as such in the legend
              (default: Cd_a Ca_t Ca_a)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[2]))
sys.path.insert(0, str(Path(__file__).parents[2] / "stage3_visualization" / "common"))

from sweep_hydro_params import (  # noqa: E402
    BOUNDS,
    PARAM_NAMES,
    build_components,
    load_ocp,
    load_sph_bag,
    make_eval_fd,
    make_objective,
)
from stage1_gait_optimization.hydro_model import hydro_params, load_robot  # noqa: E402
from thesis_style import PALETTE  # noqa: E402  also activates the plot style

# The fitted point the scan is centred on.
THETA_FIT = np.array([
    hydro_params.CD_T, hydro_params.CD_A, hydro_params.CA_T,
    hydro_params.CA_A, hydro_params.CD_LIN_T, hydro_params.CD_LIN_A,
])

# Colour by force type, line style by direction: three hues instead of six, and
# the transverse/axial pairing is visible without reading the legend.
FAMILY = {
    "Cd_t":     (PALETTE[0], "-"),
    "Cd_a":     (PALETTE[0], "--"),
    "Ca_t":     (PALETTE[1], "-"),
    "Ca_a":     (PALETTE[1], "--"),
    "Cd_lin_t": (PALETTE[2], "-"),
    "Cd_lin_a": (PALETTE[2], "--"),
}
# Transverse flow is perpendicular to the link axis, axial is parallel to it.
LABEL = {
    "Cd_t": r"$C_{d,\perp}$", "Cd_a": r"$C_{d,\parallel}$",
    "Ca_t": r"$C_{a,\perp}$", "Ca_a": r"$C_{a,\parallel}$",
    "Cd_lin_t": r"$C_{d,\mathrm{lin},\perp}$",
    "Cd_lin_a": r"$C_{d,\mathrm{lin},\parallel}$",
}


def loss_profiles(objective, theta, ranges, n):
    """J(theta_j)/J* along each coefficient, the others held at theta*.

    Returns (J*, grids, curves) with grids[j] the scanned values of j.
    """
    J_star = objective(theta)
    grids, curves = [], []
    for j in range(len(theta)):
        grid = np.linspace(*ranges[j], n)
        vals = []
        for v in grid:
            p = theta.copy()
            p[j] = v
            vals.append(objective(p) / J_star)
        grids.append(grid)
        curves.append(np.array(vals))
        print(f"  {PARAM_NAMES[j]:<9} J/J* in "
              f"[{min(vals):.2f}, {max(vals):.2f}]")
    return J_star, grids, curves


def plot_profiles(grids, curves, ranges, theta, fixed):
    """One panel: the six profiles, with theta* marked on each."""
    fig, ax = plt.subplots(figsize=(4.8, 3.3))
    ax.axhline(1.0, color="0.6", lw=0.6, zorder=1)
    for j, name in enumerate(PARAM_NAMES):
        c, ls = FAMILY[name]
        lo, hi = ranges[j]
        xi = (grids[j] - lo) / (hi - lo)
        label = LABEL[name] + (r" (fixed)" if name in fixed else "")
        ax.plot(xi, curves[j], color=c, ls=ls, lw=1.3, label=label, zorder=3)
        ax.plot((theta[j] - lo) / (hi - lo), 1.0, marker="o", ms=3.6, color=c,
                mec="k", mew=0.4, zorder=5)
    ax.set_xlim(0, 1)
    ax.set_xlabel(r"coefficient, scaled to its search range")
    ax.set_ylabel(r"$J/J^\ast$")
    ax.grid(alpha=0.3, lw=0.4)
    ax.legend(ncol=3, fontsize=6.5, handlelength=1.8, columnspacing=1.0,
              loc="upper center", borderaxespad=0.4)
    fig.tight_layout()
    return fig


def main():
    _DEFAULT_BAG = (
        "/home/ws/experiment_results/mlruns/1"
        "/1d12383d7c0f4d31b022574db8838084/artifacts/sim_log.bag"
    )
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__.split("Usage:")[0],
    )
    parser.add_argument("--ocp", nargs="+", default=["/home/ws/task3_solution.npz"])
    parser.add_argument("--bag", nargs="+", default=[_DEFAULT_BAG])
    parser.add_argument("--start", nargs="+", type=float, default=[0.0],
                        help="Bag time [s] at OCP t=0; one value, or one per bag")
    parser.add_argument("--theta", nargs="*", default=[], metavar="NAME=VALUE",
                        help="Override a coefficient of the evaluation point")
    parser.add_argument("--fixed", nargs="*", default=["Cd_a", "Ca_t", "Ca_a"],
                        help="Coefficients held out of the fit (marked in the legend)")
    parser.add_argument("--n-profile", type=int, default=21,
                        help="Points per profile")
    parser.add_argument("--out", default=".")
    parser.add_argument("--format", default="pdf")
    args = parser.parse_args()

    if len(args.ocp) != len(args.bag):
        parser.error("--ocp and --bag are paired by position")
    starts = args.start * len(args.bag) if len(args.start) == 1 else args.start

    theta = THETA_FIT.copy()
    for item in args.theta:
        name, sep, value = item.partition("=")
        if not sep or name not in PARAM_NAMES:
            parser.error(f"--theta expects NAME=VALUE with NAME in {PARAM_NAMES}")
        theta[PARAM_NAMES.index(name)] = float(value)
    unknown = set(args.fixed) - set(PARAM_NAMES)
    if unknown:
        parser.error(f"--fixed: unknown coefficient(s) {sorted(unknown)}")

    # The scan range is the search box, widened where theta* sits outside it --
    # a coefficient fixed at a literature value need not lie in the box the
    # sweep was allowed to search.
    ranges = [(min(lo, t), max(hi, t)) for (lo, hi), t in zip(BOUNDS, theta)]
    for name, t, (lo, hi) in zip(PARAM_NAMES, theta, BOUNDS):
        if not lo <= t <= hi:
            print(f"note: {name}={t:g} is outside its search box [{lo}, {hi}]; "
                  f"scanning the widened range instead")

    robot = load_robot("amph")
    q_n = robot.neutral_config()
    robot.forward_kinematics(q_n)
    robot.build_cylinders()
    base_dyn, components = build_components(robot)
    eval_fd = make_eval_fd(base_dyn, components)

    datasets = []
    for ocp_path, bag_path, start in zip(args.ocp, args.bag, starts):
        X_ocp, _U, T, N, nq = load_ocp(ocp_path)
        t_ocp = np.linspace(0.0, T, N + 1)
        xyz_sph = load_sph_bag(bag_path, start, t_ocp)
        xyz_ref_rel = xyz_sph - xyz_sph[:, [0]]
        d_cycle = float(np.linalg.norm(xyz_ref_rel[:, -1]))
        datasets.append(dict(name=Path(ocp_path).stem, X=X_ocp, T=T, N=N, nq=nq,
                             xyz_ref_rel=xyz_ref_rel, d_cycle=d_cycle))
        print(f"{datasets[-1]['name']}: N={N}, T={T:.3f}s, start={start}, "
              f"|d|={d_cycle:.4f} m")

    print("\nEvaluation point:")
    for name, t in zip(PARAM_NAMES, theta):
        mark = "  (fixed)" if name in args.fixed else ""
        print(f"  {name:<9} = {t:.4f}{mark}")

    print("\nLoss profiles …")
    objective = make_objective(datasets, eval_fd)
    J_star, grids, curves = loss_profiles(objective, theta, ranges,
                                          args.n_profile)
    print(f"  J* = {J_star:.6f}")

    fig = plot_profiles(grids, curves, ranges, theta, set(args.fixed))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"hydro_loss_profiles.{args.format.lstrip('.')}"
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.12)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
