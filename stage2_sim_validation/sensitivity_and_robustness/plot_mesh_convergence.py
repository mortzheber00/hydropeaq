#!/usr/bin/env python3
"""
Mesh-refinement figure for the gait OCP: the evidence behind a choice of N.

Reads a continuation ladder back out of MLflow (the runs ``n_sweep_continuation``
tagged with ``ladder``) and draws the three things a grid choice has to rest on:

  (a) the worst-converging joint's trajectory at every N, so a reader can see
      the coarse grids separate and the fine ones collapse;
  (b) solution error against the finest grid on log-log axes, with the scheme's
      theoretical slope for reference — this is what distinguishes "the numbers
      stopped moving" from "the discretisation is resolved";
  (c) the quantities actually reported in the thesis, as deviation from the
      finest grid, against a tolerance band declared up front.

Energy (and so COT) is integrated on the collocation points with the Radau
weights, which needs ``Xc`` in the solution file.  Runs written before that was
saved are refused rather than silently measured a second, coarser way — mixing
two quadratures inside one convergence figure would make the trend an artefact.

The tolerance band is an argument, not an observation: pick ``--tolerance``
before looking at the plot, or the figure justifies whatever N it happens to
land on.

Usage:
  python plot_mesh_convergence.py --ladder 20260827_185136
  python plot_mesh_convergence.py --ladder TAG1 TAG2 --select 48 --tolerance 5
  python plot_mesh_convergence.py --ladder TAG --save ../docs/figures/mesh_study.pdf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pinocchio as pin
from matplotlib.lines import Line2D
from mlflow.tracking import MlflowClient

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "stage1_gait_optimization"))
sys.path.insert(0, str(REPO_ROOT / "stage3_visualization" / "common"))

from hydro_model import load_robot                              # noqa: E402
from hydro_model.trajectory import load_solution                # noqa: E402
from ocp_common import collocation_coefficients                 # noqa: E402
from thesis_style import HALF, PALETTE, half_width              # noqa: E402

# The two figures go side by side, so both take the shared half-width canvas
# and are saved uncropped: same page size, same scale in LaTeX.
half_width()

MLFLOW_TRACKING_URI = "http://localhost:5000"
MLFLOW_EXPERIMENT_ID = "1"
GRAVITY = 9.81
PHASE_SAMPLES = 256          # common grid for comparing trajectories across N

# N is ordered, so it gets a sequential ramp (one hue, light -> dark) with the
# values spelled out in the legend — not the categorical palette, which encodes
# identity.  The four quantities in panel (c) are unordered, so they do.
N_RAMP = plt.get_cmap("Blues")
N_RAMP_RANGE = (0.38, 0.95)


def fetch_ladder(tags):
    """Load every rung of the named ladder(s).

    Returns ``({N: solution}, {N: timing})``.  The timing comes from the run's
    logged metrics, not the artifact, because it is a property of the solve
    rather than of the trajectory.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    c = MlflowClient()
    out, meta, seen = {}, {}, {}
    for tag in tags:
        runs = c.search_runs([MLFLOW_EXPERIMENT_ID],
                             filter_string=f"params.ladder = '{tag}'",
                             max_results=100)
        if not runs:
            raise SystemExit(f"no runs found for ladder {tag!r}")
        for r in runs:
            n = int(r.data.params["N"])
            if n in seen:
                raise SystemExit(
                    f"N={n} appears in both ladder {seen[n]!r} and {tag!r}; "
                    f"pass only ladders that partition the grid sizes"
                )
            art = [a.path for a in c.list_artifacts(r.info.run_id)
                   if a.path.endswith("_solution.npz")]
            if not art:
                raise SystemExit(f"run {r.info.run_name} has no solution artifact")
            sol = load_solution(Path(r.info.artifact_uri) / art[0])
            if "Xc" not in sol:
                raise SystemExit(
                    f"run {r.info.run_name} (N={n}) has no Xc, so its energy "
                    f"cannot be integrated the way the objective was. Re-run "
                    f"that rung; do not mix quadratures in one figure."
                )
            seen[n] = tag
            out[n] = sol
            m = r.data.metrics
            meta[n] = {"wall": m.get("wall_time_s", float("nan")),
                       "cpu": m.get("cpu_time_s", float("nan")),
                       "iters": m.get("iterations", float("nan"))}
    return dict(sorted(out.items())), dict(sorted(meta.items()))


def joint_angles(sol, samples: int) -> np.ndarray:
    """Joint angles resampled onto a common cycle-phase grid, ``(n_theta, samples)``."""
    X = sol["X"]
    n = X.shape[1] - 1
    ph = np.arange(n + 1) / n
    g = np.arange(samples) / samples
    return np.array([np.interp(g, ph, X[7 + j, :]) for j in range(sol["n_theta"])])


def metrics(sol, robot, B, d) -> dict:
    """Reported quantities, with energy on the collocation quadrature."""
    X, U, Xc, T, n = sol["X"], sol["U"], sol["Xc"], sol["T"], sol["U"].shape[1]
    n_act, nv = robot.n_actuated, robot.nv_reduced
    v_j = slice(12 + n_act, 2 * nv)          # tangent-state joint velocities
    vc, dt = Xc[v_j, :], T / n
    energy = float(sum(
        B[i] * dt * np.sum(np.abs(U[:, k] * vc[:, k * d + i]))
        for k in range(n) for i in range(d)
    ))
    forward = float(X[0, -1] - X[0, 0])
    mass = pin.computeTotalMass(robot.model)
    z = X[2, :]
    # Base pitch straight off the quaternion (x, y, z, w): asin(-R[2,0]).
    qx, qy, qz, qw = X[3, :], X[4, :], X[5, :], X[6, :]
    pitch = np.arcsin(np.clip(2.0 * (qw * qy - qx * qz), -1.0, 1.0))
    # Every entry is an integral or an RMS over the cycle.  Peak quantities are
    # deliberately excluded: tau_max sits on a torque spike at a velocity-limit
    # junction and grows with every refinement (1.79 -> 2.18 over N = 16..96),
    # so it measures how well a mesh resolves a near-discontinuity rather than
    # converging to anything.  Peak-to-peak heave has the same weakness in
    # milder form.  Both belong in the results table, not in a convergence test.
    return {
        "COT": energy / (mass * GRAVITY * abs(forward)),
        "tau_rms": float(np.sqrt((U ** 2).mean())),
        "heave_rms": float(np.sqrt(((z - z.mean()) ** 2).mean())),
        "pitch_rms": float(np.degrees(np.sqrt(((pitch - pitch.mean()) ** 2).mean()))),
    }


# Plain keys travel through the code and the console table; LaTeX only reaches
# the figure, where usetex can render it.
TRAJ = "trajectory"      # console table only; kept off the figure

LABELS = {
    "COT": "COT",
    "tau_rms": r"$\tau_{\mathrm{rms}}$",
    "heave_rms": "heave rms",
    "pitch_rms": "pitch rms",
}

# Colour follows the quantity, never its position in the list.  Enumerating the
# series and taking PALETTE[i] repaints every survivor whenever one is added or
# dropped — removing mean-square power moved tau_rms from red to green and
# heave rms from pink to red, so the same quantity had two colours across two
# drafts of the same figure.  Slots are reserved here, including for quantities
# not currently plotted, so adding one back disturbs nothing.
COLORS = {
    "COT": 0,          # blue
    "tau_rms": 1,      # green
    "heave_rms": 2,    # red
    "pitch_rms": 3,    # pink
    "power": 4,        # dark purple — reserved, off the figure at present
    "vx_ripple": 4,    # shares the spare slot; never plotted alongside power
}


def figure_quantities(Ns, quants, select, tol):
    """Deviation of each reported quantity from the finest grid, against N.

    The reference point is dropped: its deviation is zero by construction, and
    its true error is unknowable without a grid finer still.

    No axes title — these are exported one per file for \includegraphics, so the
    LaTeX caption carries the description and a title inside the PDF would
    duplicate it at a different size and font.

    Two other panels were tried and cut.  A trajectory overlay: the controls
    stay visibly noisy at every N, so it read as clutter rather than evidence.
    A solution-error curve (RMS joint difference vs N): one series, five points,
    and measured against a reference that is itself unconverged — a relative
    difference between two unresolved grids, which looks like an error bound and
    is not one.  Those numbers belong in the text; the honest verification
    measure is an ODE residual, which needs no reference grid and is not
    implemented here.
    """
    ref_N = Ns[-1]
    ns = np.array([n for n in Ns if n != ref_N], dtype=float)
    fig, ax = plt.subplots(figsize=HALF)

    names = list(next(iter(quants.values())).keys())
    devs = {name: np.array([100 * (quants[int(n)][name] / quants[ref_N][name] - 1)
                            for n in ns]) for name in names}

    ax.axhspan(-tol, tol, color="0.88", zorder=0, lw=0)
    ax.axhline(0.0, color="0.55", lw=0.8, zorder=1)
    for name in names:
        ax.plot(ns, devs[name], "-o", ms=3.2, lw=1.2,
                color=PALETTE[COLORS[name]], zorder=3)
    # Direct labels in the left margin, replacing a legend box.  Placed by value
    # at the coarsest grid and pushed apart where they would overlap.
    lo_all = min(dv.min() for dv in devs.values())
    hi_all = max(dv.max() for dv in devs.values())
    gap = 0.075 * (hi_all - lo_all)
    placed = []
    for i in sorted(range(len(names)), key=lambda i: devs[names[i]][0]):
        y = devs[names[i]][0]
        if placed and y - placed[-1] < gap:
            y = placed[-1] + gap
        placed.append(y)
        ax.annotate(LABELS[names[i]], xy=(ns[0], y), xytext=(-5, 0),
                    textcoords="offset points", fontsize=8,
                    color=PALETTE[COLORS[names[i]]], va="center", ha="right")

    if select is not None:
        ax.axvline(select, color="0.35", lw=0.9, zorder=2)
        ax.annotate(rf"selected $N={select}$", xy=(select, 1.0),
                    xycoords=("data", "axes fraction"), xytext=(0, -9),
                    textcoords="offset points", fontsize=7.5, color="0.25",
                    ha="center", va="top",
                    bbox=dict(fc="white", ec="none", pad=1.2))
    span = ns[-1] - ns[0]
    ax.set_xticks(ns)
    ax.set_xticklabels([f"{int(n)}" for n in ns])
    ax.minorticks_off()
    # The left margin holds the direct labels, so it is sized for the longest of
    # them ("heave rms" at 8 pt) at the half-width canvas, not for the data.
    ax.set_xlim(ns[0] - 0.40 * span, ns[-1] + 0.08 * span)
    lo = min(min(dv.min() for dv in devs.values()), -tol)
    hi = max(max(dv.max() for dv in devs.values()), tol)
    pad = 0.12 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("collocation intervals $N$")
    ax.set_ylabel(rf"deviation from $N={ref_N}$ [\%]")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def figure_cost(Ns, meta, select):
    """Wall time per iteration against N, with a fitted power law.

    Per iteration, not total: under continuation the total is set by how good
    the warm start was (N=64 solved in 31 iterations and finished faster than
    N=48's 70), so it measures the ladder, not the mesh.  Cost per iteration
    depends only on problem size.  Every rung is here, including the reference,
    which has no convergence datum but does have a cost.

    The fitted exponent is printed by main() rather than drawn on the axes; it
    belongs in the caption.
    """
    n_all = np.array(Ns, dtype=float)
    wpi = np.array([meta[n]["wall"] / meta[n]["iters"] for n in Ns])
    k, b = np.polyfit(np.log(n_all), np.log(wpi), 1)
    fig, ax = plt.subplots(figsize=HALF)
    ax.plot(n_all, np.exp(b) * n_all ** k, ls="--", lw=1.0, color="0.45", zorder=2)
    ax.plot(n_all, wpi, "o", ms=4.0, color=PALETTE[0], zorder=3)
    ax.set_xlabel("collocation intervals $N$")
    ax.set_ylabel("wall time per iteration [s]")
    ax.set_xticks(n_all)
    ax.set_xticklabels([f"{int(n)}" for n in Ns])
    ax.minorticks_off()
    sp = n_all[-1] - n_all[0]
    ax.set_xlim(n_all[0] - 0.10 * sp, n_all[-1] + 0.10 * sp)
    ax.set_ylim(0.0, wpi.max() * 1.15)
    if select is not None:
        ax.axvline(select, color="0.35", lw=0.9, zorder=1)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ladder", nargs="+", required=True,
                        help="MLflow ladder tag(s) to pull the rungs from")
    parser.add_argument("--robot", default="amph", help="registered robot name")
    parser.add_argument("--select", type=int, default=None,
                        help="mark this N in panel (c) as the chosen grid")
    parser.add_argument("--tolerance", type=float, default=5.0,
                        help="tolerance band in %% for panel (c); choose it "
                             "before looking at the figure")
    parser.add_argument("--save", type=Path, default=None,
                        help="write the figures here; the stem gains "
                             "_quantities and _cost, one file each, and a .pdf "
                             "also writes a 300 dpi .png beside it")
    args = parser.parse_args()

    robot = load_robot(args.robot)
    sols, meta = fetch_ladder(args.ladder)
    Ns = list(sols)
    if len(Ns) < 3:
        raise SystemExit(f"need at least 3 rungs to show a trend, found {Ns}")
    d = robot.spec.ocp.d_colloc
    _, _, _, B = collocation_coefficients(d)
    ref_N = Ns[-1]

    angles_all = {n: joint_angles(sols[n], PHASE_SAMPLES) for n in Ns}
    quants = {n: metrics(sols[n], robot, B, d) for n in Ns}


    errs = {n: float(np.degrees(np.sqrt(((angles_all[n] - angles_all[ref_N]) ** 2).mean())))
            for n in Ns}
    ref_traj = angles_all[ref_N]
    amp = float(np.degrees(np.sqrt(
        ((ref_traj - ref_traj.mean(axis=1, keepdims=True)) ** 2).mean())))
    print(f"reference trajectory rms amplitude: {amp:.2f} deg")

    print(f"ladder {args.ladder}: N = {Ns}, reference N={ref_N}, d={d} "
          f"(theoretical order {2 * d - 1})")
    U_ref = sols[ref_N]["U"]
    j = int(np.unravel_index(np.abs(U_ref).argmax(), U_ref.shape)[0])
    print(f"peak-torque joint on the finest grid: {robot.actuated_joint_names[j]} "
          f"(|tau|max {np.abs(U_ref[j]).max():.4f} Nm) — the junction that limits this study")
    # The trajectory error is a criterion like the others, so it is in the table
    # and in the band test, not only on the figure.
    traj_pct = {n: 100 * errs[n] / amp for n in Ns}
    cols = list(quants[ref_N]) + [TRAJ]
    print("\n   N   RMS diff [deg]" + "".join(f"{k:>14s}" for k in cols))
    for n in Ns:
        row = "".join(f"{100 * (quants[n][k] / quants[ref_N][k] - 1):>+13.2f}%"
                      for k in quants[n]) + f"{traj_pct[n]:>+13.2f}%"
        print(f"  {n:>3d}   {errs[n]:>13.3f}{row}")

    inside = [n for n in Ns
              if all(abs(100 * (quants[n][k] / quants[ref_N][k] - 1)) <= args.tolerance
                     for k in quants[n])
              and traj_pct[n] <= args.tolerance]
    print(f"\ninside the +-{args.tolerance:g}% band on every quantity: "
          f"{inside if inside else 'none'}")
    wpi = np.array([meta[n]["wall"] / meta[n]["iters"] for n in Ns])
    k = np.polyfit(np.log(np.array(Ns, float)), np.log(wpi), 1)[0]
    print(f"cost per iteration: {wpi[0]:.2f} s at N={Ns[0]} to {wpi[-1]:.2f} s at "
          f"N={Ns[-1]}, scaling as N^{k:.2f}")

    figs = {
        "quantities": figure_quantities(Ns, quants, args.select, args.tolerance),
        "cost": figure_cost(Ns, meta, args.select),
    }
    if args.save:
        for tag, fig in figs.items():
            # One file per figure: \includegraphics wants them separate, and a
            # LaTeX caption per float beats a title baked into the PDF.
            out = args.save.with_name(f"{args.save.stem}_{tag}{args.save.suffix}")
            # Page = the shared canvas, identical for both figures; see the
            # savefig.bbox override in thesis_style.half_width().
            fig.savefig(out, dpi=300)
            print(f"Saved → {out}")
            if out.suffix == ".pdf":
                png = out.with_suffix(".png")
                fig.savefig(png, dpi=300)
                print(f"Saved → {png}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
