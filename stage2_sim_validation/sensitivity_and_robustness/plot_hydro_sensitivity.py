#!/usr/bin/env python3
"""
Figures for the hydrodynamic-coefficient sensitivity sweep.

Two, written one per file for \includegraphics:

  <stem>_tornado     change in cost of transport when each coefficient moves
                     +-25%, sorted by total swing.
  <stem>_robustness  the same perturbations as (cost change, gait change), which
                     separates two things the tornado conflates: a coefficient
                     can move the optimal gait without moving what it costs.

Reads the runs ``hydro_sensitivity.py`` logged under one ``sweep_tag``.  Speed is
pinned by the OCP's floor in every run, so every comparison is at the same
operating point.

Bars are coloured by the *direction the coefficient moved*, not by the sign of
the effect, and that distinction carries the main result: for transverse drag
the two are opposite.  Colouring by effect would hide it.

No axes title — exported for \\includegraphics, so the LaTeX caption describes it.

Usage:
  python plot_hydro_sensitivity.py --sweep 20260828_081024
  python plot_hydro_sensitivity.py --sweep TAG --save ../docs/figures/hydro.pdf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
from matplotlib.lines import Line2D
from mlflow.tracking import MlflowClient

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "stage3_visualization"))
sys.path.insert(0, str(REPO_ROOT / "stage1_gait_optimization"))
from hydro_model.trajectory import load_solution                  # noqa: E402
from thesis_style import PALETTE                                  # noqa: E402

MLFLOW_TRACKING_URI = "http://localhost:5000"
MLFLOW_EXPERIMENT_ID = "1"

# Perturbation direction is a polarity, so the two get a warm/cool pair rather
# than two arbitrary categorical hues.
COLOR_DOWN, COLOR_UP = PALETTE[0], PALETTE[2]

# The model's suffixes are t for transverse and a for axial, which is the
# component of flow across and along a cylinder's own axis: perpendicular and
# parallel.  Written that way here, so the added-mass coefficients do not read
# as "C_{a,a}", where the two a's mean different things.
LABELS = {
    "Cd_t": r"$C_{D,\perp}$",
    "Cd_a": r"$C_{D,\parallel}$",
    "Ca_t": r"$C_{A,\perp}$",
    "Ca_a": r"$C_{A,\parallel}$",
}
# Colour follows the coefficient, fixed, so a figure that drops one does not
# repaint the others.
COEF_SLOT = {"Cd_t": 0, "Ca_t": 1, "Cd_a": 2, "Ca_a": 3}


def fetch(tag: str):
    """Return ``(nominal, {coef: {factor: case}})`` for one sweep.

    Each case carries the COT metric, the solver status and the solution itself
    — the gait-change axis needs the trajectory, which the metrics do not hold.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    c = MlflowClient()
    runs = c.search_runs([MLFLOW_EXPERIMENT_ID],
                         filter_string=f"params.sweep_tag = '{tag}'",
                         max_results=100)
    if not runs:
        raise SystemExit(f"no runs found for sweep_tag {tag!r}")
    nominal, out = None, {}
    for r in runs:
        p, m = r.data.params, r.data.metrics
        if "cot" not in m:
            raise SystemExit(f"run {r.info.run_name} has no cot metric")
        art = [a.path for a in c.list_artifacts(r.info.run_id)
               if a.path.endswith(".npz")]
        if not art:
            raise SystemExit(f"run {r.info.run_name} has no solution artifact")
        case = {"cot": m["cot"], "status": p.get("solver_status"),
                "sol": load_solution(Path(r.info.artifact_uri) / art[0])}
        if p["perturbed"] == "none":
            nominal = (case, p)
        else:
            out.setdefault(p["perturbed"], {})[float(p["perturbation"])] = case
    if nominal is None:
        raise SystemExit(f"sweep {tag!r} has no nominal run to compare against")
    return nominal, out


def build_figure(cot0, data, order):
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    ax.axvline(0.0, color="0.45", lw=1.0, zorder=2)

    for row, coef in enumerate(order):
        y = len(order) - 1 - row  # largest swing at the top
        devs = {f: 100 * (data[coef][f]["cot"] / cot0 - 1) for f in (-0.25, 0.25)}
        # Both bars share the row, growing from the centre line in opposite
        # directions — the classic form.  A coefficient whose two perturbations
        # push COT the same way (Ca_a here) puts them on the same side, so the
        # longer is drawn first and the shorter over it; the console table flags
        # that case as non-monotone rather than leaving the plot to imply it.
        same_side = devs[-0.25] * devs[0.25] > 0
        outer = max(devs.values(), key=abs)      # the longer bar's end
        for factor in sorted(devs, key=lambda f: -abs(devs[f])):
            dev, case = devs[factor], data[coef][factor]
            colour = COLOR_DOWN if factor < 0 else COLOR_UP
            ax.barh(y, dev, height=0.55, zorder=3, color=colour,
                    # A failed solve is drawn faded rather than dropped: the
                    # perturbation still happened, the optimiser just could not
                    # meet the speed floor under it.
                    alpha=1.0 if case["status"] == "optimal" else 0.35)
            # Normally each label sits at its own bar's end.  For a same-side
            # pair both bars can be a pixel wide (Ca_a: +0.20% and +0.02%), so
            # "the bar end" is the same place for both: anchor them past the
            # longer bar instead and stack them.  Which label belongs to which
            # bar is then ambiguous, deliberately — a same-side pair only occurs
            # at magnitudes the console table has already flagged as noise, and
            # colouring the text was more distracting than the ambiguity.
            x = outer if same_side else dev
            dy = (-0.22 if factor < 0 else 0.22) if same_side else 0.0
            ax.annotate(f"{dev:+.2f}\\%",
                        xy=(x, y + dy), xytext=(5 if x >= 0 else -5, 0),
                        textcoords="offset points", fontsize=7.5,
                        ha="left" if x >= 0 else "right", va="center",
                        color="0.25")

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([LABELS.get(c, c) for c in reversed(order)], fontsize=10)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlabel(r"change in cost of transport [\%]")
    span = max(abs(100 * (data[c][f]["cot"] / cot0 - 1))
               for c in order for f in (-0.25, 0.25))
    ax.set_xlim(-1.45 * span, 1.45 * span)
    ax.grid(alpha=0.3, axis="x")
    ax.set_axisbelow(True)
    ax.legend(handles=[
        Line2D([], [], marker="s", ls="", ms=7, color=COLOR_DOWN, label=r"$-25\%$"),
        Line2D([], [], marker="s", ls="", ms=7, color=COLOR_UP, label=r"$+25\%$"),
    ], loc="lower right", fontsize=8, framealpha=0.92,
        title="perturbation", title_fontsize=8)
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep", required=True, help="sweep_tag to plot")
    parser.add_argument("--save", type=Path, default=None,
                        help="write the figure here; a .pdf also writes a "
                             "300 dpi .png beside it")
    args = parser.parse_args()

    (base, params), data = fetch(args.sweep)
    cot0, status0 = base["cot"], base["status"]
    n_act = base["sol"]["U"].shape[0]
    q0 = np.degrees(base["sol"]["X"][7:7 + n_act, :])
    amp = float(np.sqrt(((q0 - q0.mean(axis=1, keepdims=True)) ** 2).mean()))
    # Sort by total swing: the tornado's shape is the ranking.
    swing = {c: sum(abs(100 * (data[c][f]["cot"] / cot0 - 1)) for f in (-0.25, 0.25))
             for c in data}
    order = sorted(data, key=lambda c: swing[c], reverse=True)

    print(f"sweep {args.sweep}: N={params.get('N')}, nominal COT {cot0:.4f} "
          f"({status0})")
    print(f"\n{'coef':<8s}{'-25%':>10s}{'+25%':>10s}{'swing':>10s}   monotone")
    for c in order:
        d_lo = 100 * (data[c][-0.25]["cot"] / cot0 - 1)
        d_hi = 100 * (data[c][0.25]["cot"] / cot0 - 1)
        mono = "yes" if d_lo * d_hi < 0 else "NO — both same sign"
        print(f"{c:<8s}{d_lo:>+9.2f}%{d_hi:>+9.2f}%{swing[c]:>9.2f}   {mono}")
    bad = [f"{c}{f:+.0%}" for c in data for f in data[c]
           if data[c][f]["status"] != "optimal"]
    if bad:
        print(f"\nnon-converged, drawn hollow: {bad}")

    print(f"\n{'coef':<8s}{'pert':>7s}{'|dCOT|':>9s}{'gait change':>14s}")
    for c in order:
        for f in (-0.25, 0.25):
            q = np.degrees(data[c][f]["sol"]["X"][7:7 + n_act, :])
            dq = 100 * float(np.sqrt(((q - q0) ** 2).mean())) / amp
            print(f"{c:<8s}{f:>+6.0%}{abs(100 * (data[c][f]['cot'] / cot0 - 1)):>8.2f}%"
                  f"{dq:>13.2f}%")

    fig = build_figure(cot0, data, order)
    if args.save:
        # pad_inches above the default: the tight bbox under-measures usetex.
        fig.savefig(args.save, dpi=300, bbox_inches="tight", pad_inches=0.15)
        print(f"Saved → {args.save}")
        if args.save.suffix == ".pdf":
            png = args.save.with_suffix(".png")
            fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.15)
            print(f"Saved → {png}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
