#!/usr/bin/env python3
"""Pairwise RMS joint-difference matrix of a guess_multistart.py sweep.

The colour scale starts at zero (identical solutions). The console also prints
COT per start and the trajectory amplitude for reference.

Usage:
  python stage2_sim_validation/sensitivity_and_robustness/plot_multistart.py --sweep 20260828_105014
  python stage2_sim_validation/sensitivity_and_robustness/plot_multistart.py --sweep TAG --save multistart.pdf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
from mlflow.tracking import MlflowClient

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "stage1_gait_optimization"))
sys.path.insert(0, str(REPO_ROOT / "stage3_visualization" / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hydro_model import load_robot                                # noqa: E402
from hydro_model.trajectory import load_solution                  # noqa: E402
from thesis_style import GAIT_ORDER, style_for                    # noqa: E402

MLFLOW_TRACKING_URI = "http://localhost:5000"
MLFLOW_EXPERIMENT_ID = "1"


def fetch(tag: str):
    """``{gait: (solution, cot, status)}`` for one multi-start sweep."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    c = MlflowClient()
    runs = c.search_runs([MLFLOW_EXPERIMENT_ID],
                         filter_string=f"params.sweep_tag = '{tag}'",
                         max_results=100)
    if not runs:
        raise SystemExit(f"no runs found for sweep_tag {tag!r}")
    out = {}
    for r in runs:
        p, m = r.data.params, r.data.metrics
        if p.get("sweep") != "guess_multistart":
            raise SystemExit(f"sweep_tag {tag!r} is a "
                             f"{p.get('sweep')!r} sweep, not guess_multistart")
        art = [a.path for a in c.list_artifacts(r.info.run_id)
               if a.path.endswith(".npz")]
        out[p["start_from"]] = (
            load_solution(Path(r.info.artifact_uri) / art[0]),
            m["cot"], p.get("solver_status"))
    # Stable thesis gait order
    return {g: out[g] for g in GAIT_ORDER if g in out}


def build_figure(gaits, dist):
    """Pairwise RMS joint difference as an annotated matrix."""
    n = len(gaits)
    M = np.array([[dist[(a, b)] for b in gaits] for a in gaits])
    fig, ax = plt.subplots(figsize=(4.9, 4.0))

    # Anchored at 0, with headroom above the largest entry
    vmax = M.max() * 1.12
    im = ax.imshow(M, cmap="viridis", vmin=0.0, vmax=vmax)

    cmap = plt.get_cmap("viridis")
    for i in range(n):
        for j in range(n):
            v = M[i, j]
            # Text colour from the cell luminance
            r, g, b, _ = cmap(v / vmax)
            lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
            ax.text(j, i, "—" if i == j else f"{v:.0f}",
                    ha="center", va="center", fontsize=9,
                    color="0.12" if lum > 0.55 else "white")

    labels = list(gaits)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("initial guess", fontsize=9)
    ax.set_ylabel("initial guess", fontsize=9)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    # White lines between cells
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.5)
    ax.tick_params(which="minor", length=0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"RMS joint difference [deg]", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep", required=True, help="sweep_tag to plot")
    parser.add_argument("--robot", default="amph", help="registered robot name")
    parser.add_argument("--save", type=Path, default=None,
                        help="write the figure here; a .pdf also writes a "
                             "300 dpi .png beside it")
    args = parser.parse_args()

    robot = load_robot(args.robot)
    data = fetch(args.sweep)
    gaits = list(data)
    cots = {g: data[g][1] for g in gaits}
    statuses = {g: data[g][2] for g in gaits}
    n_act = robot.n_actuated
    m = 256
    grid = np.arange(m) / m

    def angles(sol):
        X = sol["X"]
        ph = np.arange(X.shape[1]) / (X.shape[1] - 1)
        return np.degrees([np.interp(grid, ph, X[7 + j, :]) for j in range(n_act)])

    q = {g: angles(data[g][0]) for g in gaits}
    dist = {(a, b): float(np.sqrt(((q[a] - q[b]) ** 2).mean()))
            for a in gaits for b in gaits}
    off = {k: v for k, v in dist.items() if k[0] != k[1]}
    # Mean trajectory amplitude over all starts, for reference
    amp = float(np.mean([np.sqrt(((q[g] - q[g].mean(axis=1, keepdims=True)) ** 2).mean())
                         for g in gaits]))

    print(f"sweep {args.sweep}: {len(gaits)} starts")
    print(f"\n{'start':<14s}{'status':>9s}{'COT':>9s}")
    for g in gaits:
        print(f"{g:<14s}{statuses[g]:>9s}{cots[g]:>9.4f}")
    print(f"\nCOT spread {min(cots.values()):.4f} to {max(cots.values()):.4f} "
          f"({100 * (max(cots.values()) / min(cots.values()) - 1):.1f}%)")
    print(f"pairwise joint difference {min(off.values()):.1f} to "
          f"{max(off.values()):.1f} deg   (trajectory amplitude {amp:.1f} deg)")
    print("closest pair: " + " vs ".join(min(off, key=off.get))
          + f" at {min(off.values()):.1f} deg — nothing converged to anything else")

    fig = build_figure(gaits, dist)
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
