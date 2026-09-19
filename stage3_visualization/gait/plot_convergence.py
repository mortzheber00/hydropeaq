#!/usr/bin/env python3
"""IPOPT convergence (objective, primal and dual infeasibility) per initial guess.

Reads the per-iteration traces logged to MLflow by
``ocp_common._log_solver_stats``; by default the newest run per gait. Note that
IPOPT reports inf_pr unscaled and inf_du scaled, so the tolerance line is exact
only for the dual panel. A warning is printed if the runs solved different
problems (weights, N, ...).

Usage:
  python stage3_visualization/gait/plot_convergence.py
  python stage3_visualization/gait/plot_convergence.py --robot amph --save ocp_convergence.pdf
  python stage3_visualization/gait/plot_convergence.py --all-runs --gaits LSPG25 Prototype
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from mlflow.tracking import MlflowClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from stage3_visualization.common.thesis_style import style_for, tex  # also activates the shared plot style

DEFAULT_TRACKING_URI = "http://localhost:5000"
DEFAULT_EXPERIMENT = "gait_ocp"

# Per-iteration keys written by ocp_common._log_solver_stats.
OBJ_KEY, PR_KEY, DU_KEY = "convergence_obj", "inf_pr", "inf_du"

# Params that must match for the objectives to be comparable
COMPARABLE_PARAMS = (
    "robot", "N", "D_COLLOC", "V_TARGET", "T_MIN", "T_MAX",
    "W_POWER", "W_DIST", "W_VEL_SMOOTH", "W_DRIFT",
)


def _history(client: MlflowClient, run_id: str, key: str) -> np.ndarray:
    """``(steps, values)`` of a metric, sorted by step, last value per step."""
    by_step = {m.step: m.value for m in client.get_metric_history(run_id, key)}
    steps = sorted(by_step)
    return np.array(steps), np.array([by_step[s] for s in steps])


def fetch_traces(client: MlflowClient, experiment: str, *, robot=None,
                 gaits=None, run_ids=None, all_runs=False) -> list[dict]:
    """Runs with a solver trace, newest first (only the newest per gait unless ``all_runs``)."""
    exp = client.get_experiment_by_name(experiment)
    if exp is None:
        raise SystemExit(f"no MLflow experiment named {experiment!r}")

    runs = client.search_runs([exp.experiment_id], max_results=1000,
                              order_by=["attributes.start_time DESC"])
    out, seen = [], set()
    for run in runs:
        p, tags = run.data.params, run.data.tags
        if OBJ_KEY not in run.data.metrics:
            continue
        if run_ids and run.info.run_id not in run_ids:
            continue
        # Old runs without a robot param are amph
        run_robot = p.get("robot", tags.get("robot", "amph"))
        if robot and run_robot != robot:
            continue
        gait = p.get("GAIT") or tags.get("initial gait") or "unknown"
        if gaits and gait not in gaits:
            continue
        if not all_runs and not run_ids:
            if gait in seen:
                continue
            seen.add(gait)

        it, obj = _history(client, run.info.run_id, OBJ_KEY)
        out.append({
            "gait": gait,
            "robot": run_robot,
            "name": run.info.run_name,
            "params": p,
            "status": p.get("solver_status", "unknown"),
            "wall_s": run.data.metrics.get("wall_time_s"),
            "iter": it,
            OBJ_KEY: obj,
            PR_KEY: _history(client, run.info.run_id, PR_KEY)[1],
            DU_KEY: _history(client, run.info.run_id, DU_KEY)[1],
        })
    if not out:
        raise SystemExit(f"no runs with a {OBJ_KEY} trace matched the filters")
    return out


def warn_incomparable(traces: list[dict]) -> None:
    """Print the problem parameters that differ between the selected runs."""
    differing = {
        k: {t["params"].get(k, "-") for t in traces}
        for k in COMPARABLE_PARAMS
    }
    differing = {k: v for k, v in differing.items() if len(v) > 1}
    if differing:
        print("  (warn) selected runs disagree on:")
        for k, vals in differing.items():
            print(f"    {k}: {', '.join(sorted(map(str, vals)))}")
        print("    -> the objective panel compares different problems; the "
              "infeasibility panels are still comparable.")


def _label(t: dict, repeated: set) -> str:
    bits = [f"{int(t['iter'][-1])} it."]
    if t["wall_s"] is not None:
        bits.append(f"{t['wall_s']:.0f} s")
    if t["status"] != "optimal":
        bits.append(t["status"])
    # Add the run name if a gait appears more than once
    head = tex(t["gait"])
    if t["gait"] in repeated:
        head += f" [{tex(t['name'])}]"
    return f"{head} ({', '.join(bits)})"


def plot_convergence(traces: list[dict], *, tol: float, acceptable_tol: float,
                     obj_scale: str, title: str | None):
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.4), sharex=True)
    ax_obj, ax_pr, ax_du = axes
    gaits = [t["gait"] for t in traces]
    repeated = {g for g in gaits if gaits.count(g) > 1}

    for t in traces:
        colour, marker = style_for(t["gait"])
        k = t["iter"]
        every = max(1, len(k) // 12)  # ~12 markers per curve
        common = dict(color=colour, lw=1.4, marker=marker, markersize=4.0,
                      markevery=every, markerfacecolor="none", markeredgewidth=0.8)
        ax_obj.plot(k, t[OBJ_KEY], label=_label(t, repeated), **common)
        ax_pr.plot(k, t[PR_KEY], **common)
        ax_du.plot(k, t[DU_KEY], **common)
        # Final iterate: marker if optimal, cross if the solve failed
        end_style = (dict(marker=marker, markersize=6.0)
                     if t["status"] == "optimal"
                     else dict(marker="x", markersize=7.0, markeredgewidth=1.3))
        for ax, key in ((ax_obj, OBJ_KEY), (ax_pr, PR_KEY), (ax_du, DU_KEY)):
            ax.plot(k[-1], t[key][-1], color=colour, linestyle="none", **end_style)

    # --- Objective ---
    # Log scale; symlog if the objective turns negative (distance reward), with
    # the linear window just below the smallest final magnitude.
    if obj_scale == "auto":
        finals = [abs(float(t[OBJ_KEY][-1])) for t in traces if t[OBJ_KEY][-1] != 0]
        if all((t[OBJ_KEY] > 0).all() for t in traces):
            ax_obj.set_yscale("log")
        else:
            ax_obj.set_yscale("symlog",
                              linthresh=max(1e-6, 0.5 * min(finals, default=2e-3)))
            ax_obj.axhline(0.0, color="0.55", lw=0.6, ls=":", zorder=0)
    ax_obj.set_ylabel(r"objective $f(x^k)$")
    ax_obj.legend(loc="best", title=r"initial guess", fontsize=8,
                  title_fontsize=8, frameon=True, framealpha=0.9)

    # --- Infeasibilities ---
    for ax, ylabel in (
        (ax_pr, r"primal infeas. $\|c(x^k)\|_\infty$"),
        (ax_du, r"dual infeas. $\|\nabla_x \mathcal{L}(x^k)\|_\infty$"),
    ):
        ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.axhspan(ax.get_ylim()[0], acceptable_tol, color="0.85", alpha=0.45,
                   lw=0, zorder=0,
                   label=rf"acceptable ($10^{{{np.log10(acceptable_tol):.0f}}}$)")
        ax.axhline(tol, color="0.35", lw=0.9, ls="--", zorder=1,
                   label=rf"tol $=10^{{{np.log10(tol):.0f}}}$")
    ax_pr.legend(loc="best", fontsize=8, frameon=True, framealpha=0.9)

    for ax in axes:
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_du.set_xlabel(r"IPOPT iteration $k$")
    ax_du.set_xlim(left=0)

    if title is None:
        robots = sorted({t["robot"] for t in traces})
        title = ("OCP convergence per initial guess"
                 f" ({tex(', '.join(robots))})")
    sup = fig.suptitle(title, y=0.995)
    fig.tight_layout()
    # Centre the title over the axes rather than the figure.
    box = ax_obj.get_position()
    sup.set_x(0.5 * (box.x0 + box.x1))
    return fig


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--robot", default=None,
                        help="keep only runs of this robot")
    parser.add_argument("--gaits", nargs="+", default=None,
                        help="keep only these GAIT values, in this plot order")
    parser.add_argument("--run-ids", nargs="+", default=None,
                        help="plot exactly these MLflow run ids")
    parser.add_argument("--all-runs", action="store_true",
                        help="plot every matching run, not just the newest per guess")
    parser.add_argument("--tol", type=float, default=1e-4,
                        help="IPOPT 'tol' used for the solve")
    parser.add_argument("--acceptable-tol", type=float, default=1e-3,
                        help="IPOPT 'acceptable_tol' used for the solve")
    parser.add_argument("--obj-scale", choices=("auto", "linear"), default="auto",
                        help="auto: log, or symlog if the objective changes sign")
    parser.add_argument("--title", default=None)
    parser.add_argument("--save", type=Path, default=None,
                        help="write the figure here (format from the extension)")
    args = parser.parse_args()

    client = MlflowClient(tracking_uri=args.tracking_uri)
    traces = fetch_traces(client, args.experiment, robot=args.robot,
                          gaits=args.gaits, run_ids=args.run_ids,
                          all_runs=args.all_runs)
    if args.gaits:      # honour the order the user asked for
        traces.sort(key=lambda t: args.gaits.index(t["gait"]))

    print(f"Selected {len(traces)} run(s) from {args.experiment!r}:")
    for t in traces:
        wall = "n/a" if t["wall_s"] is None else f"{t['wall_s']:.1f} s"
        print(f"  {t['gait']:<14s} {t['robot']:<6s} {t['name']:<22s} "
              f"{int(t['iter'][-1]):>5d} it  {wall:>9s}  {t['status']}")
    warn_incomparable(traces)

    fig = plot_convergence(traces, tol=args.tol,
                           acceptable_tol=args.acceptable_tol,
                           obj_scale=args.obj_scale, title=args.title)
    if args.save:
        fig.savefig(args.save, dpi=300, bbox_inches="tight")
        print(f"Saved → {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
