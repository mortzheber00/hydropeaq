#!/usr/bin/env python3
"""Log a codesign_results/ directory produced elsewhere (e.g. on a server) as an
MLflow run in the local tracking store.

``run_codesign.py`` logs its sweep at the end of the solve.  When the sweep ran
on another machine and only the output directory came back, this replays that
logging step from the files: the metrics are all recomputed from
``codesign_summary.json`` and the whole directory is attached as artifacts.

The sweep params that are *visible in the data* (speed targets, T grid, gaits,
number of solves) are derived from the summary.  The ones that are not — solver
settings, free-T band, git SHA — are NOT taken from this checkout, because the
local config may differ from what the server actually solved with.  Pass them in
a JSON file to record them:

    python import_codesign_run.py ~/downloads/codesign_results \\
        --params server_config.json --label server

where server_config.json holds e.g.
    {"N_collocation": 40, "tau_max": 12.0, "free_T_band": 0.2,
     "git_sha": "4f49780", "parallel": true}
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import mlflow
import numpy as np

MLFLOW_TRACKING_URI = "http://localhost:5000"
MLFLOW_EXPERIMENT = "gait_codesign"


def _derived_params(rows: list[dict]) -> dict:
    """The sweep configuration that can be read back off the results."""
    speeds = np.array(sorted({r["v_target"] for r in rows}))
    t_grid = np.array(sorted({r["t_center"] for r in rows}))
    gaits = list(dict.fromkeys(r["gait"] for r in rows))
    return {
        "speed_targets": np.array2string(speeds, precision=3),
        "t_grid": np.array2string(t_grid, precision=3),
        "gaits": ",".join(gaits),
        "n_tasks": len(rows),
    }


def _metrics(rows: list[dict]) -> dict:
    feasible = [r for r in rows if r["feasible"] and np.isfinite(r["cot"])]
    m = {
        "n_total": len(rows),
        "n_feasible": len(feasible),
        "frac_feasible": len(feasible) / len(rows) if rows else 0.0,
        "n_pareto": sum(r["pareto"] for r in rows),
    }
    if feasible:
        m["min_cot"] = min(r["cot"] for r in feasible)
        m["max_speed"] = max(r["speed"] for r in feasible)
    m["total_iterations"] = sum(r["iterations"] for r in rows)
    m["total_solve_wall_s"] = sum(r["wall_time_s"] for r in rows)
    warm = [r for r in rows if r["warm_started"]]
    cold = [r for r in rows if not r["warm_started"]]
    for label, group in (("warm", warm), ("cold", cold)):
        if group:
            m[f"mean_iters_{label}"] = float(np.mean([r["iterations"] for r in group]))
            m[f"mean_wall_s_{label}"] = float(np.mean([r["wall_time_s"] for r in group]))
    for v in sorted({r["v_target"] for r in rows}):
        at_v = [r["cot"] for r in feasible if abs(r["v_target"] - v) < 1e-9]
        if at_v:
            m[f"best_cot_v{float(v):.2f}"] = min(at_v)
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dir", type=Path,
                    help="downloaded codesign_results/ directory")
    ap.add_argument("--params", type=Path,
                    help="JSON of extra params (solver settings, git_sha, ...)")
    ap.add_argument("--label", default="", help="tag appended to the run name")
    ap.add_argument("--run-name", help="override the run name entirely")
    args = ap.parse_args()

    summary_path = args.results_dir / "codesign_summary.json"
    rows = json.loads(summary_path.read_text())

    params = _derived_params(rows)
    if args.params:
        extra = json.loads(args.params.read_text())
        # null means "I don't know what the server used" — leave it unlogged
        # rather than recording a wrong value.
        params.update({k: v for k, v in extra.items() if v is not None})
    else:
        print("  (no --params: solver settings and git_sha are not recorded)")

    # The sweep finished when its summary was written, not when it was imported.
    finished = datetime.fromtimestamp(summary_path.stat().st_mtime)
    stamp = finished.strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or (f"{stamp}_{args.label}" if args.label else stamp)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags({
            "imported": "true",
            "imported_from": str(args.results_dir.resolve()),
            "source_finished_at": finished.isoformat(timespec="seconds"),
        })
        mlflow.log_params(params)
        mlflow.log_metrics(_metrics(rows))
        mlflow.log_artifacts(str(args.results_dir))
    print(f"Logged {len(rows)} solves to MLflow ({MLFLOW_EXPERIMENT}/{run_name})")


if __name__ == "__main__":
    main()
