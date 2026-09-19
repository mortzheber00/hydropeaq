# Stage 1 — Gait Optimisation

Optimal control problem (OCP) for finding an efficient swimming gait for the AMPH quadruped robot.

## Running the OCP

Run from the repository root, where the stage 3 scripts look for the
`task3_solution.npz` it writes:

```bash
cd /home/ws
python stage1_gait_optimization/trajopt/run_collocation.py
```

## Running the co-design sweep

`codesign/run_codesign.py` solves one OCP per initial gait × cycle period × target speed
and extracts the speed-vs-cost-of-transport Pareto front:

```bash
python stage1_gait_optimization/codesign/run_codesign.py
```

The sweep is configured by constants, not arguments:

| Where | Setting |
|-------|---------|
| `codesign/run_codesign.py` | `SPEED_TARGETS`, `T_GRID_DEFAULT` / `GAIT_T_OVERRIDE` (cycle-period centres), `FREE_T_BAND`, `PARALLEL` / `N_WORKERS`, `WARM_START`, `USE_MLFLOW`, `RUN_LABEL` |
| `codesign/solver.py` | `ROBOT`, `GAITS` (initial gaits), IPOPT options |

Each (gait, period) pair is a chain over ascending speeds, warm-started from the
previous speed; chains run in parallel. The sweep writes one solution per point,
`codesign_summary.json` and `pareto_front.{png,pdf}` to `codesign/codesign_results/`,
and logs to the MLflow experiment `gait_codesign` (start the server first, see below).


## Experiment Tracking (MLflow)

Results, parameters, and solution files are logged to MLflow.
The server must be started from the repository's `experiment_results/` folder so
that the database and artifact store are created there.

```bash
cd /home/ws/experiment_results
mlflow server --host 0.0.0.0 --port 5000 \
              --backend-store-uri sqlite:///mlflow.db \
              --default-artifact-root ./mlruns &
```

Then open **http://localhost:5000** in a browser to view runs.

To stop the server:

```bash
pkill -f "mlflow server"
# if that fails:
ps aux | grep mlflow | grep -v grep | awk '{print $2}' | xargs kill -9
```
To delete artifacts after you deleted the run in ml flow use:

```bash
mlflow gc --backend-store-uri sqlite:///experiment_results/mlflow.db
```
