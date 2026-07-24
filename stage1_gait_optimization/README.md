# Stage 1 — Gait Optimisation

Optimal control problem (OCP) for finding an efficient swimming gait for the AMPH quadruped robot.

## Running the OCP

```bash
cd /home/ws/stage1_gait_optimization
python trajopt/run_collocation.py
```

## Experiment Tracking (MLflow)

Results, parameters, and solution files are logged to MLflow.
The server must be started from the `experiment_results/` folder so that
the database and artifact store are created there.

```bash
cd /home/ws/stage1_gait_optimization/experiment_results
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
