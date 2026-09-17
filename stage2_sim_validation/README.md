# Stage 2 — Simulation validation

Checks that the hydrodynamic model and the gaits optimized with it hold up: calibrate
the coefficients against SPH recordings, sanity-check the symbolic model, compare Gazebo
replays with the OCP, and test how robust the OCP results are. Run every script from the
repository root; `--help` lists its options.

✅ works · ❌ does not work

## `hydro_calibration/` — fit the hydro coefficients to recordings

| Script | Description | amph | body2 |
|--------|-------------|:----:|:-----:|
| `sweep_hydro_params.py` | Fits the six drag/added-mass coefficients so the model's base motion matches one or more SPH rosbags. | ✅ | ❌ |
| `hydro_fit_sensitivity.py` | Scans each coefficient across its range around the fit to show which ones the recording can actually identify. | ✅ | ❌ |
| `simulate_ocp.py` | Forward-simulates the base with the OCP's joint trajectory prescribed, for comparing against a bag. Helpful if coefficients changed to evaluate on the model| ✅ | ✅ |

`sweep_hydro_params.py` and `hydro_fit_sensitivity.py` hardcode `amph`.

## `model_checks/` — symbolic-model sanity checks, no recording

| Script | Description | amph | body2 |
|--------|-------------|:----:|:-----:|
| `run_hydro_validation.py` | Evaluates buoyancy, drag and added mass at a sample pose and draws the cylinder approximation. | ✅ | ✅ |
| `dynamics_diagnostics.py` | Breaks down the inertias and forces (drag, buoyancy, added mass) along a saved solution. | ✅ | ✅ |

## `gazebo_replay/` — simulator vs. OCP

| Script | Description | amph | body2 |
|--------|-------------|:----:|:-----:|
| `validate_sim.py` | Compares joint and base tracking from a Gazebo/SPH rosbag with the OCP solution it replayed, as thesis figures. | ✅ | ✅ |

## `sensitivity_and_robustness/` — how robust are the OCP results?

| Script | Description | amph | body2 |
|--------|-------------|:----:|:-----:|
| `n_sweep_continuation.py` | Re-solves the OCP on a ladder of grid sizes N, warm-started rung to rung; logs to MLflow. | ✅ | ✅ |
| `plot_mesh_convergence.py` | Mesh-refinement figures from a ladder: trajectory and error convergence with N, as evidence for the chosen grid. | ✅ | ✅ |
| `guess_multistart.py` | Solves the OCP from several initial gaits to see how far apart the local optima land. | ✅ | ❌ |
| `plot_multistart.py` | Distance matrix between the multistart optima. | ✅ | ✅ |
| `plot_multistart_timing.py` | Gait diagram of the multistart optima: when each leg pushes. | ✅ | ❌ |
| `hydro_sensitivity.py` | Perturbs each calibrated coefficient ±25 % and re-solves the OCP, warm-started from a nominal solution. | ✅ | ✅ |
| `plot_hydro_sensitivity.py` | Tornado chart of the cost-of-transport change per perturbed coefficient. | ✅ | ✅ |
| `plot_hydro_gaits.py` | Foot-path comparison of the nominal and ±25 % solutions, one panel per coefficient. | ✅ | ❌ |
| `formulation_metrics.py` | Size of the OCP's two modelling approximations (frozen submersion ratio, tangent rate) on a solution. | ✅ | ❌ |

- `guess_multistart.py`: the default initial gaits (LSPG25, LSPG33, TLPG50) are written as amph thigh/calf angles.
- `plot_hydro_gaits.py`: looks up amph's `Thigh`/`Calf` joint names.
- `plot_multistart_timing.py`: passes reduced coordinates to the full-tree kinematics.
- `formulation_metrics.py`: its symbolic check assumes the tree state layout.
