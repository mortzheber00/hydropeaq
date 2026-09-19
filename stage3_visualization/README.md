# Stage 3 — Analysis

Thesis figures and the numbers quoted in the thesis, computed from solved gaits and
co-design sweeps. Run every script from the repository root; `--help` lists its options.
Scripts that take `--solution` default to `task3_solution.npz` in the repository root or
to the thesis nominal gait; `--save x.pdf` writes the figures instead of showing them.

✅ works · ❌ does not work

## `common/` — shared helpers

Imported by the scripts below, not run on their own; whether a figure supports a robot is
listed with the script.

- `collocation.py` — samples a solution at its collocation points and integrates over the cycle with the OCP's quadrature.
- `drag_model.py` — numeric per-link port of the OCP's drag model.
- `force_budget.py` — whole-robot forward-force balance, term by term.
- `sweep_io.py` — reads a finished co-design sweep: summary rows paired with their solutions.
- `thesis_style.py` — shared matplotlib style, palette and figure sizes for the thesis.

## `metrics/` — the numbers the thesis quotes

| Script | Description | amph | body2 |
|--------|-------------|:----:|:-----:|
| `nominal_metrics.py` | Every scalar of the nominal-gait results section: energy, COT, torques, limit activity, stroke structure. | ✅ | ❌ |
| `stroke_asymmetry.py` | How the stroke earns thrust: power-stroke duty, submersion and speed split, frozen-leg counterfactual. | ✅ | ❌ |

## `gait/` — one solved gait

| Script | Description | amph | body2 |
|--------|-------------|:----:|:-----:|
| `plot_solution.py` | Joint angles, base state, foot positions, and the swim-cycle GIF. | ✅ | ✅ |
| `visualize_solution.py` | Replays a solution in MeshCat with the full STL meshes. | ✅ | ✅ |
| `plot_solution_legs.py` | Sagittal leg stick figures and foot paths in the base frame, one column per solution. | ✅ | ✅ |
| `plot_base_motion.py` | Stroboscopic side view of the swimming body, plus base velocity and position traces over the cycle. | ✅ | ✅ |
| `plot_limit_activity.py` | Angle, rate, torque and bandwidth limits against the solution over one cycle, and which bind across a sweep. | ✅ | ✅ |
| `plot_power_flow.py` | Where the cycle's mechanical energy goes vs. what the optimizer is charged for. | ✅ | ✅ |
| `plot_convergence.py` | IPOPT convergence curves from MLflow, one per initial guess. | ✅ | ✅ |

## `thrust/` — how the stroke makes thrust

| Script | Description | amph | body2 |
|--------|-------------|:----:|:-----:|
| `plot_thrust_budget.py` | Whole-robot forward-force budget over one cycle. | ✅ | ❌ |
| `plot_thrust_attribution.py` | Per-link attribution of that budget: which links push. | ✅ | ❌ |
| `plot_stroke_benefit.py` | What each leg gains by stroking vs. the drag it would suffer held still. | ✅ | ❌ |
| `gait_diagnostics.py` | Where a leg's net thrust comes from: foot velocity, thrust and cumulative impulse over the cycle. | ✅ | ✅ |
| `thrust_heatmap.py` | Best-case thrust and sweep direction over a leg's thigh/calf poses. | ✅ | ❌ |
| `hind_parked_rollout.py` | Forward-simulates the gait with the hind legs held still, to show what they contribute. | ✅ | ❌ |
| `hind_workspace.py` | A leg's reachable workspace vs. the foot path of a guess or solution. | ✅ | ✅ |

## `speed_sweep/` — trends across the co-design sweep

| Script | Description | amph | body2 |
|--------|-------------|:----:|:-----:|
| `plot_cot_vs_period.py` | Cost of transport vs. cycle period at a fixed speed, one curve per gait. | ✅ | ✅ |
| `plot_mechanism_vs_speed.py` | Why the cost of transport rises with speed, mechanism by mechanism. | ✅ | ❌ |
| `plot_structure_vs_speed.py` | How duty factor, leg phasing and stride frequency change with speed along the Pareto front. | ✅ | ❌ |

The speed-sweep scripts read a `run_codesign.py` sweep, and `run_codesign.py` itself
hardcodes `amph`.

## `sph/` — flow in the SPH simulation

These scripts read the particle export of a Gazebo + SPlisHSPlasH run from
`sph_output/vtk/` (see [`src/README.md`](../src/README.md)) instead of a solution file.

| Script | Description | amph | body2 |
|--------|-------------|:----:|:-----:|
| `pool_viewer.py` | Interactive 3D view of the pool, robot and particles coloured by speed; saves single frames as PDF or a range as mp4/gif. | ✅ | ❌ |
| `leg_flow_slices.py` | In-plane speed or vorticity with streamlines on planes through the legs (side or top view), as video, stills or one gait cycle in 8 snapshots. | ✅ | ❌ |
| `wake_spacetime.py` | Space-time diagram of the wake travelling from the front to the hind legs, per body side. | ✅ | ❌ |

The Gazebo simulation does not run for BODY2, so there is no SPH export for it.

## `model/` — model figures

| Script | Description | amph | body2 |
|--------|-------------|:----:|:-----:|
| `boundary_particles_on_mesh.py` | SPH boundary particles overlaid on the robot's STL mesh. | ✅ | ❌ |

`thrust_heatmap.py`, `boundary_particles_on_mesh.py` and `nominal_metrics.py` hardcode
`amph`. Most other ❌ entries fail on BODY2's closed-chain legs: amph joint names, or
3-joint serial-leg geometry.
