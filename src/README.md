# ROS packages and launch files

The catkin packages used to run the robots in Gazebo with the SPlisHSPlasH fluid plugin.
Run every command inside the [development container](../README.md#development-container),
where ROS and the workspace are already sourced.

| Package | Contents |
|---------|----------|
| `amph/` | AMPH URDF, meshes, worlds and launch files |
| `BODY2/` | BODY2 URDF, meshes, worlds and launch files |
| `sph_replay/` | Replays an OCP solution (`.npz`) in Gazebo and generates the SPH boundary meshes |
| `gazebo_ros_pkgs/` | NRP fork of `gazebo_ros`, cloned by the container setup |

> **Note:** the Gazebo simulation currently does **not work for BODY2**. Use `amph` for
> simulator runs.

## Workflow

1. **Solve a gait** (see [Getting started](../README.md#getting-started)). This writes
   `task3_solution.npz` to the repository root.
2. **Launch the simulation** and let it replay the gait:

   ```bash
   roslaunch amph swimming_pool.launch npz_path:=/home/ws/task3_solution.npz
   ```

   The fluid plugin usually generates the SPH boundary meshes itself. If the launch fails
   because they are missing (this can happen on the first launch), generate them once and
   launch again:

   ```bash
   rosrun sph_replay generate_boundary_obj.py amph
   ```

3. **Compare the recording with the OCP** using the scripts in
   [`stage2_sim_validation/gazebo_replay/`](../stage2_sim_validation/), which read
   `/home/ws/sim_log.bag` by default.

## Launch files

### `amph`

| Launch file | What it does |
|-------------|--------------|
| `swimming_pool.launch` | Gazebo with the SPH pool, replays a gait from an `.npz` file and records a rosbag. |
### `BODY2` (Gazebo currently not working)

| Launch file | What it does |
|-------------|--------------|
| `swimming_pool.launch` | Counterpart of the amph pool launch: SPH pool, gait replay, rosbag. |
| `joint_prescription.launch` | Replays the gait in a world with only the robot (no fluid, no gravity), for inspecting the joint motion. Starts paused. |
| `gazebo.launch` | Spawns the URDF in an empty world. |
| `display.launch` | Shows the URDF in RViz with joint sliders. |

### Arguments of the replay launch files

`swimming_pool.launch` (both robots) and `BODY2/joint_prescription.launch` take:

| Argument | Default | Meaning |
|----------|---------|---------|
| `npz_path` | `/home/ws/task3_guess.npz` (amph), `/home/ws/task3_solution.npz` (BODY2) | Solution to replay; `''` disables the replay |
| `n_repeat` | `3` (`10` for `joint_prescription`) | Number of gait cycles to replay |
| `bag_path` | `/home/ws/sim_log.bag` (`''` for `joint_prescription`) | Output rosbag of joint and model states; `''` disables recording |
| `gui_required` | `true` | Start the Gazebo client with the fluid visualization (`swimming_pool` only) |

Note that the amph default replays the *initial guess*, so pass `npz_path` to replay the
solved gait.

## Example commands

```bash
# Replay the solved gait for 5 cycles and record to a custom bag
roslaunch amph swimming_pool.launch npz_path:=/home/ws/task3_solution.npz \
    n_repeat:=5 bag_path:=/home/ws/sph_output/run1.bag

# Headless run without the Gazebo window
roslaunch amph swimming_pool.launch npz_path:=/home/ws/task3_solution.npz gui_required:=false

# Only the fluid world, no replay and no recording
roslaunch amph swimming_pool.launch npz_path:='' bag_path:=''

# Inspect the recorded bag
rosbag info /home/ws/sim_log.bag
```

After editing the SPlisHSPlasH plugin, rebuild it as described in
[the main README](../README.md#building-and-installing-the-splishsplash-gazebo-plugin).
