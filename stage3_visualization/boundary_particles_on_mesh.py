#!/usr/bin/env python3
"""
SPH boundary particles overlaid on the robot's true STL mesh.

The boundary particles FluidSimulator samples on the robot's collision mesh
(Poisson-disk surface sampling, see
splishsplash/GazeboFluidSimulator/FluidSimulator.cpp::publishBoundaryParticles)
are drawn in red on top of the URDF visual meshes at the same neutral
configuration.

The boundary particles were captured once from a live run and are checked in
at data/boundary_particles_full_robot.npy (world-frame pool/wall points
already filtered out, and the spawn translation removed so the cloud sits in
the base_link-relative frame the hydro_model uses, at the zero joint
configuration).

swimming_pool.launch always starts sph_replay's replay_trajectory.py, which
drives every joint toward frame 0 of whatever ~npz_path resolves to (default:
/home/ws/task3_solution.npz -- not the zero pose). Passing npz_path:='' on
the roslaunch command line does NOT disable it: roslaunch treats an empty
CLI override as not given and falls back to the arg's default, so the replay
runs anyway. To sample at the true zero pose, point npz_path at a synthetic
solution file whose actuated coordinates are all zero instead -- one array
per key expected by hydro_model/trajectory.py::load_solution():

    n_theta = 12; nq = 7 + n_theta; nv = 6 + n_theta
    X = np.zeros((nq + nv, 2)); X[6, :] = 1.0   # identity quaternion
    np.savez("zero_pose_solution.npz", T=1.0, X=X, U=np.zeros((n_theta, 1)),
             N=1, nq=nq, version=2, robot="amph", coords="tree",
             n_theta=n_theta)

then:

    roscore &
    roslaunch amph swimming_pool.launch gui_required:=false bag_path:='' \\
        npz_path:=/path/to/zero_pose_solution.npz
    # once "Boundary particles: N" has printed:
    gz topic -e /gazebo/swimming_pool/rigids_pos -d 1 > rigids_pos_raw.txt

then parse the text dump's repeated "x:"/"y:"/"z:" fields into an (N,3) array,
keep only points with z > 0.2, |x| < 0.85, |y| < 0.4 (drops the pool floor
and walls, which is everything outside the robot's own footprint), and add
(0.5, 0, -0.5) to undo the robot's spawn pose <pose>-0.5 0 0.5 0 0 0</pose>
in src/amph/worlds/swimming_pool.world.

Usage:
  python boundary_particles_on_mesh.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import thesis_style  # noqa: E402,F401  activates the shared style on import

from stage1_gait_optimization.hydro_model.robots import load_robot  # noqa: E402
from stage1_gait_optimization.hydro_model.visualization import (  # noqa: E402
    _make_urdf_transform_manager,
    _set_equal_aspect,
)

DATA_PATH = Path(__file__).parent / "data" / "boundary_particles_full_robot.npy"
SAVE_PATH = Path(__file__).parents[1] / "docs" / "figures" / "sim" / "boundary_particles_on_mesh.pdf"

ELEV, AZIM = 25.0, -60.0

# What a point of legend text is worth on the page depends on how far LaTeX
# scales the figure, so the size is set relative to the sibling robot figures
# rather than in isolation: run_hydro_validation's are 8 pt on a 5.0 in page,
# this one is 5.91 in wide once savefig's tight bbox has cropped it, and
# 8 * 5.91 / 5.0 puts the two at the same size in the document.
LEGEND_PT = 9.5


def main() -> None:
    boundary_pts = np.load(DATA_PATH)

    robot = load_robot("amph")
    q = robot.neutral_config()
    robot.forward_kinematics(q)

    tm = _make_urdf_transform_manager(robot)
    # Every tree joint has to be set, not just the actuated ones: on a
    # closed-chain robot the passive joints carry the loop closure, and leaving
    # them at zero tears the legs off their pins.  The angle comes from the
    # Pinocchio configuration by joint index -- for a continuous joint that
    # configuration is a (cos, sin) pair rather than an angle.
    for jid in range(1, robot.model.njoints):
        joint = robot.model.joints[jid]
        if joint.nq == 1:
            angle = float(q[joint.idx_q])
        elif joint.nq == 2:
            angle = float(np.arctan2(q[joint.idx_q + 1], q[joint.idx_q]))
        else:  # free-flyer base
            continue
        tm.set_joint(robot.model.names[jid], angle)
    for v in tm.visuals:
        v.color = [0.6, 0.62, 0.66, 1.0]

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    # Axes3D normally overwrites every artist's zorder with a depth ranking
    # computed per collection, so a link mesh whose mean depth is nearest hides
    # the whole particle cloud (the front-right hip did exactly that).  Turning
    # that off keeps the draw order we ask for: meshes first, particles on top.
    ax.computed_zorder = False

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tm.plot_visuals("base_link", ax=ax, alpha=0.9, wireframe=False,
                        convex_hull_of_mesh=False)
    for coll in ax.collections:
        coll.set_zorder(1)

    # The particles sample the same surface the mesh draws, so they hide it
    # unless they stay small and semi-transparent.
    ax.scatter(
        boundary_pts[:, 0], boundary_pts[:, 1], boundary_pts[:, 2],
        s=1.5, c=thesis_style.PALETTE[2], alpha=0.45, linewidths=0, rasterized=True,
        depthshade=False, zorder=5,
    )

    ax.legend(
        handles=[
            Line2D([], [], marker="s", linestyle="none", color="#9A9EA8",
                   markersize=6, label="STL mesh"),
            Line2D([], [], marker="o", linestyle="none", color=thesis_style.PALETTE[2],
                   markersize=3, label="SPH boundary particles"),
        ],
        loc="upper left", fontsize=LEGEND_PT,
    )

    _set_equal_aspect(ax, boundary_pts)
    ax.set_xlabel("X [m]", fontsize=8)
    ax.set_ylabel("Y [m]", fontsize=8)
    ax.set_zlabel("Z [m]", fontsize=8)
    ax.view_init(elev=ELEV, azim=AZIM)

    fig.tight_layout()

    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(SAVE_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved {SAVE_PATH}")


if __name__ == "__main__":
    main()
