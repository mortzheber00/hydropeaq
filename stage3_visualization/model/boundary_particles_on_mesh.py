#!/usr/bin/env python3
"""Plot amph's SPH boundary particles on top of its STL meshes (neutral pose).

The particles in data/boundary_particles_full_robot.npy were captured from a
Gazebo run at the zero pose, in the base frame. To regenerate them:

1. Write a zero-pose solution (npz_path:='' does not disable the replay):

       n_theta = 12; nq = 7 + n_theta; nv = 6 + n_theta
       X = np.zeros((nq + nv, 2)); X[6, :] = 1.0   # identity quaternion
       np.savez("zero_pose_solution.npz", T=1.0, X=X, U=np.zeros((n_theta, 1)),
                N=1, nq=nq, version=2, robot="amph", coords="tree", n_theta=n_theta)

2. Run the simulation and dump the particles once "Boundary particles: N" is printed:

       roslaunch amph swimming_pool.launch gui_required:=false bag_path:='' \\
           npz_path:=/path/to/zero_pose_solution.npz
       gz topic -e /gazebo/swimming_pool/rigids_pos -d 1 > rigids_pos_raw.txt

3. Parse the x/y/z fields, keep z > 0.2, |x| < 0.85, |y| < 0.4 (drops the pool)
   and add (0.5, 0, -0.5) to undo the spawn pose.

Usage:
  python stage3_visualization/model/boundary_particles_on_mesh.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from stage3_visualization.common import thesis_style  # noqa: E402,F401  activates the shared style on import

from stage1_gait_optimization.hydro_model.robots import load_robot  # noqa: E402
from stage1_gait_optimization.hydro_model.visualization import (  # noqa: E402
    _make_urdf_transform_manager,
    _set_equal_aspect,
)

DATA_PATH = Path(__file__).parent / "data" / "boundary_particles_full_robot.npy"
SAVE_PATH = Path(__file__).parents[2] / "docs" / "figures" / "sim" / "boundary_particles_on_mesh.pdf"

ELEV, AZIM = 25.0, -60.0

# Matches the 8 pt legends of run_hydro_validation after LaTeX scaling (8 * 5.91 / 5.0)
LEGEND_PT = 9.5


def main() -> None:
    boundary_pts = np.load(DATA_PATH)

    robot = load_robot("amph")
    q = robot.neutral_config()
    robot.forward_kinematics(q)

    tm = _make_urdf_transform_manager(robot)
    # Set all tree joints; continuous joints are stored as (cos, sin).
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
    # Use explicit zorder (meshes below particles) instead of Axes3D depth sorting
    ax.computed_zorder = False

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tm.plot_visuals("base_link", ax=ax, alpha=0.9, wireframe=False,
                        convex_hull_of_mesh=False)
    for coll in ax.collections:
        coll.set_zorder(1)

    # Small and translucent so the mesh stays visible
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
