#!/usr/bin/env python3
"""
SPH boundary particles vs. the ch.4 drag-cylinder model, side by side.

Left panel: the boundary particles FluidSimulator samples on the robot's
collision mesh (Poisson-disk surface sampling, see
splishsplash/GazeboFluidSimulator/FluidSimulator.cpp::publishBoundaryParticles).
Right panel: the same robot's drag-cylinder approximation from
stage1_gait_optimization/hydro_model/visualization.py, at the same neutral
configuration, sharing axes and scale with the left panel.

The boundary particles were captured once from a live run and are checked in
at data/boundary_particles_full_robot.npy (world-frame pool/wall points
already filtered out, and the spawn translation removed so the cloud sits in
the same base_link-relative frame the hydro_model cylinders use, at the zero
joint configuration).

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
  python boundary_vs_cylinder.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401  registers the 'science' matplotlib style
from matplotlib.patches import Patch

# Professional thesis style with real LaTeX text rendering (Computer Modern).
plt.style.use(["science"])
plt.rcParams["text.usetex"] = True

sys.path.insert(0, str(Path(__file__).parents[1]))
from stage1_gait_optimization.hydro_model.robots import load_robot  # noqa: E402
from stage1_gait_optimization.hydro_model.visualization import (  # noqa: E402
    LINK_COLORS,
    _draw_cylinder,
    _link_color,
    _set_equal_aspect,
)

DATA_PATH = Path(__file__).parent / "data" / "boundary_particles_full_robot.npy"
SAVE_PATH = Path(__file__).parents[1] / "docs" / "figures" / "sim" / "boundary_vs_cylinder.pdf"

ELEV, AZIM = 25.0, -60.0


def main() -> None:
    boundary_pts = np.load(DATA_PATH)

    robot = load_robot("amph")

    fig = plt.figure(figsize=(12, 6))
    ax_particles = fig.add_subplot(121, projection="3d")
    ax_cylinders = fig.add_subplot(122, projection="3d")

    ax_particles.scatter(
        boundary_pts[:, 0], boundary_pts[:, 1], boundary_pts[:, 2],
        s=1.5, c="#2B2F38", alpha=0.5, linewidths=0, rasterized=True,
    )

    all_pts = [boundary_pts]
    for name, link in robot.links.items():
        cyl = link.cylinder
        if cyl is None:
            continue
        pts = _draw_cylinder(
            ax_cylinders, cyl.center, cyl.axis_world, cyl.radius, cyl.length,
            _link_color(name),
        )
        all_pts.append(np.array(pts))
    all_pts = np.concatenate(all_pts)

    legend_elements = [
        Patch(facecolor=c, edgecolor="k", label=n.capitalize())
        for n, c in LINK_COLORS.items()
    ]
    ax_cylinders.legend(handles=legend_elements, loc="upper left", fontsize=6)

    ax_particles.set_title("SPH boundary particles (sampled collision mesh)")
    ax_cylinders.set_title(r"Drag cylinder ($r$ = RMS vertex distance from axis)")

    for ax in (ax_particles, ax_cylinders):
        _set_equal_aspect(ax, all_pts)
        ax.set_xlabel("X [m]", fontsize=8)
        ax.set_ylabel("Y [m]", fontsize=8)
        ax.set_zlabel("Z [m]", fontsize=8)
        ax.view_init(elev=ELEV, azim=AZIM)

    fig.suptitle("Full-robot boundary sampling vs. ch.4 drag-cylinder approximation")
    fig.tight_layout()

    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(SAVE_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved {SAVE_PATH}")


if __name__ == "__main__":
    main()
