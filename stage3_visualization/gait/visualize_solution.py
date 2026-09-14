#!/usr/bin/env python3
"""
Replay an OCP solution in MeshCat with full STL meshes.

Usage:
    python visualize_solution.py [solution.npz]

The robot is read from the solution file.  A closed-chain robot's state is
stored in its reduced coordinates, so it is expanded onto the Pinocchio tree
before display — which is also what makes the linkage loops visibly close.

Opens a browser tab at http://localhost:7000.  The trajectory loops
continuously until the script is interrupted.
"""

import sys
import time
from pathlib import Path

import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer

sys.path.insert(0, str(Path(__file__).parents[2] / "stage1_gait_optimization"))
from hydro_model import get_spec, load_robot  # noqa: E402
from hydro_model.trajectory import expand_to_tree, load_solution  # noqa: E402

SOL_PATH = sys.argv[1] if len(sys.argv) > 1 else "task3_solution.npz"


def main():
    sol = load_solution(SOL_PATH)
    X, T, N, nq = sol["X"], sol["T"], sol["N"], sol["nq"]
    spec = get_spec(sol["robot"])
    dt = T / N

    print(f"Loaded {SOL_PATH}: robot={sol['robot']}, N={N}, T={T:.3f}s, "
          f"nq={nq}, coords={sol['coords']}")

    # Two different package-dir conventions are in play: pinocchio strips the
    # package name itself, so it wants the directory *containing* the package.
    model, collision_model, visual_model = pin.buildModelsFromUrdf(
        str(spec.urdf_path), package_dirs=[str(spec.package_root)],
        root_joint=pin.JointModelFreeFlyer(),
    )

    X_tree = expand_to_tree(load_robot(sol["robot"]), X, nq)

    viz = MeshcatVisualizer(model, collision_model, visual_model)
    viz.initViewer(open=True)
    viz.loadViewerModel()

    print("Viewer ready — replaying trajectory (Ctrl-C to stop)...")
    while True:
        for k in range(N + 1):
            viz.display(X_tree[:model.nq, k])
            time.sleep(dt)


if __name__ == "__main__":
    main()
