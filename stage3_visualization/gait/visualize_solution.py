#!/usr/bin/env python3
"""Replay an OCP solution in MeshCat (http://localhost:7000) until interrupted.

The robot is read from the solution file; reduced states are expanded to the
full tree.

Usage:
  python stage3_visualization/gait/visualize_solution.py task3_solution.npz
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

    # Pinocchio expects the directory containing the package.
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
