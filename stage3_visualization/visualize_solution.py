#!/usr/bin/env python3
"""
Replay an OCP solution in MeshCat with full STL meshes.

Usage:
    python visualize_solution.py [solution.npz]

Opens a browser tab at http://localhost:7000.  The trajectory loops
continuously until the script is interrupted.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer

URDF_PATH = Path(__file__).parent.parent / "src" / "amph" / "urdf" / "amph.urdf"
# package://amph/... URIs are resolved relative to this directory.
PACKAGE_DIR = URDF_PATH.parent.parent.parent   # dir that contains src/

SOL_PATH = sys.argv[1] if len(sys.argv) > 1 else "task3_solution.npz"


def main():
    data = np.load(SOL_PATH)
    X = data["X"]
    T = float(data["T"])
    N = int(data["N"])
    nq = int(data["nq"])
    dt = T / N

    print(f"Loaded {SOL_PATH}: N={N}, T={T:.3f}s, nq={nq}")

    model, collision_model, visual_model = pin.buildModelsFromUrdf(
        str(URDF_PATH), package_dirs=[str(PACKAGE_DIR)],
        root_joint=pin.JointModelFreeFlyer(),
    )

    viz = MeshcatVisualizer(model, collision_model, visual_model)
    viz.initViewer(open=True)
    viz.loadViewerModel()

    print("Viewer ready — replaying trajectory (Ctrl-C to stop)...")
    while True:
        for k in range(N + 1):
            viz.display(X[:nq, k])
            time.sleep(dt)


if __name__ == "__main__":
    main()
