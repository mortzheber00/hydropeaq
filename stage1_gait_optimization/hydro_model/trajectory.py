"""Reading and writing OCP solutions (``.npz``).

Files record the robot and whether the state is in tree or reduced coordinates.
Layout of ``X`` (shape ``(nq + nv, N+1)``):

    X[0:3]      base position, world frame
    X[3:7]      base quaternion, scalar-last (x, y, z, w)
    X[7:nq]     joint coordinates  -- theta if coords="reduced", else tree joints
    X[nq:nq+6]  base twist, body frame
    X[nq+6:]    joint velocities

``U`` is ``(n_theta, N)`` joint torques and ``T`` the cycle period. Files
without a ``version`` field are read as amph solutions in tree coordinates.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

SOLUTION_VERSION = 2
_LEGACY_ROBOT = "amph"


def save_solution(path, *, T, X, U, N, nq, robot: str, coords: str = "tree",
                  n_theta: int | None = None, Xc=None) -> Path:
    """Write a solution; ``coords`` is ``"tree"`` or ``"reduced"``.

    ``Xc`` optionally holds the ``(2*nv, N*d)`` collocation states (column
    ``k*d + i``), needed to evaluate interval integrals such as energy exactly.
    """
    if coords not in ("tree", "reduced"):
        raise ValueError(f"coords must be 'tree' or 'reduced', got {coords!r}")
    path = Path(path)
    arrays = dict(
        T=T, X=X, U=U, N=N, nq=nq,
        version=SOLUTION_VERSION, robot=robot, coords=coords,
        n_theta=int(nq - 7 if n_theta is None else n_theta),
    )
    if Xc is not None:
        arrays["Xc"] = np.asarray(Xc)
    np.savez(path, **arrays)
    return path


def load_solution(path) -> dict:
    """Read a solution; v1 files get default metadata."""
    data = np.load(Path(path), allow_pickle=False)
    out = {
        "T": float(data["T"]),
        "X": data["X"],
        "U": data["U"],
        "N": int(data["N"]),
        "nq": int(data["nq"]),
    }
    if "Xc" in data.files:
        out["Xc"] = data["Xc"]
    if "version" in data.files:
        out["version"] = int(data["version"])
        out["robot"] = str(data["robot"])
        out["coords"] = str(data["coords"])
        out["n_theta"] = int(data["n_theta"])
    else:
        out["version"] = 1
        out["robot"] = _LEGACY_ROBOT
        out["coords"] = "tree"
        out["n_theta"] = out["nq"] - 7
    return out


def coords_of(robot) -> str:
    """Coordinate convention of this robot's states: ``"tree"`` or ``"reduced"``."""
    from .coordinate_map import IdentityMap

    return "tree" if isinstance(robot.coord_map, IdentityMap) else "reduced"


def expand_to_tree(robot, X: np.ndarray, nq: int) -> np.ndarray:
    """Convert reduced-coordinate states to tree coordinates (``X`` itself for serial robots)."""
    from .coordinate_map import IdentityMap

    cmap = robot.coord_map
    if isinstance(cmap, IdentityMap):
        return X

    model = robot.model
    K = X.shape[1]
    out = np.zeros((model.nq + model.nv, K))
    for k in range(K):
        theta = X[7:nq, k]
        thd = X[nq + 6:, k]
        out[0:7, k] = X[0:7, k]
        out[7:model.nq, k] = cmap.expand_numeric(theta)
        out[model.nq:model.nq + 6, k] = X[nq:nq + 6, k]
        out[model.nq + 6:, k] = cmap.v_numeric(theta, thd)
    return out
