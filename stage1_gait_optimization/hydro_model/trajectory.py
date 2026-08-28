"""Reading and writing OCP solutions.

A solution file must say which robot it belongs to and which coordinates its
state is expressed in, because a closed-chain robot's state is written in the
reduced actuated coordinates while its Pinocchio tree is much larger.

Layout of ``X`` (shape ``(nq + nv, N+1)``):

    X[0:3]      base position, world frame
    X[3:7]      base quaternion, scalar-last (x, y, z, w)
    X[7:nq]     joint coordinates  -- theta if coords="reduced", else tree joints
    X[nq:nq+6]  base twist, body frame
    X[nq+6:]    joint velocities

``U`` is ``(n_theta, N)`` joint torques and ``T`` the cycle period.

Files written before this module existed carry no ``version`` and are read as
amph solutions in tree coordinates, which is exactly what they are.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

SOLUTION_VERSION = 2
_LEGACY_ROBOT = "amph"


def save_solution(path, *, T, X, U, N, nq, robot: str, coords: str = "tree",
                  n_theta: int | None = None, Xc=None) -> Path:
    """Write a solution.  ``coords`` is ``"tree"`` or ``"reduced"``.

    ``Xc`` is optional: the ``(2*nv, N*d)`` tangent states at the collocation
    points, in the transcription's own column order ``k*d + i``.  Grid states
    alone cannot reproduce any integral the OCP took over an interval — the
    objective's Radau quadrature included — so a solution written without it
    can only be re-measured by resampling, which is what made the old
    grid-node energy figures 30-40% low.  Written when the caller has it;
    files without it stay readable.
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
    """Read a solution, filling in the metadata that v1 files predate."""
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
    """Which convention this robot's states are written in."""
    from .coordinate_map import IdentityMap

    return "tree" if isinstance(robot.coord_map, IdentityMap) else "reduced"


def expand_to_tree(robot, X: np.ndarray, nq: int) -> np.ndarray:
    """Convert a reduced-coordinate state block to tree coordinates.

    Returns ``X`` itself for a robot whose map is the identity, so a serial
    robot's arrays are never copied or perturbed.
    """
    from .coordinate_map import IdentityMap

    cmap = robot.coord_map
    if isinstance(cmap, IdentityMap):
        return X

    model = robot.model
    K = X.shape[1]
    out = np.zeros((model.nq + model.nv, K))
    for k in range(K):
        theta = X[7:nq, k]                 # reduced velocity is [base(6); thetadot]
        thd = X[nq + 6:, k]
        out[0:7, k] = X[0:7, k]
        out[7:model.nq, k] = cmap.expand_numeric(theta)
        out[model.nq:model.nq + 6, k] = X[nq:nq + 6, k]
        out[model.nq + 6:, k] = cmap.v_numeric(theta, thd)
    return out
