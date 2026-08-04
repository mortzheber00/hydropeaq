"""Solution files must stay readable across the format change.

v1 files (written before the format carried robot identity) are the ones
already sitting in the repo and referenced by every stage2/stage3 default
argument, so reading them must be bit-exact.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import REPO_ROOT
from hydro_model import load_robot
from hydro_model.trajectory import expand_to_tree, load_solution, save_solution

V1_FILES = [p for p in (REPO_ROOT / "task3_solution.npz",
                        REPO_ROOT / "task3_guess.npz") if p.exists()]


@pytest.mark.parametrize("path", V1_FILES, ids=lambda p: p.name)
def test_v1_files_load_bit_exactly(path):
    raw = np.load(path, allow_pickle=False)
    got = load_solution(path)
    assert got["version"] == 1
    assert got["robot"] == "amph"
    assert got["coords"] == "tree"
    assert got["n_theta"] == int(raw["nq"]) - 7
    np.testing.assert_array_equal(got["X"], raw["X"])
    np.testing.assert_array_equal(got["U"], raw["U"])
    assert (got["T"], got["N"], got["nq"]) == (
        float(raw["T"]), int(raw["N"]), int(raw["nq"]))


def test_v2_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    nq, nv, N = 15, 14, 6
    X = rng.normal(size=(nq + nv, N + 1))
    U = rng.normal(size=(8, N))
    p = save_solution(tmp_path / "s.npz", T=0.8, X=X, U=U, N=N, nq=nq,
                      robot="body2", coords="reduced", n_theta=8)
    got = load_solution(p)
    assert (got["version"], got["robot"], got["coords"], got["n_theta"]) == (2, "body2", "reduced", 8)
    np.testing.assert_array_equal(got["X"], X)
    np.testing.assert_array_equal(got["U"], U)


def test_bad_coords_rejected(tmp_path):
    with pytest.raises(ValueError, match="coords"):
        save_solution(tmp_path / "x.npz", T=1.0, X=np.zeros((2, 2)), U=np.zeros((1, 1)),
                      N=1, nq=1, robot="amph", coords="nonsense")


def test_expand_to_tree_is_identity_for_serial_robots():
    """Must return the same object, so amph arrays are never touched."""
    robot = load_robot("amph")
    X = np.zeros((robot.nq + robot.nv, 3))
    assert expand_to_tree(robot, X, robot.nq) is X
